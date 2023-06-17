# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

from tools.ai.torch_utils import set_layers_require_grad, gap2d, resize_for_tensors

from . import ccam, regularizers
from .aff_utils import PathIndex
from .deeplab_utils import ASPP, Decoder
from .mcar import mcar_resnet50, mcar_resnet101
from .puzzle_utils import merge_features, tile_features


class FixedBatchNorm(nn.BatchNorm2d):

  def forward(self, x):
    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)


def group_norm(features):
  return nn.GroupNorm(4, features)


#######################################################################


def build_backbone(name, dilated, strides, norm_fn, weights='imagenet'):
  if dilated:
    dilation, dilated = 4, True
  else:
    dilation, dilated = 2, False

  if not strides:
    strides = [1, 2, 2, 1]

  if 'resnet38d' == name:
    from .arch_resnet import resnet38d
    out_features = 4096

    model = resnet38d.ResNet38d()
    state_dict = resnet38d.convert_mxnet_to_torch('./experiments/models/resnet_38d.params')
    model.load_state_dict(state_dict, strict=True)

    stage1 = nn.Sequential(model.conv1a, model.b2)
    stage2 = nn.Sequential(model.b2_1, model.b2_2, model.b3, model.b3_1, model.b3_2)
    stage3 = nn.Sequential(model.b4, model.b4_1, model.b4_2, model.b4_3, model.b4_4, model.b4_5)
    stage4 = nn.Sequential(model.b5, model.b5_1, model.b5_2)
    stage5 = nn.Sequential(model.b6, model.b7, model.bn7, nn.ReLU())
  else:
    out_features = 2048

    if 'resnet' in name:
      from .arch_resnet import resnet
      model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[name], strides=strides, batch_norm_fn=norm_fn)

      if weights == 'imagenet':
        print(f'loading weights from {resnet.urls_dic[name]}')
        state_dict = model_zoo.load_url(resnet.urls_dic[name])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        model.load_state_dict(state_dict)
    elif 'resnest' in name:
      from .arch_resnest import resnest

      model_fn = getattr(resnest, name)
      model = model_fn(pretrained=weights == 'imagenet', dilated=dilated, dilation=dilation, norm_layer=norm_fn)

      del model.avgpool
      del model.fc
    elif 'res2net' in name:
      from .res2net import res2net_v1b

      model_fn = getattr(res2net_v1b, name)
      model = model_fn(pretrained=weights == 'imagenet', strides=strides, norm_layer=norm_fn)

      del model.avgpool
      del model.fc

    if weights and weights != 'imagenet':
      print(f'loading weights from {weights}')
      checkpoint = torch.load(weights, map_location="cpu")
      model.load_state_dict(checkpoint['state_dict'], strict=False)

    stage1 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    stage2 = nn.Sequential(model.layer1)
    stage3 = nn.Sequential(model.layer2)
    stage4 = nn.Sequential(model.layer3)
    stage5 = nn.Sequential(model.layer4)

  return out_features, model, (stage1, stage2, stage3, stage4, stage5)


class Backbone(nn.Module):

  def __init__(
    self,
    model_name,
    weights='imagenet',
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
  ):
    super().__init__()

    self.mode = mode
    self.trainable_stem = trainable_stem
    self.not_training = []
    self.from_scratch_layers = []

    if mode == 'normal':
      self.norm_fn = nn.BatchNorm2d
    elif mode == 'fix':
      self.norm_fn = FixedBatchNorm
    else:
      raise ValueError(f'Unknown mode {mode}. Must be `normal` or `fix`.')

    out_features, model, stages = build_backbone(
      name=model_name, dilated=dilated, strides=strides, norm_fn=self.norm_fn, weights=weights
    )

    self.model = model
    self.out_features = out_features
    self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = stages

    if self.mode == "fix":
      set_layers_require_grad(model, torch.nn.BatchNorm2d)
      self.not_training.extend([
        m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
      ])

    if not self.trainable_stem:
      self.not_training.extend([self.stage1])

  def initialize(self, modules):
    for m in modules:
      if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        torch.nn.init.kaiming_normal_(m.weight)

      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def get_parameter_groups(self, exclude_partial_names=(), with_names=False):
    names = ([], [], [], [])
    groups = ([], [], [], [])

    for name, param in self.named_parameters():
      if param.requires_grad:
        for p in exclude_partial_names:
          if p in name:
            continue

        if 'model' in name:
          if 'weight' in name:
            names[0].append(name)
            groups[0].append(param)
          else:
            names[1].append(name)
            groups[1].append(param)

        # scracthed weights
        else:
          if 'weight' in name:
            names[2].append(name)
            groups[2].append(param)
          else:
            names[3].append(name)
            groups[3].append(param)

    if with_names:
      return groups, names

    return groups


class Classifier(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    mode='fix',
    dilated=False,
    strides=(2, 2, 2, 1),
    regularization=None,
    trainable_stem=True
  ):
    super().__init__(model_name, mode=mode, dilated=dilated, strides=strides, trainable_stem=trainable_stem)

    self.num_classes = num_classes
    self.regularization = regularization

    cin = self.out_features

    if not regularization or regularization.lower() == 'none':
      self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)
    elif regularization.lower() in ('kernel_usage', 'ku'):
      self.classifier = regularizers.Conv2dKU(cin, num_classes, 1, bias=False)
    elif regularization.lower() in ('minmax', 'minmaxcam'):
      self.classifier = regularizers.MinMaxConv2d(cin, num_classes, 1, bias=False)
    else:
      raise ValueError(f'Unknown regularization strategy {regularization}.')

    self.from_scratch_layers.extend([self.classifier])
    self.initialize([self.classifier])

  def train(self, mode=True):
    super().train(mode)
    for m in self.not_training:
      m.eval()
    return self

  def forward(self, x, with_cam=False):
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    if with_cam:
      features = self.classifier(x)
      logits = gap2d(features)
      return logits, features
    else:
      x = gap2d(x, keepdims=True)
      logits = self.classifier(x).view(-1, self.num_classes)
      return logits


class CCAM(Backbone):

  def __init__(
    self,
    model_name,
    weights='imagenet',
    mode='fix',
    dilated=False,
    strides=(2, 2, 2, 1),
    trainable_stem=True,
    stage4_out_features=1024
  ):
    super().__init__(
      model_name, weights=weights, mode=mode, dilated=dilated, strides=strides, trainable_stem=trainable_stem
    )

    self.ac_head = ccam.Disentangler(stage4_out_features + self.out_features)
    self.from_scratch_layers += [self.ac_head]

  def forward(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x1 = self.stage4(x)
    x2 = self.stage5(x1)

    feats = torch.cat([x2, x1], dim=1)

    return self.ac_head(feats)

  def get_parameter_groups(self):
    groups = ([], [], [], [])
    for m in self.modules():
      if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):
        if m.weight.requires_grad:
          if m in self.from_scratch_layers:
            groups[2].append(m.weight)
          else:
            groups[0].append(m.weight)

        if m.bias is not None and m.bias.requires_grad:
          if m in self.from_scratch_layers:
            groups[3].append(m.bias)
          else:
            groups[1].append(m.bias)
    return groups


class AffinityNet(Backbone):

  def __init__(self, model_name, path_index=None, mode='fix', dilated=False, strides=(2, 2, 2, 1)):
    super().__init__(model_name, mode=mode, dilated=dilated, strides=strides)

    in_features = self.out_features

    if '50' in model_name:
      fc_edge1_features = 64
    else:
      fc_edge1_features = 128

    self.fc_edge1 = nn.Sequential(
      nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.ReLU(inplace=True),
    )
    self.fc_edge2 = nn.Sequential(
      nn.Conv2d(256, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.ReLU(inplace=True),
    )
    self.fc_edge3 = nn.Sequential(
      nn.Conv2d(512, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge4 = nn.Sequential(
      nn.Conv2d(1024, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge5 = nn.Sequential(
      nn.Conv2d(in_features, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

    self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
    self.edge_layers = nn.ModuleList(
      [self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6]
    )

    if path_index is not None:
      self.path_index = path_index
      self.n_path_lengths = len(self.path_index.path_indices)
      for i, pi in enumerate(self.path_index.path_indices):
        self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))

  def train(self, mode=True):
    super().train(mode)
    self.backbone.eval()

  def forward(self, x, with_affinity=False):
    x1 = self.stage1(x).detach()
    x2 = self.stage2(x1).detach()
    x3 = self.stage3(x2).detach()
    x4 = self.stage4(x3).detach()
    x5 = self.stage5(x4).detach()

    edge1 = self.fc_edge1(x1)
    edge2 = self.fc_edge2(x2)
    edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
    edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
    edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

    edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

    if with_affinity:
      return edge, self.to_affinity(torch.sigmoid(edge))
    else:
      return edge

  def get_edge(self, x, image_size=512, stride=4):
    feat_size = (x.size(2) - 1) // stride + 1, (x.size(3) - 1) // stride + 1

    H, W = x.shape[2:]
    x = F.pad(x, [0, max(image_size - H, 0), 0, max(image_size - W, 0)])

    edge_out = self.forward(x)
    edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
    edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)

    return edge_out

  """
    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)
    """

  def to_affinity(self, edge):
    aff_list = []
    edge = edge.view(edge.size(0), -1)

    for i in range(self.n_path_lengths):
      ind = self._buffers["path_indices_" + str(i)]
      ind_flat = ind.view(-1)
      dist = torch.index_select(edge, dim=-1, index=ind_flat)
      dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
      aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
      aff_list.append(aff)
    aff_cat = torch.cat(aff_list, dim=1)
    return aff_cat


class DeepLabV3Plus(Backbone):

  def __init__(self, model_name, num_classes=21, mode='fix', dilated=False, strides=None, use_group_norm=False):
    super().__init__(model_name, mode=mode, dilated=dilated, strides=strides)

    if use_group_norm:
      norm_fn_for_extra_modules = group_norm
    else:
      norm_fn_for_extra_modules = self.norm_fn

    in_features = self.out_features

    self.aspp = ASPP(in_features, output_stride=16, norm_fn=norm_fn_for_extra_modules)
    self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)

  def forward(self, x, with_cam=False):
    inputs = x

    x = self.stage1(x)
    x = self.stage2(x)
    x_low_level = x

    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    x = self.aspp(x)
    x = self.decoder(x, x_low_level)
    x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

    return x
