import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from tools.ai.torch_utils import (L1_Loss, L2_Loss, gap2d, label_smoothing, make_cam, resize_tensor,
                                  set_trainable_layers)
from core.networks import *


def to_2d(x):
      return x[..., None, None]


class SingleStageModel(Backbone):
  def __init__(
    self,
    model_name,
    num_classes=20,
    mode='fix',
    dilated=True,
    strides=None,
    regularization=None,
    trainable_stem=False,
    trainable_backbone=True,
    use_group_norm=False,
    use_saliency_head=False,
    criterion_c=None,
    criterion_s=None,
    criterion_b=None,
  ):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_backbone=trainable_backbone,
    )

    in_features = self.out_features
    norm_fn = group_norm if use_group_norm else nn.BatchNorm2d

    self.aspp = ASPP(in_features, output_stride=16, norm_fn=norm_fn)
    self.decoder = Decoder(num_classes + 1, 256, norm_fn)

    self.num_classes = num_classes
    self.regularization = regularization
    self.use_saliency_head = use_saliency_head

    cin = self.out_features
    self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)
    self.initialize([self.classifier])

    self.from_scratch_layers.extend([self.classifier] + list(self.aspp.modules()) + list(self.decoder.modules()))

    if use_saliency_head:
      self.saliency_head = nn.Conv2d(num_classes, 1, 1)
      self.initialize([self.saliency_head])
      self.from_scratch_layers.append(self.saliency_head)

    self.criterion_c = criterion_c or torch.nn.MultiLabelSoftMarginLoss()
    self.criterion_s = criterion_s or torch.nn.CrossEntropyLoss(ignore_index=255)
    # self.criterion_b = criterion_b or torch.nn.MultiLabelSoftMarginLoss()
    self.criterion_b = criterion_b or L1_Loss

  def forward_features(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    features_s2 = x

    x = self.stage3(x)
    x = self.stage4(x)
    features = self.stage5(x)

    return features, features_s2

  def forward(self, x, with_cam=False, with_mask=True):
    features, features_s2 = self.forward_features(x)
    outputs = self.classification_branch(features, with_cam=with_cam)

    if with_mask:
      return (*outputs, self.segmentation_branch(features, features_s2))

    return outputs

  def segmentation_branch(self, features, features_s2):
    features = self.aspp(features)
    features = self.decoder(features, features_s2)

    return features

  def classification_branch(self, features, with_cam=False):
    if with_cam:
      features = self.classifier(features)
      logits = gap2d(features)
      return logits, features
    else:
      features = gap2d(features, keepdims=True)
      logits = self.classifier(features).view(-1, self.num_classes)
      return logits

  def train_step(self, images, targets, bg_t=0.05, fg_t=0.3, resize_align_corners=None, ls=0.0, w_s=1.0, w_u=1.0, w_b=1.0):
    sizes = images.shape[2:]

    # Forward
    features_s5, features_s2 = self.forward_features(images)

    logits, features = self.classification_branch(features_s5, with_cam=True)
    logits_s = self.segmentation_branch(features_s5, features_s2)
    logits_s_resized = resize_tensor(logits_s, sizes, align_corners=resize_align_corners)

    # (1) Classification loss.
    loss_c = self.criterion_c(logits, label_smoothing(targets, ls))

    # (2) Segmentation loss.
    pseudo_masks = self._get_pseudo_label(features, targets, sizes, bg_t, fg_t, resize_align_corners)

    loss_s = self.criterion_s(logits_s_resized, pseudo_masks)

    # (3) BG loss.
    pseudo_saliencies = (1 - torch.softmax(logits_s, dim=1)[:, 0:1])

    # cams = make_cam(features, global_norm=True, inplace=False)
    if self.use_saliency_head:
      saliency_logits = self.saliency_head(features)
    else:
      saliency_logits = features.sum(dim=1, keepdim=True)

    saliency_logits = resize_tensor(saliency_logits, pseudo_saliencies.shape[2:], align_corners=True)
    loss_b = self.criterion_b(saliency_logits, pseudo_saliencies)

    # (4) Uncertain loss. (Push outputs for classes not in `targets` down.)
    y_n = to_2d(F.pad(1 - targets, (1, 0), value=0.0))  # Mark background=0 (occurs). # B, C
    pixels_u = pseudo_masks == 255  # [B, H, W]
    loss_u = torch.clamp(1 - torch.sigmoid(logits_s_resized), min=0.0005, max=0.9995)
    loss_u = -y_n * torch.log(loss_u)  # [A, B]
    loss_u = loss_u.sum(dim=1) / y_n.sum(dim=1)
    loss_u = loss_u[pixels_u].mean()

    loss = loss_c + w_s * loss_s + w_u * loss_u + w_b * loss_b

    losses_values = {
      "loss_c": loss_c,
      "loss_s": loss_s,
      "loss_b": loss_b,
      "loss_u": loss_u,
    }

    return loss, losses_values

  def _get_pseudo_label(self, cams, targets, sizes, bg_t=0.05, fg_t=0.4, resize_align_corners=None):
    # cams = cams.detach()
    cams = cams.detach() * to_2d(targets)
    cams = make_cam(cams, inplace=True, global_norm=False)
    cams = resize_tensor(cams, sizes, align_corners=resize_align_corners)

    saliency, masks = cams.max(dim=1)
    bg_conf = saliency < bg_t
    fg_conf = saliency >= fg_t

    masks += 1  # shift to make room for bg class.
    masks[bg_conf] = 0
    masks[~bg_conf & ~fg_conf] = 255

    return masks

if __name__ == "__main__":
  DEVICE = "cuda"

  model = SingleStageModel(
    "resnet50", # args.architecture,
    21, # train_dataset.info.num_classes,
    mode="fix", # mode=args.mode,
    dilated=True, # dilated=args.dilated,
    regularization=None, # regularization=args.regularization,
    trainable_stem=True, # trainable_stem=args.trainable_stem,
    trainable_backbone=True,
    criterion_c=torch.nn.MultiLabelSoftMarginLoss().to(DEVICE),
    criterion_s=torch.nn.CrossEntropyLoss(ignore_index=255).to(DEVICE),
  )
