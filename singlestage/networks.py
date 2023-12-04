from typing import Tuple, Union
import torch.nn as nn

from core.networks import *
from tools.ai.torch_utils import gap2d, resize_tensor


class CSRM(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    num_classes_segm=None,
    mode='fix',
    dilated=False,
    strides=None,
    regularization=None,
    trainable_stem=False,
    trainable_backbone=True,
    use_group_norm=False,
    use_sal_head=False,
    use_rep_head=True,
    rep_output_dim=256,
    dropout: Union[float, Tuple[float, float]] = 0.1,
  ):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_backbone=True,  # This will be performed up ahead.
    )

    cin = self.out_features
    norm_fn = group_norm if use_group_norm else nn.BatchNorm2d

    self.num_classes = num_classes
    self.num_classes_segm = num_classes_segm or num_classes + 1
    self.regularization = regularization
    self.trainable_backbone = trainable_backbone
    self.use_sal_head = use_sal_head
    self.use_rep_head = use_rep_head
    self.dropout = [dropout, dropout] if isinstance(dropout, float) else dropout

    ## Pretrained parameters
    # self.stage1/stage5 = ...
    self.classifier = nn.Conv2d(cin, self.num_classes, 1, bias=False)

    ## Scratch parameters
    self.aspp = ASPP(cin, output_stride=16, norm_fn=norm_fn)

    self.project = nn.Sequential(
      nn.Conv2d(256, 48, 1, bias=False),
      norm_fn(48),
      nn.ReLU(inplace=True),
    )

    self.decoder = nn.Sequential(
      nn.Conv2d(304, 256, 3, padding=1, bias=False),
      norm_fn(256),
      nn.ReLU(inplace=True),
      nn.Dropout(self.dropout[0]),
      nn.Conv2d(256, 256, 3, padding=1, bias=False),
      norm_fn(256),
      nn.ReLU(inplace=True),
      nn.Dropout(self.dropout[1]),
      nn.Conv2d(256, self.num_classes_segm, 1),
    )

    self.from_scratch_layers += [*self.aspp.modules(), *self.project.modules(), *self.decoder.modules()]

    if use_sal_head:
      self.saliency_head = nn.Conv2d(304, 1, 1)
      self.from_scratch_layers += [self.saliency_head]

    if use_rep_head:
      self.representation = nn.Sequential(
        nn.Conv2d(304, 256, 3, padding=1, bias=False),
        norm_fn(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, rep_output_dim, 1)
      )
      self.from_scratch_layers += [*self.representation.modules()]

    if not trainable_backbone:
      # Do not frozen stage 5, so rep layer can be trained.
      frozen_stages = [self.stage1, self.stage2, self.stage3, self.stage4]
      self.not_training.extend(frozen_stages)
      for stage in frozen_stages:
        set_trainable_layers(stage, trainable=False)

    self.initialize(self.from_scratch_layers)

  def forward_features(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    features_s2 = x

    x = self.stage3(x)
    x = self.stage4(x)
    features_s5 = self.stage5(x)

    return features_s5, features_s2

  def forward(self, inputs, with_cam=False, with_saliency=False, with_mask=True, with_rep=True, resize_mask=True):
    features_s5, features_s2 = self.forward_features(inputs)

    outputs = self.classification_branch(features_s5, with_cam=with_cam or with_saliency)

    if with_saliency or with_mask or with_rep:
      features_s2 = self.project(features_s2)
      features = self.aspp(features_s5)

      features = resize_tensor(features, features_s2.size()[2:], align_corners=True)
      features = torch.cat((features, features_s2), dim=1)

      if with_saliency:
        masks = self.saliency_branch(outputs["features_c"], features)
        outputs["masks_sal"] = masks
        if resize_mask:
          outputs["masks_sal_large"] = resize_tensor(masks, inputs.shape[2:], align_corners=True)

      if with_mask:
        masks = self.decoder(features)
        outputs["masks_seg"] = masks
        if resize_mask:
          outputs["masks_seg_large"] = resize_tensor(masks, inputs.shape[2:], align_corners=True)

      if with_rep and self.use_rep_head:
        rep = self.representation(features)
        outputs["rep"] = rep

    return outputs

  def saliency_branch(self, features_c, features):
    if not self.use_sal_head:
      return features_c.sum(dim=1, keepdim=True)

    x = self.saliency_head(features)
    return x

  def classification_branch(self, features, with_cam=False):
    if with_cam:
      features = self.classifier(features)
      logits = gap2d(features)
      return {
        "logits_c": logits,
        "features_c": features,
      }

    features = gap2d(features, keepdims=True)
    logits = self.classifier(features).view(-1, self.num_classes)
    return {"logits": logits}


if __name__ == "__main__":
  DEVICE = "cuda"

  model = CSRM(
    "resnet50",  # args.architecture,
    21,  # train_dataset.info.num_classes,
    mode="fix",  # mode=args.mode,
    dilated=True,  # dilated=args.dilated,
    regularization=None,  # regularization=args.regularization,
    trainable_stem=True,  # trainable_stem=args.trainable_stem,
    trainable_backbone=True,
  )
