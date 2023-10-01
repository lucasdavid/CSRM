import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from core.networks import *
from tools.ai.torch_utils import gap2d, resize_tensor


class SingleStageModel(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    num_classes_segm=None,
    mode='fix',
    dilated=True,
    strides=None,
    regularization=None,
    trainable_stem=False,
    trainable_backbone=True,
    use_group_norm=False,
    use_sal_head=False,
  ):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_backbone=trainable_backbone,
    )

    cin = self.out_features
    norm_fn = group_norm if use_group_norm else nn.BatchNorm2d

    self.num_classes = num_classes
    self.regularization = regularization
    self.use_sal_head = use_sal_head

    # Pretrained parameters

    self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)

    # Scratch parameters

    self.aspp = ASPP(cin, output_stride=16, norm_fn=norm_fn)
    self.decoder = Decoder(num_classes_segm or (num_classes + 1), 256, norm_fn)

    self.from_scratch_layers += [*self.aspp.modules(), *self.decoder.modules()]

    if use_sal_head:
      self.saliency_head = nn.Conv2d(cin, 1, 1)
      self.from_scratch_layers += [self.saliency_head]

  def forward_features(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    features_s2 = x

    x = self.stage3(x)
    x = self.stage4(x)
    features_s5 = self.stage5(x)

    return features_s5, features_s2

  def forward(self, inputs, with_cam=False, with_saliency=False, with_mask=True, resize_mask=True):
    features_s5, features_s2 = self.forward_features(inputs)
    outputs = self.classification_branch(features_s5, with_cam=with_cam or with_saliency)

    if with_saliency:
      masks = self.saliency_branch(outputs["features_c"], features_s5, features_s2)
      masks = resize_tensor(masks, inputs.shape[2:], align_corners=True) if resize_mask else masks
      outputs["masks_sal"] = masks

    if with_mask:
      masks = self.segmentation_branch(features_s5, features_s2)
      masks = resize_tensor(masks, inputs.shape[2:], align_corners=True) if resize_mask else masks
      outputs["masks_seg"] = masks

    return outputs

  def saliency_branch(self, features_c, features_s5, features_s2):
    if not self.use_sal_head:
      return features_c.sum(dim=1, keepdim=True)

    x = self.saliency_head(features_s5)
    return x

  def segmentation_branch(self, features, features_low):
    features = self.aspp(features)
    features = self.decoder(features, features_low)

    return features

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

  model = SingleStageModel(
    "resnet50",  # args.architecture,
    21,  # train_dataset.info.num_classes,
    mode="fix",  # mode=args.mode,
    dilated=True,  # dilated=args.dilated,
    regularization=None,  # regularization=args.regularization,
    trainable_stem=True,  # trainable_stem=args.trainable_stem,
    trainable_backbone=True,
  )
