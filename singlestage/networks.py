import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from core.networks import *
from datasets import imagenet_stats
from tools.ai.demo_utils import crf_inference_label, denormalize
from tools.ai.torch_utils import gap2d, make_cam, resize_tensor, to_numpy


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
    self.criterion_b = criterion_b or torch.nn.BCEWithLogitsLoss()

  def forward_features(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    features_s2 = x

    x = self.stage3(x)
    x = self.stage4(x)
    features = self.stage5(x)

    return features, features_s2

  def forward(self, inputs, with_cam=False, with_saliency=False, with_mask=True):
    features_s5, features_s2 = self.forward_features(inputs)
    outputs = self.classification_branch(features_s5, with_cam=with_cam or with_saliency)

    if with_saliency:
      _, features = outputs
      if self.use_saliency_head:
        logits_saliency = self.saliency_head(features)
      else:
        logits_saliency = features.sum(dim=1, keepdim=True)

      outputs = (*outputs, logits_saliency)

    if with_mask:
      masks = self.segmentation_branch(features_s5, features_s2)
      masks_resized = resize_tensor(masks, inputs.size()[2:], align_corners=True)

      outputs = (*outputs, masks, masks_resized)

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
    criterion_c=torch.nn.MultiLabelSoftMarginLoss().to(DEVICE),
    criterion_s=torch.nn.CrossEntropyLoss(ignore_index=255).to(DEVICE),
  )
