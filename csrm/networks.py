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
    channels=3,
    backbone_weights="imagenet",
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=False,
    trainable_stage4=True,
    trainable_backbone=True,
    use_group_norm=False,
    use_sal_head=False,
    use_rep_head=True,
    rep_output_dim=256,
    dropout: Union[float, Tuple[float, float]] = 0.1,
  ):
    super().__init__(
      model_name,
      channels=channels,
      weights=backbone_weights,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_stage4=trainable_stage4,
      trainable_backbone=trainable_backbone,
    )

    low_level_cin = self.backbone.stage_features[0]
    cin = self.backbone.outplanes
    norm_fn = group_norm if use_group_norm else nn.BatchNorm2d

    self.num_classes = num_classes
    self.num_classes_segm = num_classes_segm or num_classes + 1
    self.use_sal_head = use_sal_head
    self.use_rep_head = use_rep_head
    self.dropout = [dropout, dropout] if isinstance(dropout, float) else dropout

    ## Pretrained parameters
    # self.backbone = ...
    self.classifier = nn.Conv2d(cin, self.num_classes, 1, bias=False)

    ## Scratch parameters
    self.aspp = ASPP(cin, output_stride=16, norm_fn=norm_fn)

    self.project = nn.Sequential(
      nn.Conv2d(low_level_cin, 48, 1, bias=False),
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

    self.initialize(self.from_scratch_layers)

  def forward_features(self, x):
    return self.backbone(x)

  def forward(self, inputs, with_cam=False, with_saliency=False, with_mask=True, with_rep=True, resize_mask=True):
    outs = self.forward_features(inputs)
    features_s2 = outs[0]
    features_s5 = outs[-1]

    outputs = self.classification_branch(features_s5, with_cam=with_cam or with_saliency)

    if not (with_saliency or with_mask or with_rep):
      return outputs

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
    trainable_stem=True,  # trainable_stem=args.trainable_stem,
    trainable_backbone=True,
  )
