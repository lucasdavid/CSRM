from typing import Union, Tuple, Callable, Dict

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from PIL import Image, ImageFilter

import datasets


def transform(
  entry: Dict[str, torch.Tensor],
  crop_size: Union[int, Tuple[int, int]] = (512, 512),
  scale_size: Tuple[float, float] = (1., 1.),
  augmentation: bool = True,
  color: bool = True,
  blur: bool = True,
):
  image, label, logits = entry["image"], entry["mask"], entry.get("logits", None)

  # Random rescale image
  raw_w, raw_h = image.size

  scale_ratio = random.uniform(scale_size[0], scale_size[1])
  resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
  image = transforms_f.resize(image, resized_size, Image.BILINEAR)
  label = transforms_f.resize(label, resized_size, Image.NEAREST)
  if logits is not None:
    logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

  # Add padding if rescaled image size is less than crop size
  if crop_size == -1:  # use original im size without crop or padding
    crop_size = (raw_h, raw_w)

  if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
    right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
    image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
    label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
    if logits is not None:
      logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

  # Random Cropping
  i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
  image = transforms_f.crop(image, i, j, h, w)
  label = transforms_f.crop(label, i, j, h, w)
  if logits is not None:
    logits = transforms_f.crop(logits, i, j, h, w)

  if augmentation:
    # Random color jitter
    if color and torch.rand(1) > 0.2:
      color_transform = transforms.ColorJitter(
        (0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25)
      )  # For PyTorch 1.9/TorchVision 0.10 users
      # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
      image = color_transform(image)

    # Random Gaussian filter
    if blur and torch.rand(1) > 0.5:
      sigma = random.uniform(0.15, 1.15)
      image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    # Random horizontal flipping
    if torch.rand(1) > 0.5:
      image = transforms_f.hflip(image)
      label = transforms_f.hflip(label)
      if logits is not None:
        logits = transforms_f.hflip(logits)

  # Transform to tensor
  image = transforms_f.to_tensor(image)
  label = (transforms_f.to_tensor(label) * 255).long()
  # label[label == 255] = -1  # invalid pixels are re-mapped to index -1
  if logits is not None:
    logits = transforms_f.to_tensor(logits)

  # Apply (ImageNet) normalisation
  image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  entry["image"] = image
  entry["mask"] = label[0]

  if logits is not None:
    entry["logits"] = logits

  return entry


# region Augmentation Policy


def apply_mixaug(images, labels, masks, lgs_seg, beta=1., mix="classmix", ignore_class=None):
  images_mix = images.detach().clone()
  labels_mix = labels.detach().clone()
  masks_mix = masks.detach().clone()
  lgs_seg_mix = lgs_seg.detach().clone()
  # lgs_sal_mix = lgs_sal.detach().clone()

  ids = np.arange(len(images), dtype="int")
  bids = np.asarray([np.random.choice(ids[ids != i]) for i in ids])

  for idx_a, idx_b in zip(ids, bids):
    if mix == "cutmix":
      mix_b = generate_cutout_mask(masks[idx_b])
      alpha_b = mix_b.float().mean()
      labels_mix = (1 - alpha_b) * labels[idx_a] + alpha_b * labels[idx_b]
    elif mix == "classmix":
      mix_b, new_labels = generate_class_mask(masks[idx_b], ignore_class)
      alpha_b = mix_b.float().mean()
      labels_mix[idx_a, :] = alpha_b * labels[idx_a]
      labels_mix[idx_a][new_labels] = 1.0
    else:
      raise ValueError(f"Unknown mix {mix}. Options are `cutmix` and `classmix`.")

    # adjusting for padding and empty/uncertain regions.
    mix_b = (mix_b == 1) & (masks[idx_b] != 255)

    images_mix[idx_a][:, mix_b] = images[idx_b][:, mix_b]
    masks_mix[idx_a][mix_b] = masks[idx_b][mix_b]
    lgs_seg_mix[idx_a][:, mix_b] = lgs_seg[idx_b][:, mix_b]
    # lgs_sal_mix[idx_a][:, mix_b] = lgs_sal[idx_b][:, mix_b]

  return images_mix, labels_mix, masks_mix, lgs_seg_mix  # , lgs_sal_mix


def generate_cutout_mask(pseudo_labels, ratio=2):
  img_size = pseudo_labels.shape
  cutout_area = img_size[0] * img_size[1] / ratio

  w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
  h = np.round(cutout_area / w)

  x_start = np.random.randint(0, img_size[1] - w + 1)
  y_start = np.random.randint(0, img_size[0] - h + 1)

  x_end = int(x_start + w)
  y_end = int(y_start + h)

  mask = torch.zeros(img_size, dtype=torch.int32)
  mask[y_start:y_end, x_start:x_end] = 1
  return mask


def generate_class_mask(pseudo_labels, ignore_class=None):
  labels = torch.unique(pseudo_labels)
  labels = labels[labels != 255]
  if ignore_class is not None:
    labels = labels[labels != ignore_class]
  labels_select = labels[torch.randperm(len(labels))][:max(len(labels) // 2, 1)]
  mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
  return mask.float(), labels_select


# endregion

# region ReCo loss


def compute_reco_loss(
  rep, prob, pseudo_mask, valid_mask, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256
):
  batch_size, num_feat, im_w_, im_h = rep.shape
  num_segments = pseudo_mask.shape[1]
  device = rep.device

  # compute valid binary mask for each pixel
  certain_pixel = (pseudo_mask * valid_mask).to(device)  # BCHW

  # permute representation for indexing: batch x im_h x im_w x feature_channel
  rep = rep.permute(0, 2, 3, 1)

  # compute prototype (class mean representation) for each class across all valid pixels
  all_feats = []
  hard_feats = []
  num_pixels = []
  protos = []
  for i in range(num_segments):
    certain_pixel_i = certain_pixel[:, i]  # select binary mask for i-th class
    if certain_pixel_i.sum() == 0:  # not all classes would be available in a mini-batch
      continue

    prob_ci = prob[:, i, :, :]
    rep_mask_hard = (prob_ci < strong_threshold) * certain_pixel_i.bool()  # select hard queries

    protos.append(torch.mean(rep[certain_pixel_i.bool()], dim=0, keepdim=True))
    all_feats.append(rep[certain_pixel_i.bool()])
    hard_feats.append(rep[rep_mask_hard])
    num_pixels.append(int(certain_pixel_i.sum().item()))

  # compute regional contrastive loss
  if len(num_pixels) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
    return torch.tensor(0.0)
  else:
    reco_loss = torch.tensor(0.0)
    seg_proto = torch.cat(protos)
    valid_seg = len(num_pixels)
    seg_len = torch.arange(valid_seg)

    for i in range(valid_seg):
      # sample hard queries
      if len(hard_feats[i]) > 0:
        seg_hard_idx = torch.randint(len(hard_feats[i]), size=(num_queries,))
        anchor_feat_hard = hard_feats[i][seg_hard_idx]
        anchor_feat = anchor_feat_hard
      else:  # in some rare cases, all queries in the current query class are easy
        continue

      # apply negative key sampling (with no gradients)
      with torch.no_grad():
        # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
        seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

        # compute similarity for each negative segment prototype (semantic class relation graph)
        proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
        proto_prob = torch.softmax(proto_sim / temp, dim=0)

        # sampling negative keys based on the generated distribution [num_queries x num_negatives]
        negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
        samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
        samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

        # sample negative indices from each negative class
        negative_num_list = num_pixels[i + 1:] + num_pixels[:i]
        negative_index = negative_index_sampler(samp_num, negative_num_list)

        # index negative keys (from other classes)
        negative_feat_all = torch.cat(all_feats[i + 1:] + all_feats[:i])
        negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

        # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
        positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
        all_feat = torch.cat((positive_feat, negative_feat), dim=1)

      seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
      reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))

    return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
  negative_index = []
  for i in range(samp_num.shape[0]):
    for j in range(samp_num.shape[1]):
      negative_index += np.random.randint(
        low=sum(seg_num_list[:j]), high=sum(seg_num_list[:j + 1]), size=int(samp_num[i, j])
      ).tolist()
  return negative_index


def label_onehot(inputs, num_segments):
  batch_size, im_h, im_w = inputs.shape
  # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
  # we will still mask out those invalid values in valid mask
  # inputs = torch.relu(inputs)
  outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
  return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


# endregion
