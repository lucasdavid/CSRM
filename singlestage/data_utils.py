from typing import Union, Tuple, Callable, Dict, Optional

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


def mixaug(
  images,
  labels,
  masks,
  logits,
  beta: float = 1.,
  mix: str = "classmix",
  k: int = 1,
  ignore_class: Optional[int] = None,
  bg_in_labels: bool = False,
):
  images_ = images.detach().clone()
  labels_ = labels.detach().clone()
  masks_ = masks.detach().clone()
  logits_ = logits.detach().clone()

  ids = np.arange(len(images), dtype="int")

  for _ in range(k):
    bids = np.asarray([np.random.choice(ids[ids != i]) for i in ids])

    for ai, bi in zip(ids, bids):
      if mix == "cutmix":
        b_mask = generate_cutout_mask(masks[bi])
        b_alpha = b_mask.float().mean()
        labels_[ai, :] = (1 - b_alpha) * labels_[ai] + b_alpha * labels[bi]
      elif mix == "classmix":
        b_mask, new_labels = generate_class_mask(masks[bi], ignore_class)
        if not bg_in_labels:
          # Classification labels do not contain the bg, while
          # masks do. Shift mask label to match the target.
          new_labels -= 1
        labels_[ai, new_labels] = 1.
      else:
        raise ValueError(f"Unknown mix {mix}. Options are `cutmix` and `classmix`.")

      # adjusting for padding and empty/uncertain regions.
      b_mask = (b_mask == 1) & (masks[bi] != 255)

      images_[ai][:, b_mask] = images[bi][:, b_mask]
      logits_[ai][:, b_mask] = logits[bi][:, b_mask]
      masks_[ai][b_mask] = masks[bi][b_mask]

  return images_, labels_, masks_, logits_  # , lgs_sal_mix


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
