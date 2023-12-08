from multiprocessing import Pool

import numpy as np
import torch

import datasets

from tools.ai.demo_utils import crf_inference_label, denormalize
from tools.ai.torch_utils import make_cam, resize_tensor, to_numpy


def get_pseudo_label(
  images,
  targets,
  cams,
  masks_bg,
  thresholds,
  resize_align_corners=None,
  mode="cam",
  c2s_sigma=0.5,
  clf_t=0.5,
  num_workers: int = None,
):
  sizes = images.shape[2:]

  cams = cams.cpu().to(torch.float32)
  cams *= to_2d(targets)
  cams = make_cam(cams, inplace=True, global_norm=False)
  cams = resize_tensor(cams, sizes, align_corners=resize_align_corners)
  cams = to_numpy(cams)

  targets_b = to_numpy(targets) > clf_t

  images = to_numpy(images)
  masks_bg = to_numpy(masks_bg)

  with Pool(processes=min(num_workers, len(images)) if num_workers else len(images)) as pool:
    if mode == "cam":
      _fn = _get_pseudo_label_from_cams
      _args = [(i, c, t, thresholds) for i, c, t in zip(images, cams, targets_b)]
    else:
      _fn = _get_pseudo_label_from_mutual_promotion
      _args = [(i, c, t, m, c2s_sigma) for i, c, t, m in zip(images, cams, targets_b, masks_bg)]

    masks = pool.map(_fn, _args)

  return torch.as_tensor(np.asarray(masks, dtype="int64"))


def to_2d(x):
  return x[..., None, None]


def _get_pseudo_label_from_cams(args):
  image, cam, target, (bg_t, fg_t) = args
  image = denormalize(image, *datasets.imagenet_stats())

  cam = cam[target]
  labels = np.concatenate(([0], np.where(target)[0] + 1))

  fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=fg_t)
  fg_cam = np.argmax(fg_cam, axis=0)
  fg_conf = labels[crf_inference_label(image, fg_cam, n_labels=labels.shape[0], t=10, gt_prob=0.7)]

  bg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=bg_t)
  bg_cam = np.argmax(bg_cam, axis=0)
  bg_conf = labels[crf_inference_label(image, bg_cam, n_labels=labels.shape[0], t=10, gt_prob=0.7)]

  mask = fg_conf.copy()
  mask[fg_conf == 0] = 255
  mask[bg_conf + fg_conf == 0] = 0

  return mask


def _get_pseudo_label_from_mutual_promotion(args):
  image, cam, target, bg_mask, c2s_sigma = args
  image = denormalize(image, *datasets.imagenet_stats())

  cam = cam[target]
  labels = np.concatenate(([0], np.where(target)[0] + 1))

  ## bg_mask and cam are normalized probs:
  probs = np.concatenate((bg_mask[np.newaxis, ...], cam), axis=0)
  mask = np.argmax(probs, axis=0)
  # mask = crf_inference_label(image, mask, t=10, n_labels=labels.shape[0], gt_prob=0.7)
  mask = labels[mask]

  ## bg_mask and cam are logits:
  # logit = np.concatenate((bg_mask[np.newaxis, ...], cam), axis=0)
  # probs = scipy.special.softmax(logit, axis=0)
  # probs = crf_inference(image, probs, t=10)
  # mask = labels[np.argmax(probs, axis=0)]

  # logit = np.concatenate((bg_mask[np.newaxis, ...], cam), axis=0)
  # probs = scipy.special.softmax(logit, axis=0)
  # mask = np.argmax(probs, axis=0)
  # mask = labels[crf_inference_label(image, mask, t=10, n_labels=labels.shape[0])]

  uncertain_pixels = probs.max(axis=0) < c2s_sigma
  mask[uncertain_pixels] = 255

  # print(f"labels={labels} cam={cam.shape} bg_mask={bg_mask.shape} logit={logit.shape} probs={probs.shape} mask={mask.shape}")
  # print(f"mask={mask.shape} uncertain_pixels={uncertain_pixels.shape}")

  return mask
