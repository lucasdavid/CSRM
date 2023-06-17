# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import random

import numpy as np
from PIL import Image, UnidentifiedImageError


def create_directory(path):
  if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
  return path


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_cam_file(png_file, npy_file):
  if png_file and os.path.exists(png_file):
    with Image.open(png_file) as y_pred:
      y_pred = np.asarray(y_pred)

    labels, cam = np.unique(y_pred, return_inverse=True)
    cam = cam.reshape(y_pred.shape)

  else:
    data = np.load(npy_file, allow_pickle=True).item()

    if "keys" in data:
      # Affinity/Puzzle/PNOC
      labels = data["keys"]

      if "hr_cam" in data.keys():
        cam = data["hr_cam"]
      elif "rw" in data.keys():
        cam = data["rw"]
    else:
      # OC-CSE
      labels = list(data.keys())
      cam = np.stack([data[k] for k in labels], 0)
      labels = np.asarray([0] + [k+1 for k in labels])

  return cam, labels


def load_saliency_file(sal_file, kind='saliency'):

  with Image.open(sal_file) as s:
    s = np.asarray(s)

  if kind == 'saliency':
    s = s.astype(float) / 255.
  elif kind == 'segmentation':
    s = (~np.isin(s, [0, 255])).astype(float)

  if len(s.shape) == 2:
    s = s[np.newaxis, ...]

  return s


def load_background_file(sal_file, kind='saliency'):
  return 1 - load_saliency_file(sal_file, kind=kind)
