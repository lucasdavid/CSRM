# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
from pickle import UnpicklingError
import sys

import numpy as np
import pandas as pd
import torch
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

import ensemble
import datasets
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import create_directory, str2bool


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)

parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--sample_ids', default=None, type=str)
parser.add_argument('--exclude_bg_images', default=True, type=str2bool)

parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--merge", type=str, required=True, choices=list(ensemble.STRATEGIES))
parser.add_argument("--alpha", type=float, default=None, help="only used if merge is `weighted` or `ranked`.")

parser.add_argument("--out_dir", default="", type=str)
parser.add_argument('--experiments', nargs="+", type=str)
parser.add_argument('--weights_path', default=None, type=str)

parser.add_argument('--verbose', default=1, type=int)


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, dataset: Subset, args, experiments, cam_dirs, weights, out_dir):
  subset = dataset[process_id]
  data_source: datasets.CustomDataSource = subset.dataset.data_source
  processed = 0
  missing = []
  errors = []

  if process_id == 0:
    subset = tqdm(subset, mininterval=5.)

  with torch.no_grad():
    for image_id, image_path, mask_path in subset:
      out_path = os.path.join(out_dir, image_id + '.npy')

      if os.path.isfile(out_path):
        # Skip samples already processed.
        continue

      image = data_source.get_image(image_id)
      targets = data_source.get_label(image_id)

      partial_cams = []


      W, H = image.size

      for cam_dir in cam_dirs:
        cam_path = os.path.join(cam_dir, image_id + '.npy')

        try:
          data = np.load(cam_path, allow_pickle=True).item()
        except UnpicklingError as error:
          errors.append(cam_path)
          if args.verbose >= 3:
            print(f"{image_id} skipped (cam error={error})")
        except FileNotFoundError:
          missing.append(cam_path)
          if args.verbose >= 3:
            print(f"{image_id} skipped (cam missing)")
        else:
          if "keys" in data:
            # Affinity/Puzzle/PNOC
            keys = data["keys"]

            if "hr_cam" in data.keys():
              cam = data["hr_cam"]
            elif "rw" in data.keys():
              cam = data["rw"]
          else:
            # OC-CSE
            keys = list(data.keys())
            cam = np.stack([data[k] for k in keys], 0)
            keys = np.asarray([0] + [k+1 for k in keys])

          partial_cams.append((cam, keys))

      if args.verbose >= 2:
        print(f"  shapes = {[e[0].shape for e in partial_cams]}")
        print(f"  label  = {[[0] + (np.where(targets)[0]+1)]}")
        print(f"  keys   = {[e[1] for e in partial_cams]}")

      try:
        cam = ensemble.merge(partial_cams, targets, args.merge, weights=weights, alpha=args.alpha)
      except ValueError as error:
        print(f"Cannot merge ensemble for {image_id} due to:")
        print(error)
        print(f"parts: {[e[0].shape for e in partial_cams]}", file=sys.stderr)
        continue

      if args.verbose >= 2:
        print(f"  cam    = {cam.shape}")

      try:
        np.save(out_path, {"keys": keys, "hr_cam": cam})
      except:
        if os.path.exists(out_path):
          os.remove(out_path)
        raise

      processed += 1

  if missing:
    print(f"{len(missing)} files were missing and were not processed:", *missing, sep='\n  - ', flush=True)
  if errors:
    print(f"{len(errors)} CAM files could not be read:", *errors, sep="\n  - ", flush=True)

  print(f"{processed} images successfully processed")


if __name__ == '__main__':
  try:
    multiprocessing.set_start_method('spawn')
  except RuntimeError:
    ...

  args = parser.parse_args()

  TAG = args.tag
  EXPERIMENTS = args.experiments
  OUT_DIR = create_directory(args.out_dir or f"./experiments/predictions/{TAG}/")
  CAM_DIRS = [f'./experiments/predictions/{e}/' for e in EXPERIMENTS]

  set_seed(args.seed)

  print(TAG)
  print("Experiments:", *EXPERIMENTS, sep="\n  - ", end="\n\n")

  source = datasets.custom_data_source(args.dataset, args.data_dir, args.domain, sample_ids=args.sample_ids)
  dataset = datasets.PathsDataset(source, ignore_bg_images=True)
  dataset = split_dataset(dataset, args.num_workers)

  if args.weights_path and args.merge in ("weighted", "ranked", "highest"):
    weights = pd.read_csv(args.weights_path)
    print(f"Merging weights loaded from {args.weights_path}:")
    print(weights.round(2))
  else:
    weights = None

  if args.num_workers > 1:
    multiprocessing.spawn(
      _work, nprocs=args.num_workers, args=(dataset, args, EXPERIMENTS, CAM_DIRS, weights, OUT_DIR), join=True
    )
  else:
    _work(0, dataset, args, EXPERIMENTS, CAM_DIRS, weights, OUT_DIR)
