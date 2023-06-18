# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

from core.datasets import get_inference_dataset
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import create_directory, load_cam_file

MERGE_OPS = ["sum", "avg", "max", "weighted", "ranked", "highest"]

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)

parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--sample_ids', default=None, type=str)

parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--merge", type=str, required=True, choices=MERGE_OPS)
parser.add_argument("--out_dir", default="", type=str)
parser.add_argument('--EXPERIMENTS', nargs="+", type=str)
parser.add_argument('--weights_path', default=None, type=str)

parser.add_argument('--verbose', default=1, type=int)


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, dataset, args, experiments, cam_dirs, weights, out_dir):
  subset = dataset[process_id]
  processed = 0
  missing = []
  errors = []

  if process_id == 0:
    subset = tqdm(subset, mininterval=2.)

  with torch.no_grad():
    for image, image_id, label in subset:
      out_path = os.path.join(out_dir, image_id + '.npy')

      if os.path.isfile(out_path) or label.sum() == 0:
        # Skip samples already processed or containing only bg
        continue

      ensemble = []

      W, H = image.size

      for experiment_id, cam_dir in zip(experiments, cam_dirs):
        cam_path = os.path.join(cam_dir, image_id + '.npy')

        cam, keys = load_cam_file(npy_file=cam_path, png_file=None)
        ensemble.append((cam, keys))

      if args.verbose >= 2:
        print(f"  shapes = {[e[0].shape for e in ensemble]}")
        print(f"  label  = {[[0] + (np.where(label)[0]+1)]}")
        print(f"  keys   = {[e[1] for e in ensemble]}")

      try:
        if args.merge == "sum":
          cam = np.sum([e[0] for e in ensemble], axis=0)
        elif args.merge == "avg":
          cam = np.mean([e[0] for e in ensemble], axis=0)
        elif args.merge == "max":
          cam = np.max([e[0] for e in ensemble], axis=0)

        elif args.merge == "weighted":
          wc = weights.loc[:, label + 1].mean(axis=1)  # (E, C) --> (E, K) --> (E)
          wc /= wc.sum()

          cam = np.sum([e[0] * w for e, w in zip(ensemble, wc)], axis=0)
        elif args.merge == "ranked":
          wc = weights.loc[:, label + 1].mean(axis=1)  # (E, C) --> (E, K) --> (E)
          wc = wc.argsort()[::-1]
          wc = wc.argsort()
          rank_weights = np.exp(-wc)
          rank_weights /= rank_weights.sum()

          cam = np.sum([e[0] * w for e, w in zip(ensemble, rank_weights)], axis=0)

        elif args.merge == "highest":
          wc = weights.loc[:, label + 1].mean(axis=1)  # (E, C) --> (E, K) --> (E)
          wc = wc.argmax()

          cam = ensemble[wc][0]

      except ValueError as error:
        print(f"Cannot merge ensemble for {image_id} due to:")
        print(error)
        print(f"parts: {[e[0].shape for e in ensemble]}", file=sys.stderr)
        continue

      if args.verbose >= 2:
        print(f"  cam    = {cam.shape}")

      try:
        np.save(out_path, {"keys": keys, "hr_cam": cam})
      except KeyboardInterrupt:
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
  EXPERIMENTS = args.EXPERIMENTS
  OUT_DIR = create_directory(args.out_dir or f"./EXPERIMENTS/predictions/{TAG}/")
  cam_dirs = [f'./EXPERIMENTS/predictions/{e}/' for e in EXPERIMENTS]

  set_seed(args.seed)

  print(TAG)
  print("Experiments:", *EXPERIMENTS, sep="\n  - ", end="\n\n")

  dataset = get_inference_dataset(args.dataset, args.data_dir, args.domain, sample_ids=args.sample_ids)
  dataset = split_dataset(dataset, args.num_workers)

  if args.weights_path:
    weights = pd.read_csv(args.weights_path)

  if args.num_workers > 1:
    multiprocessing.spawn(
      _work, nprocs=args.num_workers, args=(dataset, args, EXPERIMENTS, cam_dirs, weights, OUT_DIR), join=True
    )
  else:
    _work(0, dataset, args, EXPERIMENTS, cam_dirs, weights, OUT_DIR)
