import argparse
import copy
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

import datasets
from csrm import CSRM
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
# parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--pred_dir', default=None, type=str)
parser.add_argument('--sample_ids', default=None, type=str)

parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)
parser.add_argument('--use_sal_head', default=False, type=str2bool)
parser.add_argument('--use_rep_head', default=True, type=str2bool)

parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--hflip', default=True, type=str2bool)
parser.add_argument('--exclude_bg_images', default=False, type=str2bool)


parser.add_argument("--threshold", default=0.25, type=float)
parser.add_argument("--crf_t", default=0, type=int)
parser.add_argument("--crf_gt_prob", default=0.7, type=float)

parser.add_argument("--save_cams", default=True, type=str2bool)
parser.add_argument("--save_masks", default=True, type=str2bool)
parser.add_argument("--save_pseudos", default=False, type=str2bool)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

normalize_fn = Normalize(*datasets.imagenet_stats())


def run(args):
  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=args.exclude_bg_images)
  info = ds.classification_info
  print(f'{TAG} dataset={args.dataset} num_classes={info.num_classes}')

  model = CSRM(
    args.architecture,
    num_classes=ds.classification_info.num_classes,
    num_classes_segm=ds.segmentation_info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    use_group_norm=args.use_gn,
    use_sal_head=args.use_sal_head,
    use_rep_head=args.use_rep_head,
  )
  load_model(model, WEIGHTS_PATH, map_location=torch.device(DEVICE))
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]
  scales = [float(scale) for scale in args.scales.split(',')]

  if GPUS_COUNT > 1:
    multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, dataset, scales, args.hflip, PREDS_DIR, DEVICE, args), join=True)
  else:
    _work(0, model, dataset, scales, args.hflip, PREDS_DIR, DEVICE, args)


def _work(
    process_id: int,
    model: Classifier,
    dataset: List[datasets.PathsDataset],
    scales: List[float],
    hflip: bool,
    preds_dir: str,
    device: str,
    args: "Namespace",
):
  dataset = dataset[process_id]
  data_source = dataset.dataset.data_source

  if process_id == 0:
    dataset = tqdm(dataset, mininterval=2.0)

  with torch.no_grad(), torch.cuda.device(process_id):
    model.cuda()

    for image_id, _, _ in dataset:
      cam_path = os.path.join(preds_dir, "cams", image_id + '.npy')
      seg_path = os.path.join(preds_dir, "masks", image_id + '.npy')
      pse_path = os.path.join(preds_dir, f"pseudos-t{args.threshold}-c{args.crf_t}", image_id + '.png')

      if ((not args.save_cams or os.path.isfile(cam_path)) and
          (not args.save_masks or os.path.isfile(seg_path)) and
          (not args.save_pseudos or os.path.isfile(pse_path))):
        continue

      image = data_source.get_image(image_id)
      label = data_source.get_label(image_id)

      W, H = image.size

      strided_size = get_strided_size((H, W), 4)
      strided_up_size = get_strided_up_size((H, W), 16)

      if label.sum() == 0:
        keys = np.asarray([data_source.segmentation_info.bg_class])
        cams_st = np.zeros((0, *strided_size))
        cams_hr = np.zeros((0, *strided_up_size))
      else:
        cams, masks = zip(*(forward_tta(model, image, scale, hflip, device, args.save_masks) for scale in scales))

        cams_st = [resize_tensor(c.unsqueeze(0), strided_size)[0] for c in cams]
        cams_st = torch.sum(torch.stack(cams_st), dim=0)

        cams_hr = [resize_tensor(cams.unsqueeze(0), strided_up_size)[0] for cams in cams]
        cams_hr = torch.sum(torch.stack(cams_hr), dim=0)[:, :H, :W]

        keys = torch.nonzero(torch.from_numpy(label))[:, 0]
        cams_st = cams_st[keys]
        cams_st /= F.adaptive_max_pool2d(cams_st, (1, 1)) + 1e-5
        cams_hr = cams_hr[keys]
        cams_hr /= F.adaptive_max_pool2d(cams_hr, (1, 1)) + 1e-5

        keys = np.pad(keys + 1, (1, 0), mode='constant')  # add background
        cams_st = cams_st.cpu()
        cams_hr = to_numpy(cams_hr)

      if args.save_cams and not os.path.isfile(cam_path):
        safe_save(cam_path, {"keys": keys, "cam": cams_st, "hr_cam": cams_hr})

      if args.save_pseudos:
        keys = np.pad(np.nonzero(label)[0] + 1, (1, 0), mode="constant")
        cam = np.pad(cams_hr, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=args.threshold)
        prob = cam
        cam = np.argmax(cam, axis=0)

        if args.crf_t and len(keys) > 1:
          img = np.asarray(image).astype(np.uint8)

          if prob is not None and args.crf_gt_prob == 1.0:
            # DeepLab-pytorch's CRF
            prob = crf_inference_dlv2_softmax(img, prob, t=args.crf_t)
            cam = prob.argmax(0)
          else:
            cam = crf_inference_label(img, cam, n_labels=max(len(keys), 2), t=args.crf_t, gt_prob=args.crf_gt_prob)

        y_pred = keys[cam]

        try:
          with Image.fromarray(y_pred.astype(np.uint8)) as p:
            p.save(pse_path)
        except:
          if os.path.exists(pse_path):
            os.remove(pse_path)
          raise

      if args.save_masks and not os.path.isfile(seg_path):
        # Masks.
        keys = np.nonzero(label)[0]
        keys = np.pad(keys + 1, (1, 0), mode='constant')
        masks = [resize_tensor(logits[keys, ...][None, ...], (H, W))[0] for logits in masks]
        masks = torch.softmax(torch.stack(masks).mean(dim=0), dim=0)

        safe_save(seg_path, {"keys": keys, "hr_cam": to_numpy(masks)})


def forward_tta(model, ori_image, scale, hflip, DEVICE, with_mask=True):
  W, H = ori_image.size

  # Preprocessing
  x = copy.deepcopy(ori_image)
  x = x.resize((round(W * scale), round(H * scale)), resample=PIL.Image.BICUBIC)
  x = normalize_fn(x)
  x = x.transpose((2, 0, 1))
  x = torch.from_numpy(x)
  images = torch.stack([x, x.flip(-1)]) if hflip else x[None, ...]
  images = images.to(DEVICE)

  outputs = model(images, with_cam=True, with_mask=True, with_rep=False)
  features = outputs["features_c"]
  cams = F.relu(features)
  if hflip:
    cams = cams[0] + cams[1].flip(-1)
  else:
    cams = cams[0]

  masks = None

  if with_mask:
    masks = outputs["masks_seg_large"].cpu().float()
    if hflip:
      masks = masks[0] + masks[1].flip(-1)
    else:
      masks = masks[0]

  return cams, masks


def safe_save(path, data):
  try:
    np.save(path, data)
  except:
    if os.path.exists(path):
      os.remove(path)
    raise


if __name__ == '__main__':
  args = parser.parse_args()

  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  SEED = args.seed
  TAG = args.tag
  TAG += '@train' if 'train' in args.domain else '@val'

  log_config(vars(args), TAG)

  PREDS_DIR = args.pred_dir or f'./experiments/predictions/{TAG}/'
  WEIGHTS_PATH = args.weights or os.path.join('./experiments/models/', f'{args.tag}.pth')

  create_directory(os.path.join(PREDS_DIR, "cams"))
  create_directory(os.path.join(PREDS_DIR, "masks"))
  create_directory(os.path.join(PREDS_DIR, f"pseudos-t{args.threshold}-c{args.crf_t}"))

  set_seed(SEED)
  run(args)
