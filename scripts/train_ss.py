import argparse
import os
from multiprocessing import Pool
from typing import List

import numpy as np
import scipy.special
import sklearn.metrics as skmetrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
from core.networks import *
from singlestage import SingleStageModel
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general import wandb_utils
from tools.general.io_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--progress', default=True, type=str2bool)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)

# Dataset
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--domain_train', default=None, type=str)
parser.add_argument('--domain_valid', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str, choices=["normal", "fix"])
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--use_sal_head', default=False, type=str2bool)
parser.add_argument('--dilated', default=True, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument('--mixed_precision', default=False, type=str2bool)
parser.add_argument('--amp_min_scale', default=None, type=float)
parser.add_argument('--validate', default=True, type=str2bool)
parser.add_argument('--validate_max_steps', default=None, type=int)
parser.add_argument('--validate_thresholds', default=None, type=str)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)
parser.add_argument('--optimizer', default="sgd", choices=["sgd", "adamw", "lion"])
parser.add_argument('--momentum', default=.9, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)
parser.add_argument('--lr_poly_power', default=2., type=float)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

# Single-Stage
parser.add_argument('--c2s_mode', default="cam", choices=["cam", "mp", "gt"])
parser.add_argument('--c2s_sigma', default=0.5, type=float)
parser.add_argument('--s2c_mode', default="mp", choices=["bce", "kld", "mp"])
parser.add_argument('--s2c_sigma', default=0.5, type=float)
parser.add_argument('--s2c_warmup_epochs', default=3, type=int)


import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)
THRESHOLDS = list(np.arange(0.10, 0.50, 0.05))


def to_2d(x):
  return x[..., None, None]


def train_singlestage(args, wb_run, model_path):
  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
  tt, tv = datasets.get_segmentation_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset = datasets.SegmentationDataset(ts, transform=tt)
  valid_dataset = datasets.SegmentationDataset(vs, transform=tv)
  train_dataset = datasets.apply_augmentation(train_dataset, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_val = len(train_loader)
  step_log = max(int(step_val * args.print_ratio), 1)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f"Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  criterions = (
    torch.nn.MultiLabelSoftMarginLoss().to(DEVICE),
    torch.nn.CrossEntropyLoss(ignore_index=255, label_smoothing=args.label_smoothing).to(DEVICE),
    torch.nn.BCEWithLogitsLoss(reduction="none").to(DEVICE) if args.s2c_mode == "bce"
      else torch.nn.KLDivLoss(reduction="batchmean").to(DEVICE) if args.s2c_mode == "kld"
      else None,  # if args.s2c_mode == "mp"
  )

  # Network
  model = SingleStageModel(
    args.architecture,
    num_classes=ts.classification_info.num_classes,
    num_classes_segm=ts.segmentation_info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    use_group_norm=args.use_gn,
    trainable_stem=args.trainable_stem,
    trainable_backbone=args.trainable_backbone,
    use_sal_head=args.use_sal_head,
  )

  if args.restore:
    print(f"Restoring weights from {args.restore}")
    state_dict = torch.load(args.restore, map_location="cpu")
    for m in model.state_dict():
      if m not in state_dict:
        print("    Skip init:", m)
    model.load_state_dict(state_dict, strict=False)
  log_model("SingleStage", model, args)

  param_groups = model.get_parameter_groups()
  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  # Loss, Optimizer
  optimizer = get_optimizer(
    args.lr,
    args.wd,
    momentum=args.momentum,
    nesterov=args.nesterov,
    start_step=int(step_init // args.accumulate_steps),
    max_step=int(step_max // args.accumulate_steps),
    param_groups=param_groups,
    algorithm=args.optimizer,
    alpha_scratch=args.lr_alpha_scratch,
    alpha_bias=args.lr_alpha_bias,
    poly_power=args.lr_poly_power,
  )
  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
  log_opt_params("SingleStage", param_groups)
  print("", flush=True)

  # Train
  train_meter = MetricsContainer("loss loss_c loss_c2s loss_s2c loss_u conf_pixels_s2c conf_pixels_c2s".split())
  train_timer = Timer()
  miou_best = -1

  try:
    miou_best = valid_loop(model, valid_loader, ts, 0, optimizer, miou_best)

    for epoch in range(args.first_epoch, args.max_epoch):
      for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=2.0, disable=not args.progress, ncols=60)):
        _, images, targets, true_masks = batch

        s2c_mode = args.s2c_mode
        c2s_mode = args.c2s_mode
        s2c_sigma = args.s2c_sigma
        c2s_sigma = args.c2s_sigma

        fg_t = 0.30
        bg_t = 0.05

        # Default
        w_c = 1.0
        w_c2s = 1.0
        w_u = 0
        w_s2c = linear_schedule(optimizer.global_step, optimizer.max_step, 0.1, 1.0, 1.0)

        if epoch < args.s2c_warmup_epochs:
          w_s2c = 0

        if args.s2c_mode == "bce":
          # print("s2c_mode is BCE => setting s2c_sigma=lerp(0.9, 0.5, 1, max).")
          s2c_sigma = linear_schedule(optimizer.global_step, optimizer.max_step, args.s2c_sigma, 0.5, 1.0, constraint=max)

        if args.s2c_mode == "kld":
          # print(f"s2c_mode is KLD => setting w_c=.9, w_s2c=.1 (user T={s2c_sigma})")
          w_c = 0.9
          w_s2c = 0.1

        # Others:
        # fg_t = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5,  0.2, 1.0, constraint=max)
        # bg_t = linear_schedule(optimizer.global_step, optimizer.max_step, 0.01, 0.2, 1.0)
        # w_c2s = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5, 1.0, 0.5)
        # w_u = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5, 1.0, 0.5)

        with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
          loss, metrics = train_step(
            model,
            images,
            targets,
            true_masks,
            criterions,
            thresholds=(bg_t, fg_t),
            ls=args.label_smoothing,
            w_c=w_c,
            w_c2s=w_c2s,
            w_s2c=w_s2c,
            w_u=w_u,
            s2c_mode=s2c_mode,
            c2s_mode=c2s_mode,
            s2c_sigma=s2c_sigma,
            c2s_sigma=c2s_sigma,
            bg_class=ts.segmentation_info.bg_class,
          )

        scaler.scale(loss / args.accumulate_steps).backward()

        if (step + 1) % args.accumulate_steps == 0:
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()

        train_meter.update({m: v.item() for m, v in metrics.items()})

        do_logging = (step + 1) % step_log == 0
        do_validation = args.validate and (step + 1) == step_val

        if do_logging:
          lr = float(get_learning_rate_from_optimizer(optimizer))

          loss, loss_c, loss_c2s, loss_s2c, loss_u, conf_pixels_s2c, conf_pixels_c2s = train_meter.get(clear=True)
          data = {
            'step': optimizer.global_step,
            "epoch": epoch,
            'lr': lr,
            'time': train_timer.tok(clear=True),
            "fg_t": fg_t,
            "bg_t": bg_t,
            "loss": loss,
            "loss_c": loss_c,
            "loss_c2s": loss_c2s,
            "loss_s2c": loss_s2c,
            "loss_u": loss_u,
            "w_c2s": w_c2s,
            "w_s2c": w_s2c,
            "w_u": w_u,
            "s2c_sigma": s2c_sigma,
            "c2s_sigma": c2s_sigma,
            "conf_pixels_s2c": conf_pixels_s2c,
            "conf_pixels_c2s": conf_pixels_c2s,
          }
          wandb.log({f"train/{k}": v for k, v in data.items()}, commit=not do_validation)
          print(f" loss={loss:.3f} loss_c={loss_c:.3f} loss_c2s={loss_c2s:.3f} loss_s2c={loss_s2c:.3f} loss_u={loss_u:.3f} lr={lr:.3f}")

      miou_best = valid_loop(model, valid_loader, ts, epoch, optimizer, miou_best)
      save_model(model, model_path, parallel=GPUS_COUNT > 1)

  except KeyboardInterrupt:
    print("training halted")

  wb_run.finish()


def valid_loop(model, valid_loader, ts, epoch, optimizer, miou_best, commit=True):
  model.eval()
  with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
    metric_results = valid_step(model, valid_loader, ts, THRESHOLDS, DEVICE, log_samples=True, max_steps=args.validate_max_steps)
    metric_results.update({"step": optimizer.global_step, "epoch": epoch+1})
  model.train()

  wandb.log({f"val/{k}": v for k, v in metric_results.items()}, commit=commit)
  print(f"[Epoch {epoch}/{args.max_epoch}]",
        *(f"{m}={v:.3f}" if isinstance(v, float) else f"{m}={v}"
          for m, v in metric_results.items()))

  if metric_results["segmentation/miou"] > miou_best:
    miou_best = metric_results["segmentation/miou"]
    for k in ("miou", "iou"):
      wandb.run.summary[f"val/segmentation/best_{k}"] = metric_results[f"segmentation/{k}"]
  return miou_best


def train_step(
  model: SingleStageModel,
  images,
  targets,
  masks,
  criterions,
  thresholds,
  ls,
  w_c,
  w_c2s,
  w_s2c,
  w_u,
  s2c_sigma: float,
  c2s_sigma: float,
  s2c_mode: str,
  c2s_mode: str,
  bg_class: int,
  void_class: int = 255,
):
  criterion_c, criterion_c2s, criterion_s2c = criterions

  # Forward
  outputs = model(images.to(DEVICE), with_saliency=True, with_mask=True, with_rep=True)

  # (1) Classification loss.
  loss_c = criterion_c(outputs["logits_c"], label_smoothing(targets, ls).to(DEVICE))
  logit_seg = outputs["masks_seg"].detach()
  probs_seg = torch.softmax(logit_seg, dim=1)

  # (2) Segmentation loss.
  if c2s_mode == "gt":  # only a test.
    pseudo_masks = masks
  else:
    pseudo_masks = get_pseudo_label(
      images,
      targets,
      outputs["features_c"],
      probs_seg[:, bg_class],
      thresholds,
      resize_align_corners=True,
      mode=c2s_mode,
      c2s_sigma=c2s_sigma,
    )

  pixels_u = pseudo_masks == void_class
  pixels_b = pseudo_masks == bg_class
  pixels_fg = ~(pixels_u | pixels_b)
  samples_fg = pixels_fg.sum((1, 2)) > 0

  conf_pixels_c2s = (~pixels_u.cpu()).float().mean()

  if samples_fg.sum() == 0:
    # All label maps have only bg or void class.
    loss_c2s = torch.zeros_like(loss_c)
  else:
    loss_c2s = criterion_c2s(outputs["masks_seg"][samples_fg, ...], pseudo_masks[samples_fg, ...].to(DEVICE))

  # (3) Consistency loss.
  conf_pixels_s2c = torch.zeros_like(conf_pixels_c2s)

  if not w_s2c:
    loss_s2c = torch.zeros(()).to(loss_c)
  elif s2c_mode == "bce":
    prob_bg = 1 - probs_seg[:, bg_class].unsqueeze(1)
    loss_s2c = criterion_s2c(outputs["masks_sal"], prob_bg)
    # conf_pixels_s2c = (prob_bg - 0.5).abs() >= s2c_sigma  # conf >= 0.9
    conf_pixels_s2c = (prob_bg < (1 - s2c_sigma)) | (prob_bg >= s2c_sigma)

    loss_s2c = torch.sum(loss_s2c * conf_pixels_s2c) / conf_pixels_s2c.sum().clamp_(min=1.)

    conf_pixels_s2c = conf_pixels_s2c.float().mean()

  else:
    logit_sal = (
      outputs["masks_sal"]
      if model.use_sal_head
      else logit_seg[:, bg_class].unsqueeze(1)
    )

    feats_c = resize_tensor(outputs["features_c"], logit_sal.shape[2:], "bilinear", align_corners=True)
    masks_c = torch.concat((logit_sal, feats_c), dim=1)  # B(C+1)HW
    conf_pixels_s2c += 1.0

    if s2c_mode == "kld":
      ## KL Divergence Loss. (KLD)
      T = s2c_sigma

      loss_s2c = (T ** 2) * criterion_s2c(
        torch.log_softmax(masks_c / T, dim=1),
        torch.softmax(logit_seg / T, dim=1)
      )

    if s2c_mode == "mp":
      # Branches Mutual Promotion
      # https://arxiv.org/pdf/2308.04949.pdf

      pmasks_sal = probs_seg.argmax(dim=1)
      conf_pixels_s2c = probs_seg.max(1)[0] > s2c_sigma
      pmasks_sal[~conf_pixels_s2c] = void_class
      loss_s2c = criterion_c2s(masks_c, pmasks_sal)

      conf_pixels_s2c = conf_pixels_s2c.float().mean()

  # (4) Uncertain loss. (Push outputs for classes not in `targets` down.)
  if not w_u or pixels_u.sum() == 0:
    loss_u = torch.zeros(()).to(loss_c)
  else:
    ...  # outputs["rep"]
    # y_n = to_2d(F.pad(1 - targets, (1, 0), value=0.0)).to(DEVICE)  # Mark background=0 (occurs). # B, C
    # loss_u = torch.clamp(1 - torch.sigmoid(segms), min=0.0005, max=0.9995)
    # loss_u = (-y_n * torch.log(loss_u)).sum(dim=1)
    # loss_u = loss_u[pixels_u].mean()

  loss = w_c * loss_c + w_c2s * loss_c2s + w_s2c * loss_s2c + w_u * loss_u

  metrics = {
    "loss": loss,
    "loss_c": loss_c,
    "loss_c2s": loss_c2s,
    "loss_s2c": loss_s2c,
    "loss_u": loss_u,
    "conf_pixels_s2c": conf_pixels_s2c,
    "conf_pixels_c2s": conf_pixels_c2s,
  }

  return loss, metrics


def valid_step(
    model: torch.nn.Module,
    loader: DataLoader,
    data_source: datasets.CustomDataSource,
    thresholds: List[float],
    device: str,
    log_samples: bool = True,
    max_steps: Optional[int] = None,
):
  info_cls = data_source.classification_info
  info_seg = data_source.segmentation_info

  meters_cam = {t: MIoUCalculator.from_dataset_info(info_cls) for t in thresholds}
  meters_seg = MIoUCalculator.from_dataset_info(info_seg)

  start = time.time()

  preds_ = []
  targets_ = []

  with torch.no_grad():
    for step, (ids, inputs, targets, masks) in enumerate(loader):
      targets = to_numpy(targets)
      masks = to_numpy(masks)

      outputs = model(inputs.to(device), with_cam=True, with_mask=True)
      logits, features, pred_masks = outputs["logits_c"], outputs["features_c"], outputs["masks_seg"]

      preds = to_numpy(torch.sigmoid(logits.cpu().float()))
      cams = to_numpy(make_cam(features.cpu().float())) * targets[..., np.newaxis, np.newaxis]
      cams = cams.transpose(0, 2, 3, 1)

      pred_masks = torch.argmax(pred_masks, dim=1).cpu()
      pred_masks = to_numpy(pred_masks)

      preds_.append(preds)
      targets_.append(targets)

      meters_seg.add_many(pred_masks, masks)
      accumulate_batch_iou_priors(masks, cams, meters_cam, include_bg=info_cls.bg_class is None)

      if step == 0 and log_samples:
        inputs = to_numpy(inputs)
        wandb_utils.log_cams(ids, inputs, targets, cams, preds, classes=info_cls.classes, tag="val/priors")
        wandb_utils.log_masks(ids, inputs, targets, masks, pred_masks, info_seg.classes, info_seg.void_class, tag="val/segmentation")

      if max_steps and step >= max_steps:
        break

  elapsed = time.time() - start

  miou, miou_fg, iou, FP, FN = meters_seg.get(clear=True, detail=True)
  iou = [round(iou[c], 2) for c in info_seg.classes]

  preds_ = np.concatenate(preds_, axis=0)
  targets_ = np.concatenate(targets_, axis=0)

  try:
    precision, recall, f_score, _ = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average="macro")
    roc = skmetrics.roc_auc_score(targets_, preds_, average="macro")
  except ValueError:
    precision = recall = f_score = roc = 0.

  results_cam = maximum_miou_from_thresholds(meters_cam)
  results_seg = {
    "miou": miou,
    "miou_fg": miou_fg,
    "iou": iou,
    "fp": FP,
    "fn": FN,
    "time": elapsed,
  }
  results_clf = {
    "precision": round(100 * precision, 3),
    "recall": round(100 * recall, 3),
    "f_score": round(100 * f_score, 3),
    "roc_auc": round(100 * roc, 3),
  }

  results = {}
  results.update({f"priors/{k}": v for k, v in results_cam.items()})
  results.update({f"segmentation/{k}": v for k, v in results_seg.items()})
  results.update({f"classification/{k}": v for k, v in results_clf.items()})
  return results


def get_pseudo_label(images, targets, cams, masks_bg, thresholds, resize_align_corners=None, mode="cam", c2s_sigma=0.5, clf_t=0.5):
  sizes = images.shape[2:]

  cams = cams.cpu().to(torch.float32)
  cams *= to_2d(targets)
  cams = make_cam(cams, inplace=True, global_norm=False)
  cams = resize_tensor(cams, sizes, align_corners=resize_align_corners)
  cams = to_numpy(cams)

  targets = to_numpy(targets)
  targets_b = targets > clf_t

  images = to_numpy(images)
  masks_bg = to_numpy(masks_bg)

  with Pool(processes=len(images)) as pool:
    if mode == "cam":
      _fn = _get_pseudo_label_from_cams
      _args = [(i, c, t, thresholds) for i, c, t in zip(images, cams, targets_b)]
    else:
      _fn = _get_pseudo_label_from_mutual_promotion
      _args = [(i, c, t, m, c2s_sigma) for i, c, t, m in zip(images, cams, targets_b, masks_bg)]

    masks = pool.map(_fn, _args)

  return torch.as_tensor(np.asarray(masks, dtype="int64"))


def _get_pseudo_label_from_cams(args):
  image, cam, target, (bg_t, fg_t) = args
  image = denormalize(image, *datasets.imagenet_stats())

  cam = cam[target]
  labels = np.concatenate(([0], np.where(target)[0]+1))

  fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=fg_t)
  fg_cam = np.argmax(fg_cam, axis=0)
  fg_conf = labels[crf_inference_label(image, fg_cam, n_labels=labels.shape[0])]

  bg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=bg_t)
  bg_cam = np.argmax(bg_cam, axis=0)
  bg_conf = labels[crf_inference_label(image, bg_cam, n_labels=labels.shape[0])]

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

if __name__ == '__main__':
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  if args.validate_thresholds:
    THRESHOLDS = list(map(float, args.validate_thresholds.split(",")))

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  model_path = f"./experiments/models/{TAG}.pth"
  create_directory(os.path.dirname(model_path))

  set_seed(SEED)

  train_singlestage(args, wb_run, model_path)
