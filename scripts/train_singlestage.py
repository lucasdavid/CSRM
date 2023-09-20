import argparse
import os
from multiprocessing import Pool
from typing import List

import numpy as np
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
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general import wandb_utils

parser = argparse.ArgumentParser()

parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--print_ratio', default=0.25, type=float)
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
parser.add_argument('--use_saliency_head', default=False, type=str2bool)
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
parser.add_argument('--optimizer', default="sgd", choices=["sgd", "lion"])
parser.add_argument('--momentum', default=.9, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

# Single-Stage
parser.add_argument('--loss_b', default='l1', choices=["l1", "kld"])


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
    torch.nn.L1Loss() if args.loss_b == "l1" else torch.nn.KLDivLoss(log_target=True, reduction="batchmean"),
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
    use_saliency_head=args.use_saliency_head,
    criterion_c=criterions[0],
    criterion_s=criterions[1],
    criterion_b=criterions[2],
  )

  if args.restore:
    print(f"Restoring weights from {args.restore}")
    state_dict = torch.load(args.restore, map_location="cpu")
    for m in model.state_dict():
      if m not in state_dict:
        print("    Skip init:", m)
    model.load_state_dict(state_dict, strict=False)
    model.from_scratch_layers.remove(model.classifier)
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
    alpha_bias=args.lr_alpha_bias
  )
  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
  log_opt_params("SingleStage", param_groups)
  print("", flush=True)

  # Train
  train_meter = MetricsContainer("loss loss_c loss_s loss_b loss_u".split())
  train_timer = Timer()
  miou_best = -1

  for epoch in range(args.first_epoch, args.max_epoch):
    for step, batch in enumerate(tqdm(train_loader, mininterval=2.0, disable=not args.progress, ncols=10)):
      _, images, targets, true_masks = batch

      # fg_t = linear_schedule(step, step_max, 0.5, 0.3, 1.0, constraint=max)
      # bg_t = linear_schedule(step, step_max, 0.001, 0.3, 1.0)
      fg_t = 0.40
      bg_t = 0.05
      # w_s = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5, 1.0, 0.5)
      w_s = 1
      # w_u = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5, 1.0, 0.5)
      # w_b = linear_schedule(optimizer.global_step, optimizer.max_step, 0.0, 1.0, 0.5)
      w_u = w_b = 0
      # w_s = w_u = w_b = 0.0

      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        loss, metrics = train_step(
          step,
          model,
          images,
          targets,
          true_masks,
          criterions,
          thresholds=(bg_t, fg_t),
          ls=args.label_smoothing,
          w_s=w_s,
          w_u=w_u,
          w_b=w_b,
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

        loss, loss_c, loss_s, loss_b, loss_u = train_meter.get(clear=True)
        data = {
          'step': step + 1,
          "epoch": epoch + 1,
          'lr': lr,
          'time': train_timer.tok(clear=True),
          "fg_t": fg_t,
          "bg_t": bg_t,
          "loss": loss,
          "loss_c": loss_c,
          "loss_s": loss_s,
          "loss_b": loss_b,
          "loss_u": loss_u,
          "w_s": w_s,
          "w_u": w_u,
          "w_b": w_b,
        }

        wb_logs = {f"train/{k}": v for k, v in data.items()}
        wandb.log(wb_logs, commit=not do_validation)
        print(f" loss={loss:.5f} loss_c={loss_c:.5f} loss_s={loss_s:.5f} loss_b={loss_b:.5f} loss_u={loss_u:.5f}")

    # region Evaluation
    model.eval()
    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      metric_results = valid_step(
        model,
        valid_loader,
        ts,
        THRESHOLDS,
        DEVICE,
        log_samples=True,
      )
      metric_results.update({"iteration": step+1, "epoch": epoch+1})
    model.train()

    wandb.log({f"val/{k}": v for k, v in metric_results.items()})
    print(f"[Epoch {epoch}/{args.max_epoch}]",
          *(f"{m}={v:.5f}" if isinstance(v, float) else f"{m}={v}"
            for m, v in metric_results.items()))

    if metric_results["segmentation/miou"] > miou_best:
      miou_best = metric_results["segmentation/miou"]
      for k in ("miou", "iou"):
        wandb.run.summary[f"val/segmentation/best_{k}"] = metric_results[f"segmentation/{k}"]
    # endregion

    save_model(model, model_path, parallel=GPUS_COUNT > 1)

  wb_run.finish()


def train_step(
  step,
  model: SingleStageModel,
  images,
  targets,
  masks,
  criterions,
  thresholds,
  ls=0.0,
  w_s=1.0,
  w_u=1.0,
  w_b=1.0,
  bg_class: int = 0,
  b_modes = (),
  t_b=1.0,
  sigma_b=0.5,
):
  criterion_c, criterion_s, criterion_b = criterions
  bg_t, fg_t = thresholds

  # Forward
  outputs = model(images.to(DEVICE), with_saliency=True, with_mask=True)
  # logits_c, features_c, masks_sal, masks_seg

  # (1) Classification loss.
  loss_c = criterion_c(outputs["logits_c"], label_smoothing(targets, ls).to(DEVICE))

  # (2) Segmentation loss.
  # pseudo_masks = get_pseudo_label(images, features, targets, bg_t, fg_t, resize_align_corners=True)
  pseudo_masks = masks

  if step == 0:
    import matplotlib.pyplot as plt
    source=datasets.custom_data_source("voc12", "/home/ldavid/workspace/datasets/VOCDevkit", "train")
    fig1=np.concatenate([denormalize(i, *datasets.imagenet_stats()) for i in images.numpy()], 1)
    fig2=np.concatenate(source.segmentation_info.colors[np.minimum(pseudo_masks, 21)], 1)
    plt.figure(figsize=(len(pseudo_masks) * 3, 3)); plt.imshow(fig1); plt.axis("off"); plt.tight_layout(); plt.savefig(f"experiments/visual-results/ss_step_{step}_image.jpg");
    plt.figure(figsize=(len(pseudo_masks) * 3, 3)); plt.imshow(fig2); plt.axis("off"); plt.tight_layout(); plt.savefig(f"experiments/visual-results/ss_step_{step}_pseudo_mask.jpg");

  pixels_u = pseudo_masks == 255
  pixels_b = pseudo_masks == bg_class
  pixels_fg = ~(pixels_u | pixels_b)
  samples_fg = pixels_fg.sum((1, 2)) > 0

  loss_s = criterion_s(outputs["masks_seg"][samples_fg], pseudo_masks[samples_fg].to(DEVICE))

  # (3) Consistency loss.
  loss_b = torch.zeros(())
  pprobs_sal = outputs["masks_seg"].detach()
  pprobs_sal = torch.softmax(pprobs_sal / t_b, dim=1)
  if not w_b:
    ...
  if 1 in b_modes:
    loss_b += criterion_b(outputs["masks_sal"], pprobs_sal[:, bg_class].unsqueeze(1))
  if 2 in b_modes:
    # Branches Mutual Promotion
    # https://arxiv.org/pdf/2308.04949.pdf

    pmasks_sal = pprobs_sal.argmax(dim=1)
    p_conf_sal = pprobs_sal.max(1, keepdim=True)[0] > sigma_b
    pmasks_sal[~p_conf_sal] = 255

    if model.use_saliency_head:
      masks_bg = outputs["masks_sal"]
    else:
      masks_bg = outputs["masks_seg"][:, bg_class].unsqueeze(1).detach()

    masks_c = torch.concat((masks_bg, outputs["features_c"]), dim=1)  # B(C+1)HW
    masks_c = resize_tensor(masks_c, pmasks_sal.shape[2:], "bilinear", align_corners=True)
    loss_b = criterion_b(masks_c, pmasks_sal)

  # (4) Uncertain loss. (Push outputs for classes not in `targets` down.)
  if not w_u or pixels_u.sum() == 0:
    loss_u = torch.zeros(())
  else:
    y_n = to_2d(F.pad(1 - targets, (1, 0), value=0.0)).to(DEVICE)  # Mark background=0 (occurs). # B, C
    loss_u = torch.clamp(1 - torch.sigmoid(segms), min=0.0005, max=0.9995)
    loss_u = (-y_n * torch.log(loss_u)).sum(dim=1)
    loss_u = loss_u[pixels_u].mean()

  loss = loss_c + w_s * loss_s  + w_u * loss_u + w_b * loss_b

  metrics = {
    "loss": loss,
    "loss_c": loss_c,
    "loss_s": loss_s,
    "loss_b": loss_b,
    "loss_u": loss_u,
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
  meters_seg = MIoUCalculator(info_cls.classes, bg_class=info_seg.bg_class, include_bg=False)

  start = time.time()

  preds_ = []
  targets_ = []

  with torch.no_grad():
    for step, (ids, inputs, targets, masks) in enumerate(loader):
      targets = to_numpy(targets)
      masks = to_numpy(masks)

      logits, features, pred_masks = model(inputs.to(device), with_cam=True, with_mask=True)

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
        wandb_utils.log_cams(ids, inputs, targets, cams, preds, classes=info_cls.classes)
        wandb_utils.log_masks(ids, inputs, targets, masks, pred_masks, info_seg.classes, info_seg.void_class)

      if max_steps and step >= max_steps:
        break

  elapsed = time.time() - start

  miou, miou_fg, iou, FP, FN = meters_seg.get(clear=True, detail=True)
  iou = [round(iou[c], 2) for c in info_cls.classes]

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


def get_pseudo_label(images, cams, targets, bg_t=0.05, fg_t=0.4, resize_align_corners=None):
  sizes = images.shape[2:]

  cams = cams.cpu().to(torch.float32) * to_2d(targets)
  cams = make_cam(cams, inplace=True, global_norm=False)
  cams = resize_tensor(cams, sizes, align_corners=resize_align_corners)
  cams = to_numpy(cams)

  images = to_numpy(images)
  targets = to_numpy(targets)

  with Pool(processes=len(images)) as pool:
    args = [(i, c, t, bg_t, fg_t) for i, c, t in zip(images, cams, targets)]
    masks = pool.map(_get_pseudo_label, args)

  return torch.as_tensor(np.asarray(masks, dtype="int64"))


def _get_pseudo_label(args):
  image, cam, target, bg_t, fg_t = args
  image = denormalize(image, *datasets.imagenet_stats())

  labels = target > 0.5
  cam = cam[labels]
  keys = np.concatenate(([0], np.where(labels)[0]+1))

  fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=fg_t)
  fg_cam = np.argmax(fg_cam, axis=0)
  fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]

  bg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=bg_t)
  bg_cam = np.argmax(bg_cam, axis=0)
  bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]

  mask = fg_conf.copy()
  mask[fg_conf == 0] = 255
  mask[bg_conf + fg_conf == 0] = 0

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
