import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool

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

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--print_ratio', default=0.05, type=float)
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
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

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
  tt, tv = datasets.get_classification_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset = datasets.ClassificationDataset(ts, transform=tt)
  valid_dataset = datasets.SegmentationDataset(vs, transform=tv)
  train_dataset = datasets.apply_augmentation(train_dataset, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
  train_iterator = datasets.Iterator(train_loader)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_val = len(train_loader)
  step_log = int(max(step_val * args.print_ratio, 1))
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f"Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  criterions = (
    torch.nn.MultiLabelSoftMarginLoss().to(DEVICE),
    torch.nn.CrossEntropyLoss(ignore_index=255, label_smoothing=args.label_smoothing).to(DEVICE),
    torch.nn.MultiLabelSoftMarginLoss().to(DEVICE),
  )

  # Network
  model = SingleStageModel(
    args.architecture,
    train_dataset.info.num_classes,
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
    int(step_max // args.accumulate_steps),
    param_groups,
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

  for step in tqdm(range(step_init, step_max), 'Training', mininterval=2.0, disable=not args.progress):
    _, images, targets = train_iterator.get()

    # bg_t = linear_schedule(step, step_max, 0.001, 0.3, 1.0)
    # fg_t = linear_schedule(step, step_max, 0.5, 0.3, 1.0, constraint=max)
    bg_t = 0.05
    fg_t = 0.3
    w_s = linear_schedule(step, step_max, 0.5, 1.0, 0.5)
    w_u = linear_schedule(step, step_max, 0.5, 1.0, 0.5)
    w_b = linear_schedule(step, step_max, 0.0, 1.0, 0.5)
    # w_s = w_u = w_b = 0.0

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      loss, metrics = train_step(
        step,
        model,
        images,
        targets,
        criterions,
        bg_t=bg_t,
        fg_t=fg_t,
        resize_align_corners=None,
        ls=args.label_smoothing,
        w_s=w_s,
        w_u=w_u,
        w_b=w_b,
      )

    scaler.scale(loss / args.accumulate_steps).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad(set_to_none=True)

    train_meter.update({m: v.item() for m, v in metrics.items()})

    epoch = step // step_val
    do_logging = (step + 1) % step_log == 0
    do_validation = args.validate and (step + 1) % step_val == 0

    if do_logging:
      learning_rate = float(get_learning_rate_from_optimizer(optimizer))

      loss, loss_c, loss_s, loss_b, loss_u = train_meter.get(clear=True)
      data = {
        'iteration': step + 1,
        'learning_rate': learning_rate,
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
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not do_validation)

      print(
        "step={iteration:,} "
        "lr={learning_rate:.4f} "
        "loss={loss:.4f} "
        "loss_c={loss_c:.4f} "
        "loss_s={loss_s:.4f} "
        "loss_b={loss_b:.4f} "
        "loss_u={loss_u:.4f} "
        "time={time:.0f}sec".format(**data)
      )

    if do_validation:
      model.eval()
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        metric_results = valid_step(
          model,
          valid_loader,
          valid_dataset.info,
          DEVICE,
          log_samples=True,
        )
      metric_results["iteration"] = step + 1
      model.train()

      wandb.log({f"val/{k}": v for k, v in metric_results.items()})
      print(*(f"{metric}={value}" for metric, value in metric_results.items()))

      if metric_results["miou"] > miou_best:
        miou_best = metric_results["miou"]
        for k in ("miou", "iou"):
          wandb.run.summary[f"val/best_{k}"] = metric_results[k]

      save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()


def train_step(
  step,
  model,
  images,
  targets,
  criterions,
  bg_t=0.05,
  fg_t=0.3,
  resize_align_corners=True,
  ls=0.0,
  w_s=1.0,
  w_u=1.0,
  w_b=1.0,
):
  criterion_c, criterion_s, criterion_b = criterions

  # Forward
  logits, features, logits_saliency, logits_segm, logits_segm_res = model(
    images.to(DEVICE), with_cam=True, with_mask=True, with_saliency=True
  )

  # (1) Classification loss.
  loss_c = criterion_c(logits, label_smoothing(targets, ls).to(DEVICE))

  # (2) Segmentation loss.
  pseudo_masks = get_pseudo_label(images, features, targets, bg_t, fg_t, resize_align_corners)

  pixels_u = pseudo_masks == 255
  pixels_b = pseudo_masks == 0
  pseudo_masks[pixels_b] = 255  # Ignore BG.

  loss_s = criterion_s(logits_segm_res, pseudo_masks.to(DEVICE))

  loss_u = loss_b = torch.zeros(())
  # loss_u = 0

  # # (3) BG loss.
  # pseudo_saliencies = torch.softmax(logits_segm.detach(), dim=1)[:, 1:].max(dim=1, keepdim=True)[0]
  # logits_saliency = resize_tensor(logits_saliency, pseudo_saliencies.shape[2:], align_corners=True)
  # loss_b = criterion_b(logits_saliency, pseudo_saliencies)

  # # (4) Uncertain loss. (Push outputs for classes not in `targets` down.)
  # y_n = to_2d(F.pad(1 - targets, (1, 0), value=0.0)).to(DEVICE)  # Mark background=0 (occurs). # B, C

  # if pixels_u.sum() == 0:
  #   loss_u = 0
  # else:
  #   loss_u = torch.clamp(1 - torch.sigmoid(logits_segm_res), min=0.0005, max=0.9995)
  #   loss_u = -y_n * torch.log(loss_u)  # [A, B]
  #   loss_u = loss_u.sum(dim=1) / y_n.sum(dim=1).clamp(min=1.0)
  #   loss_u = loss_u[pixels_u].mean()

  # if torch.isnan(loss_c):
  #   print(f"step={step} loss_c={loss_c} loss_s={loss_s} loss_u={loss_u} loss_b={loss_b}")
  #   loss_c = 0
  # if torch.isnan(loss_s):
  #   print(f"step={step} loss_c={loss_c} loss_s={loss_s} loss_u={loss_u} loss_b={loss_b}")
  #   loss_s = 0
  # if torch.isnan(loss_u):
  #   print(f"step={step} loss_c={loss_c} loss_s={loss_s} loss_u={loss_u} loss_b={loss_b}")
  #   loss_u = 0
  # if torch.isnan(loss_b):
  #   print(f"step={step} loss_c={loss_c} loss_s={loss_s} loss_u={loss_u} loss_b={loss_b}")
  #   loss_b = 0

  loss = loss_c + w_s * loss_s  + w_u * loss_u + w_b * loss_b

  losses_values = {
    "loss": loss,
    "loss_c": loss_c,
    "loss_s": loss_s,
    "loss_b": loss_b,
    "loss_u": loss_u,
  }

  return loss, losses_values

def valid_step(
    model: torch.nn.Module,
    loader: DataLoader,
    info: datasets.DatasetInfo,
    device: str,
    log_samples: bool = True,
    max_steps: Optional[int] = None,
):
  classes = [c for i, c in enumerate(info.classes) if i != info.void_class]

  start = time.time()
  meter = MIoUCalculator(classes, bg_class=info.bg_class, include_bg=False)

  with torch.no_grad():
    for step, (ids, inputs, targets, masks) in enumerate(loader):
      _, H, W = masks.shape

      logits, mask, mask_resized = model(inputs.to(device), with_mask=True)
      preds = torch.argmax(mask_resized, dim=1).cpu()
      preds = resize_tensor(preds.float().unsqueeze(1), (H, W), mode="nearest").squeeze().to(masks)
      preds = to_numpy(preds)
      masks = to_numpy(masks)

      meter.add_many(preds, masks)

      if step == 0 and log_samples:
        inputs = to_numpy(inputs)
        wandb_utils.log_masks(ids, inputs, targets, masks, preds, info.classes, void_class=info.void_class)

      if max_steps and step >= max_steps:
        break

  miou, miou_fg, iou, FP, FN = meter.get(clear=True, detail=True)
  iou = [round(iou[c], 2) for c in classes]

  elapsed = time.time() - start

  return {
    "miou": miou,
    "miou_fg": miou_fg,
    "iou": iou,
    "fp": FP,
    "fn": FN,
    "time": elapsed,
  }


def get_pseudo_label(images, cams, targets, bg_t=0.05, fg_t=0.4, resize_align_corners=None):
  sizes = images.shape[2:]

  cams = cams.cpu().to(torch.float32) * to_2d(targets)
  cams = make_cam(cams, inplace=True, global_norm=True)
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
  keys = np.concatenate(([0], np.where(labels)[0]))

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
