import argparse
from copy import deepcopy
import os
import time
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import sklearn.metrics as skmetrics
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
from singlestage import CSRM, reco, u2pl
from tools.ai.demo_utils import crf_inference_label, denormalize
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import get_optimizer, linear_schedule
from tools.ai.torch_utils import *
from tools.general import wandb_utils
from tools.general.io_utils import create_directory, str2bool
from tools.general.time_utils import Timer
from tools.ai.augment_utils import *

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
parser.add_argument('--sampler', default="default", type=str, choices=["default", "balanced"])

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str, choices=["normal", "fix"])
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--use_sal_head', default=False, type=str2bool)
parser.add_argument('--use_rep_head', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--restore', default=None, type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--max_steps', default=None, type=int)
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
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--nesterov', default=False, type=str2bool)
parser.add_argument('--lr_poly_power', default=0.9, type=float)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)
parser.add_argument('--grad_max_norm', default=None, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=0.5, type=float)
parser.add_argument('--mixup_prob', default=0.5, type=float)

# Cls <=> Seg
parser.add_argument('--c2s_mode', default="cam", choices=["cam", "mp", "gt"])
parser.add_argument('--c2s_sigma', default=0.5, type=float)
parser.add_argument('--s2c_sigma', default=0.5, type=float)
parser.add_argument('--warmup_epochs', default=1, type=int)

# RECO
parser.add_argument('--reco_strong_threshold', default=0.97, type=float)
parser.add_argument('--reco_temp', default=0.5, type=float)
parser.add_argument('--reco_num_queries', default=256, type=int)
parser.add_argument('--reco_num_negatives', default=512, type=int)

import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)
THRESHOLDS = list(np.arange(0.10, 0.50, 0.05))


class RecoAugSegmentationDataset(datasets.SegmentationDataset):

  def __init__(
    self,
    data_source: datasets.CustomDataSource,
    crop_size: Tuple[int, int] = (512, 512),
    scale_size: Tuple[float, float] = (1.0, 1.0),
    augmentation=False,
    **kwargs
  ):
    super().__init__(data_source, **kwargs)
    self.crop_size = crop_size
    self.scale_size = scale_size
    self.augmentation = augmentation

  def __getitem__(self, index):
    sample_id, label = self.get_valid_sample(index)
    image = self.data_source.get_image(sample_id)
    mask = self.data_source.get_mask(sample_id)
    image, mask = reco.transform(
      image,
      mask,
      crop_size=self.crop_size,
      scale_size=self.scale_size,
      augmentation=self.augmentation,
    )
    return sample_id, image, label, mask[0]


def train_u2pl(args, wb_run, model_path):
  tls = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train")
  tus = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train_aug")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
  train_l_dataset = RecoAugSegmentationDataset(tls, crop_size=(args.image_size,) * 2, augmentation=True, scale_size=(0.5, 1.5))
  train_u_dataset = RecoAugSegmentationDataset(tus, crop_size=(args.image_size,)*2, augmentation=False)
  valid_dataset = RecoAugSegmentationDataset(vs, crop_size=(args.image_size,) * 2, augmentation=False)

  if args.sampler == "default":
    sampler = None
    shuffle = True
  else:
    from sklearn.utils import compute_sample_weight
    from torch.utils.data import WeightedRandomSampler
    labels = np.asarray([tls.get_label(_id) for _id in tls.sample_ids])
    weights = compute_sample_weight("balanced", labels)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 153)
    sampler = WeightedRandomSampler(weights, len(train_l_dataset), replacement=True, generator=generator)
    shuffle = None

  train_l_loader = DataLoader(train_l_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, shuffle=shuffle, drop_last=True, pin_memory=True)
  train_u_loader = DataLoader(train_u_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True)
  log_dataset(args.dataset, train_l_dataset, reco.transform, reco.transform)

  step_val = args.max_steps or len(train_l_loader)
  step_log = max(int(step_val * args.print_ratio), 1)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f"Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  tqdm_custom = partial(tqdm, total=step_val, disable=not args.progress, mininterval=2.0, ncols=60)

  criterions = (
    torch.nn.MultiLabelSoftMarginLoss().to(DEVICE),
    torch.nn.CrossEntropyLoss(ignore_index=255, label_smoothing=args.label_smoothing).to(DEVICE),
  )

  # Network
  model = CSRM(
    args.architecture,
    num_classes=tls.classification_info.num_classes,
    num_classes_segm=tls.segmentation_info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    use_group_norm=args.use_gn,
    trainable_stem=args.trainable_stem,
    trainable_backbone=args.trainable_backbone,
    use_sal_head=args.use_sal_head,
    use_rep_head=args.use_rep_head,
    dropout=args.dropout,
  )

  if args.restore:
    print(f"Restoring weights from {args.restore}")
    state_dict = torch.load(args.restore, map_location="cpu")
    for m in model.state_dict():
      if m not in state_dict:
        print("    Skip init:", m)
    model.load_state_dict(state_dict, strict=False)
  else:
    model.from_scratch_layers.append(model.classifier)
    model.initialize([model.classifier])

  log_model("CSRM", model, args)

  param_groups, param_names = model.get_parameter_groups(with_names=True)
  model = model.to(DEVICE)
  model.train()

  teacher = deepcopy(model)
  for p in teacher.parameters():
    p.requires_grad = False

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)
    teacher = torch.nn.DataParallel(teacher)

  # build class-wise memory bank
  num_classes_segm = tls.segmentation_info.num_classes
  memobank = []
  queue_ptrlis = []
  queue_size = []
  for i in range(num_classes_segm):
    memobank.append([torch.zeros(0, 256)])
    queue_size.append(30000)
    queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
  queue_size[0] = 50000

  # build prototype
  # prototype = torch.zeros((num_classes_segm, args.reco_num_queries, 1, 256)).to(DEVICE)
  prototype = None

  memory = (memobank, queue_size, queue_ptrlis, prototype)

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
  log_opt_params("CSRM", param_names)

  # Train
  train_meter = MetricsContainer("loss loss_c loss_c2s loss_s2c loss_u loss_contra conf_pixels_s2c conf_pixels_c2s".split())
  train_timer = Timer()
  miou_best = 0

  try:
    if args.first_epoch == 0:
      print("Checking the initial mIoU from pretrained CAM generating model...")
      miou_best, _ = valid_loop(model, valid_loader, tls, 0, optimizer, miou_best=0)

    for epoch in range(args.first_epoch, args.max_epoch):
      for step, (batch_l, batch_u) in enumerate(tqdm_custom(zip(train_l_loader, train_u_loader), desc=f"Epoch {epoch}")):
        c2s_mode = args.c2s_mode
        s2c_sigma = args.s2c_sigma
        c2s_sigma = args.c2s_sigma

        fg_t = 0.30
        bg_t = 0.05

        # Default
        w_c = w_c2s = 1

        if args.warmup_epochs and epoch < args.warmup_epochs:
          w_s2c = 0
          w_u = 0
          w_contra = 0
        else:
          # w_s2c = linear_schedule(optimizer.global_step, optimizer.max_step, 0.1, 1.0, 1.0)
          w_s2c = linear_schedule(epoch, args.max_epoch, 0.1, 1.0, 1.0)
          w_u = 1
          w_contra = 1

        # Others:
        # fg_t = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5,  0.2, 1.0, constraint=max)
        # bg_t = linear_schedule(optimizer.global_step, optimizer.max_step, 0.01, 0.2, 1.0)
        # w_c2s = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5, 1.0, 0.5)
        # w_u = linear_schedule(optimizer.global_step, optimizer.max_step, 0.5, 1.0, 0.5)

        with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
          loss, metrics = train_step(
            step,
            epoch,
            model,
            teacher,
            batch_l,
            batch_u,
            criterions,
            memory,
            thresholds=(bg_t, fg_t),
            ls=args.label_smoothing,
            w_c=w_c,
            w_c2s=w_c2s,
            w_s2c=w_s2c,
            w_u=w_u,
            w_contra=w_contra,
            c2s_mode=c2s_mode,
            s2c_sigma=s2c_sigma,
            c2s_sigma=c2s_sigma,
            num_classes=num_classes_segm,
            bg_class=tls.segmentation_info.bg_class,
            augment=args.augment,
            cutmix_prob=args.cutmix_prob,
            use_sal_head=args.use_sal_head,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epoch,
          )

        scaler.scale(loss / args.accumulate_steps).backward()

        if (step + 1) % args.accumulate_steps == 0:
          if args.grad_max_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_max_norm)
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()

          with torch.no_grad():
            if epoch <= args.warmup_epochs:
              for t_params, s_params in zip(teacher.parameters(), model.parameters()):
                t_params.copy_(s_params)
            else:
              ema_decay_origin = 0.99
              warmup_steps = args.warmup_epochs * step_val / args.accumulate_steps
              ema_decay = min(1 - 1 / (1 + optimizer.global_step - warmup_steps), ema_decay_origin)
              for t_params, s_params in zip(teacher.parameters(), model.parameters()):
                t_params.copy_(ema_decay * t_params + (1 - ema_decay) * s_params)

        train_meter.update({m: v.item() for m, v in metrics.items()})

        do_logging = (step + 1) % step_log == 0
        do_validation = args.validate and (step + 1) == step_val

        if do_logging:
          lr = float(get_learning_rate_from_optimizer(optimizer))

          loss, loss_c, loss_c2s, loss_s2c, loss_u, loss_contra, conf_pixels_s2c, conf_pixels_c2s = train_meter.get(clear=True)
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
            "loss_contra": loss_contra,
            "w_c2s": w_c2s,
            "w_s2c": w_s2c,
            "w_u": w_u,
            "w_contra": w_contra,
            "s2c_sigma": s2c_sigma,
            "c2s_sigma": c2s_sigma,
            "conf_pixels_s2c": conf_pixels_s2c,
            "conf_pixels_c2s": conf_pixels_c2s,
          }
          wandb.log({f"train/{k}": v for k, v in data.items()}, commit=not do_validation)
          print(
            f" loss={loss:.3f} loss_c={loss_c:.3f} loss_c2s={loss_c2s:.3f} loss_s2c={loss_s2c:.3f} loss_u={loss_u:.3f} loss_contra={loss_contra:.3f} lr={lr:.3f}"
          )

        if do_validation:
          # Interrupt epoch in case `max_steps < len(train_loader)`.
          break

      valid_model = model if epoch <= args.warmup_epochs else teacher
      miou_best, improved = valid_loop(valid_model, valid_loader, tls, epoch, optimizer, miou_best)
      save_model(valid_model, model_path, parallel=GPUS_COUNT > 1)

      if improved:
        save_model(valid_model, f"./experiments/models/{TAG}-best.pth", parallel=GPUS_COUNT > 1)

  except KeyboardInterrupt:
    print("training halted")

  wb_run.finish()


def valid_loop(model, valid_loader, ts, epoch, optimizer, miou_best, commit=True):
  model.eval()
  with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
    metric_results = valid_step(model, valid_loader, ts, THRESHOLDS, DEVICE, log_samples=True, max_steps=args.validate_max_steps)
    metric_results.update({"step": optimizer.global_step, "epoch": epoch})
  model.train()

  wandb.log({f"val/{k}": v for k, v in metric_results.items()}, commit=commit)
  print(f"[Epoch {epoch}/{args.max_epoch}]",
        *(f"{m}={v:.3f}" if isinstance(v, float) else f"{m}={v}"
          for m, v in metric_results.items()))

  improved = metric_results["segmentation/miou"] > miou_best
  if improved:
    miou_best = metric_results["segmentation/miou"]
    for k in ("miou", "iou"):
      wandb.run.summary[f"val/segmentation/best_{k}"] = metric_results[f"segmentation/{k}"]
  return miou_best, improved


def train_step(
  step: int,
  epoch: int,
  model: CSRM,
  teacher: CSRM,
  batch_l,
  batch_u,
  criterions,
  memory,
  thresholds,
  ls,
  w_c,
  w_c2s,
  w_s2c,
  w_u,
  w_contra,
  s2c_sigma: float,
  c2s_sigma: float,
  c2s_mode: str,
  num_classes: int,
  bg_class: int,
  augment: str = "none",
  cutmix_prob: float = 0.5,
  use_sal_head: bool = False,
  warmup_epochs: int = 0,
  max_epochs: int = 15,
  low_entropy_threshold: int = 20,
):
  _, images_l, true_labels_l, true_masks_l = batch_l
  _, images_u, true_labels_u, true_masks_u = batch_u
  NL = len(images_l)

  criterion_c, criterion_c2s = criterions
  memobank, queue_size, queue_ptrlis, prototype = memory

  images = torch.cat((images_l, images_u)).to(DEVICE)
  true_labels = torch.cat((true_labels_l, true_labels_u))
  true_masks = torch.cat((true_masks_l, true_masks_u))

  # Generate Pseudo-labels.
  teacher.eval()
  teacher_outputs = teacher(images, with_cam=True, with_saliency=False, with_rep=False)
  features_cls_t = teacher_outputs["features_c"].detach()
  logit_seg_large_t = teacher_outputs["masks_seg_large"].detach()

  features_cls_t_l, features_cls_t_u = features_cls_t[:NL], features_cls_t[NL:]
  # rep_t = teacher_outputs["rep"].detach()
  # if use_sal_head:
  #   logit_sal_t = teacher_outputs["masks_sal_large"].detach()
  # else:
  #   logit_sal_t = logit_seg_large_t[:, bg_class:bg_class + 1]
  del teacher_outputs

  if c2s_mode == "gt":  # only a sanity test.
    pseudo_masks = true_masks
  else:
    pseudo_masks = get_pseudo_label(
      images,
      true_labels,
      cams=features_cls_t,
      masks_bg=logit_seg_large_t[:, bg_class],
      thresholds=thresholds,
      resize_align_corners=True,
      mode=c2s_mode,
      c2s_sigma=c2s_sigma,
    )

  if "mix" in augment and np.random.uniform(0, 1) < cutmix_prob:
    mix = "cutmix" if "cutmix" in augment else "classmix"
    (
      images,
      true_labels,
      pseudo_masks,
      logit_seg_large_t,
      # logit_sal_t,
    ) = apply_mixaug(
      images,
      true_labels,
      pseudo_masks,
      logit_seg_large_t.cpu(),
      # logit_sal_t.cpu(),
      beta=1.0,
      mix=mix,
      ignore_class=bg_class,
    )

    logit_seg_large_t = logit_seg_large_t.float().to(DEVICE)
    # logit_sal_t = logit_sal_t.float().to(DEVICE)
    # rep_t = rep_t.float().to(DEVICE)

  pseudo_masks_l, pseudo_masks_u = pseudo_masks[:NL], pseudo_masks[NL:]
  logit_seg_large_l_t, logit_seg_large_u_t = logit_seg_large_t[:NL], logit_seg_large_t[NL:]
  label_seg_large_u_t = logit_seg_large_u_t.argmax(dim=1)

  # Student Forward
  outputs = model(images, with_saliency=True, with_mask=True, with_rep=True)
  logit_seg_large = outputs["masks_seg_large"]
  if use_sal_head:
    logit_sal = outputs["masks_sal_large"]
  else:
    logit_sal = logit_seg_large[:, bg_class].unsqueeze(1).detach()

  logit_seg_large_l, logit_seg_large_u = logit_seg_large[:NL], logit_seg_large[NL:]

  # Teacher Forward
  teacher.train()
  with torch.no_grad():
    teacher_outputs = teacher(images)
    pred_all_t, rep_all_t = teacher_outputs["masks_seg"], teacher_outputs["rep"]
    prob_all_t = F.softmax(pred_all_t, dim=1)
    prob_l_t, prob_u_t = prob_all_t[:NL], prob_all_t[NL:]
    logit_seg_large_u_t = teacher_outputs["masks_seg_large"][NL:]
    prob_large_u_t = F.softmax(logit_seg_large_u_t, dim=1)
    del teacher_outputs

  # (1) Classification loss.
  logit_c = outputs["logits_c"]
  # logit_c_l, logit_c_u = outputs["logits_c"][:num_labeled], outputs["logits_c"][num_labeled:]
  loss_c = criterion_c(logit_c, label_smoothing(true_labels, ls).to(DEVICE))

  # (2) Segmentation loss.
  pixels_un = pseudo_masks_l == 255
  pixels_bg = pseudo_masks_l == bg_class
  pixels_fg = ~(pixels_un | pixels_bg)
  samples_valid = pixels_fg.sum((1, 2)) > 0

  conf_pixels_c2s = (~pixels_un.cpu()).float().mean()

  pseudo_masks_l = pseudo_masks_l.to(DEVICE)

  if samples_valid.sum() == 0:
    # All label maps have only bg or void class.
    loss_c2s = torch.zeros_like(loss_c)
  else:
    loss_c2s = criterion_c2s(logit_seg_large_l[samples_valid], pseudo_masks_l[samples_valid])

  # (3) Mutual Promotion Branches Consistency loss.
  if not w_s2c:
    loss_s2c = torch.zeros_like(loss_c)
    conf_pixels_s2c = torch.zeros_like(conf_pixels_c2s)
  else:
    feats_c = resize_tensor(outputs["features_c"], logit_sal.shape[2:], "bilinear", align_corners=True)
    masks_c = torch.cat((logit_sal.to(feats_c), feats_c), dim=1)  # B(C+1)HW

    probs_seg_large_t = torch.softmax(logit_seg_large_t, dim=1)
    conf_pixels_s2c, pmasks_sal = probs_seg_large_t.max(1)

    conf_pixels_s2c = conf_pixels_s2c > s2c_sigma
    pmasks_sal[~conf_pixels_s2c] = 255

    loss_s2c = criterion_c2s(masks_c, pmasks_sal.to(DEVICE))

    conf_pixels_s2c = conf_pixels_s2c.float().mean()

  # (4) Unsupervised Loss.
  if not w_u:
    loss_u = torch.zeros_like(loss_c)
  else:
    drop_percent = 80
    percent_unreliable = (100 - drop_percent) * (1 - epoch / max_epochs)
    drop_percent = 100 - percent_unreliable
    loss_u = u2pl.compute_unsupervised_loss(
      logit_seg_large_u,
      label_seg_large_u_t.clone(),
      drop_percent,
      logit_seg_large_u_t.detach(),
    )

  # (5) Contrastive loss.
  if not w_contra:
    loss_contra = torch.zeros_like(loss_c)
  else:
    pred_all = outputs["masks_seg"]
    rep_all = outputs["rep"]

    alpha_t = low_entropy_threshold * (1 - epoch / max_epochs)

    with torch.no_grad():
      entropy = -torch.sum(prob_large_u_t * torch.log(prob_large_u_t + 1e-10), dim=1)
      low_thresh = np.percentile(entropy[label_seg_large_u_t != 255].cpu().numpy().flatten(), alpha_t)
      low_entropy_mask = (entropy < low_thresh) & (label_seg_large_u_t != 255)

      high_thresh = np.percentile(entropy[label_seg_large_u_t != 255].cpu().numpy().flatten(), 100 - alpha_t)
      high_entropy_mask = (entropy >= high_thresh) & (label_seg_large_u_t != 255)

      low_mask_all = torch.cat((
        (pseudo_masks_l.unsqueeze(1) != 255).float(),
        low_entropy_mask.unsqueeze(1),
      ))

      low_mask_all = F.interpolate(low_mask_all, size=pred_all.shape[2:], mode="nearest")  # down sample

      high_mask_all = torch.cat((
        (pseudo_masks_l.unsqueeze(1) != 255).float(),
        high_entropy_mask.unsqueeze(1),
      ))
      high_mask_all = F.interpolate(high_mask_all, size=pred_all.shape[2:], mode="nearest")  # down sample

      # down sample and concat
      label_l_small = F.interpolate(u2pl.label_onehot(pseudo_masks_l, num_classes), size=pred_all.shape[2:], mode="nearest")
      label_u_small = F.interpolate(u2pl.label_onehot(label_seg_large_u_t, num_classes), size=pred_all.shape[2:], mode="nearest")

    cfg_contra = dict(
      negative_high_entropy=True,
      low_rank=3,
      high_rank=20,
      current_class_threshold=0.3,
      current_class_negative_threshold=1,
      unsupervised_entropy_ignore=80,
      low_entropy_threshold=20,
      num_negatives=50,
      num_queries=256,
      temperature=0.5,
    )

    new_keys, loss_contra = u2pl.compute_contra_memobank_loss(
      rep_all,
      label_l_small.long(),
      label_u_small.long(),
      prob_l_t.detach(),
      prob_u_t.detach(),
      low_mask_all,
      high_mask_all,
      cfg_contra,
      memobank,
      queue_ptrlis,
      queue_size,
      rep_all_t.detach(),
    )

  loss = w_c * loss_c + w_c2s * loss_c2s + w_s2c * loss_s2c + w_u * loss_u + w_contra * loss_contra

  metrics = {
    "loss": loss,
    "loss_c": loss_c,
    "loss_c2s": loss_c2s,
    "loss_s2c": loss_s2c,
    "loss_u": loss_u,
    "loss_contra": loss_contra,
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
      logits, features, pred_masks = outputs["logits_c"], outputs["features_c"], outputs["masks_seg_large"]

      preds = to_numpy(torch.sigmoid(logits.cpu().float()))
      cams = to_numpy(make_cam(features.cpu().float())) * targets[..., None, None]
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
        wandb_utils.log_masks(
          ids, inputs, targets, masks, pred_masks, info_seg.classes, info_seg.void_class, tag="val/segmentation"
        )

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


def get_pseudo_label(
  images, targets, cams, masks_bg, thresholds, resize_align_corners=None, mode="cam", c2s_sigma=0.5, clf_t=0.5
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

  with Pool(processes=len(images)) as pool:
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

  train_u2pl(args, wb_run, model_path)
