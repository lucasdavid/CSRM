import os

import cv2

import wandb
from core.datasets import imagenet_stats
from tools.ai.demo_utils import colormap, denormalize
from tools.general.txt_utils import add_txt


def setup(name, config, job_type="train", tags=None):
  wb_run = wandb.init(
    name=name,
    job_type=job_type,
    entity="lerdl",
    project="research-wsss",
    config=config,
    tags=tags,
  )

  return wb_run


def cams_to_wb_images(images, cams):
  wb_images, wb_cams = [], []

  mu_std = imagenet_stats()
  cams = cams.max(-1)

  for b in range(8):
    image = denormalize(images[b], *mu_std)
    img = image[..., ::-1]
    cam = colormap(cams[b], img.shape)
    cam = cv2.addWeighted(img, 0.5, cam, 0.5, 0)
    cam = cam[..., ::-1]

    wb_images.append(wandb.Image(image))
    wb_cams.append(wandb.Image(cam))

  return wb_images, wb_cams


def log_cams(
    classes,
    images,
    targets,
    cams,
    predictions,
    oc_predictions=None,
    commit=False,
):
  wb_images, wb_cams = cams_to_wb_images(images, cams)
  wb_targets = _predictions_to_names(targets, classes)
  wb_predics = _predictions_to_names(predictions, classes)
  wb_oc_pred = _predictions_to_names(oc_predictions, classes)

  columns = ("Image", "CAM", "Labels", "CG Predictions", "OC Predictions")
  entries = (wb_images, wb_cams, wb_targets, wb_predics, wb_oc_pred)

  columns_v, entries_v = [], []

  for c, e in zip(columns, entries):
    if e is not None:
      columns_v += [c]
      entries_v += [e]

  data = [list(row) for row in zip(*entries_v)]
  table = wandb.Table(columns=columns_v, data=data)

  wandb.log({
    "val/predictions": table,
    "val/cams": wb_cams
  }, commit=commit)


def _predictions_to_names(predictions, classes, threshold=0.5):
  if predictions is not None:
    return [classes[p > threshold].tolist() for p in predictions]
