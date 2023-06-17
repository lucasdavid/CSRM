import os

import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from core.aff_utils import *

MASKS_DIR = "coco_seg_anno"
IGNORE = 255

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'coco14')


def _decode_id(sample_id):
  s = str(sample_id).split('\n')[0]
  if len(s) != 12:
    s = '%012d' % int(s)
  return s


def _load_labels_from_npy(images, root_dir):
  filepath = os.path.join(root_dir, 'cls_labels_coco.npy')
  cls_labels_dict = np.load(filepath, allow_pickle=True).item()
  return np.array([cls_labels_dict[int(img_name)] for img_name in images])


def _load_images_list(domain):
  filepath = os.path.join(DATA_DIR, f"{domain}.txt")
  return np.loadtxt(filepath, dtype=np.int32)[::-1]


class COCO14Dataset(Dataset):

  IGNORE_BG_IMAGES = False

  def __init__(self, root_dir, domain, transform=None, sample_ids=None, ignore_bg_images=None):
    self.root_dir = root_dir
    self.domain = domain
    self.transform = transform
    if sample_ids is None:
      self.img_name_list = _load_images_list(domain)
    else:
      self.img_name_list = (
        sample_ids.split(",")
        if isinstance(sample_ids, str)
        else sample_ids
      )
    self.ignore_bg_images = (
      ignore_bg_images
      if ignore_bg_images is not None
      else self.IGNORE_BG_IMAGES
    )
    self.label_list = _load_labels_from_npy(self.img_name_list, root_dir)

  def __len__(self):
    return len(self.img_name_list)

  def __getitem__(self, idx):
    return self.load_sample_with_labels(idx)

  def get_image_path(self, image_id):
    return os.path.join(self.root_dir, self.domain, f"COCO_{self.domain}_{image_id}.jpg")

  def load_sample_with_labels(self, idx):
    label = self.label_list[idx]

    if self.ignore_bg_images and label.sum() == 0:
      return self.load_sample_with_labels(idx + 1)

    image_id = _decode_id(self.img_name_list[idx])
    file_path = self.get_image_path(image_id)
    image = Image.open(file_path).convert('RGB')

    return image_id, image, label


class ClassificationDataset(COCO14Dataset):

  IGNORE_BG_IMAGES = True

  def __getitem__(self, idx):
    _, image, label = super().__getitem__(idx)

    if self.transform:
      image = self.transform(image)

    return image, label


class InferenceDataset(COCO14Dataset):

  def __getitem__(self, idx):
    image_id, image, label = super().__getitem__(idx)

    if self.transform:
      image = self.transform(image)

    return image, image_id, label


class CAMEvaluationDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, transform=None, masks_dir=None, sample_ids=None):
    super().__init__(root_dir=root_dir, domain=domain, transform=transform, sample_ids=sample_ids)

    self.masks_dir = masks_dir or os.path.join(self.root_dir, MASKS_DIR)

  def __getitem__(self, idx):
    image_id, image, label = super().__getitem__(idx)

    maskpath = os.path.join(self.masks_dir, image_id + '.png')
    mask = Image.open(maskpath) if os.path.isfile(maskpath) else None

    if self.transform:
      entry = self.transform({'image': image, 'mask': mask})
      image, mask = entry['image'], entry['mask']

    return image, label, mask


class SegmentationDataset(CAMEvaluationDataset):

  def __getitem__(self, idx):
    image, _, mask = super().__getitem__(idx)

    return image, mask


class PathsDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, masks_dir=None, sample_ids=None, ignore_bg_images=None):
    super().__init__(root_dir=root_dir, domain=domain, sample_ids=sample_ids, ignore_bg_images=ignore_bg_images)

    self.masks_dir = masks_dir or os.path.join(self.root_dir, MASKS_DIR)

  def __getitem__(self, idx):
    label = self.label_list[idx]
    while self.ignore_bg_images and label.sum() == 0:
      idx += 1
      label = self.label_list[idx]

    image_id = _decode_id(self.img_name_list[idx])
    image_path = self.get_image_path(image_id)
    mask_path = os.path.join(self.masks_dir, image_id + '.png')

    return image_id, image_path, mask_path


class AffinityDataset(COCO14Dataset):

  IGNORE_BG_IMAGES = True

  def __init__(self, root_dir, domain, path_index, label_dir, transform=None, sample_ids=None):
    super().__init__(root_dir, domain, transform=transform, sample_ids=sample_ids)

    self.label_dir = label_dir
    self.path_index = path_index

    self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices, classes=81)

  def __getitem__(self, idx):
    image_id, image, _ = super().__getitem__(idx)

    label = imageio.imread(os.path.join(self.label_dir, image_id + '.png'))
    label = Image.fromarray(label)

    entry = self.transform({'image': image, 'mask': label})
    image, label = entry['image'], entry['mask']

    return image, self.extract_aff_lab_func(label)


class HRCAMsDataset(COCO14Dataset):

  IGNORE_BG_IMAGES = True

  def __init__(self, root_dir, domain, cams_dir, transform, sample_ids=None):
    super().__init__(root_dir, domain, transform, sample_ids=sample_ids)
    self.cams_dir = cams_dir

  def __getitem__(self, index):
    image_id, image, label = super().__getitem__(index)

    mask_path = os.path.join(self.cams_dir, f'{image_id}.npy')
    mask_pack = np.load(mask_path, allow_pickle=True).item()
    cams = torch.from_numpy(mask_pack['hr_cam'].max(0, keepdims=True))

    data = self.transform({'image': image, 'mask': cams})
    image, cams = data['image'], data['mask']

    return image, label, cams
