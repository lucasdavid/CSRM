import os
from typing import List

import imageio
import numpy as np
import torch
from PIL import Image

from core.aff_utils import *
from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding
from tools.general.json_utils import read_json
from tools.general.xml_utils import read_xml

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'voc12')


def color_map(N=256):

  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

  cmap = np.zeros((N, 3), dtype=np.uint8)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    cmap[i] = np.array([b, g, r])

  return cmap


def get_color_map_dic():
  labels = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void'
  ]
  # colors = color_map()
  colors = np.asarray(
    [
      [0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128],
      [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64],
      [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128], [128, 64, 0], [192, 224, 224]
    ]
  )

  # n_classes = 21
  n_classes = len(labels)

  h = 20
  w = 500

  color_index_list = [index for index in range(n_classes)]

  cmap_dic = {label: colors[color_index] for label, color_index in zip(labels, range(n_classes))}
  cmap_image = np.empty((h * len(labels), w, 3), dtype=np.uint8)

  for color_index in color_index_list:
    cmap_image[color_index * h:(color_index + 1) * h, :] = colors[color_index]

  return cmap_dic, cmap_image, labels


class VOC12Dataset(torch.utils.data.Dataset):

  def __init__(
    self, root_dir,
    domain: str,
    masks_dir: str = None,
    with_image: bool = True,
    with_id: bool = False,
    with_tags: bool = False,
    with_mask: bool = False,
    sample_ids: List[str] = None,
  ):
    self.root_dir = root_dir
    self.image_dir = os.path.join(self.root_dir, 'JPEGImages/')
    self.xml_dir = os.path.join(self.root_dir, 'Annotations/')
    self.mask_dir = masks_dir or os.path.join(self.root_dir, 'SegmentationClass/')

    filepath = os.path.join(DATA_DIR, f"{domain}.txt")

    if sample_ids is not None:
      self.image_id_list = (
        sample_ids.split(",")
        if isinstance(sample_ids, str)
        else sample_ids
      )
    else:
      with open(filepath) as f:
        image_ids = f.readlines()
      self.image_id_list = [image_id.strip() for image_id in image_ids]

    data = read_json(os.path.join(DATA_DIR, "meta.json"))
    self.class_dic = data['class_dic']
    self.classes = data['classes']

    self.with_image = with_image
    self.with_id = with_id
    self.with_tags = with_tags
    self.with_mask = with_mask

  def __len__(self):
    return len(self.image_id_list)

  def get_image_path(self, image_id):
    return os.path.join(self.image_dir, image_id + '.jpg')

  def get_image(self, image_id):
    return Image.open(self.get_image_path(image_id)).convert('RGB')

  def get_mask(self, image_id):
    mask_path = os.path.join(self.mask_dir, image_id + '.png')
    if os.path.isfile(mask_path):
      mask = Image.open(mask_path)
      if mask.mode == "RGB":
        # Correction for SBD dataset.
        mask = np.array(mask)
        mask = mask[..., 0] * 256**2 + mask[..., 1] * 256 + mask[..., 2]
        mask = np.argmax(mask[..., np.newaxis] == self.color_ids, axis=-1)
        mask[mask == 21] = 255
        mask = Image.fromarray(mask.astype('uint8'))
    else:
      mask = None
    return mask

  def get_tags(self, image_id):
    try:
      _, tags = read_xml(self.xml_dir + image_id + '.xml')
    except FileNotFoundError:
      tags = None

    return tags

  def __getitem__(self, index):
    image_id = self.image_id_list[index]
    entry = []
    if self.with_image:
      entry.append(self.get_image(image_id))
    if self.with_id:
      entry.append(image_id)
    if self.with_tags:
      entry.append(self.get_tags(image_id))

    if self.with_mask:
      entry.append(self.get_mask(image_id))

    return entry


class ClassificationDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None, sample_ids=None):
    super().__init__(root_dir, domain, with_tags=True, sample_ids=sample_ids)
    self.transform = transform

  def __getitem__(self, index):
    image, tags = super().__getitem__(index)

    if self.transform is not None:
      image = self.transform(image)

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, label


class CAMEvaluationDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None, masks_dir=None, sample_ids=None):
    super().__init__(root_dir, domain, masks_dir=masks_dir, with_tags=True, with_mask=True, sample_ids=sample_ids)
    self.transform = transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    # Reversed, because it was stored as BGR:
    self.color_ids = self.colors[:, 2] * 256**2 + self.colors[:, 1] * 256 + self.colors[:, 0]

  def __getitem__(self, index):
    image, tags, mask = super().__getitem__(index)

    if self.transform is not None:
      data = self.transform({'image': image, 'mask': mask})
      image, mask = data['image'], data['mask']

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, label, mask


class SegmentationDataset(CAMEvaluationDataset):

  def __getitem__(self, index):
    image, _, mask = super().__getitem__(index)
    return image, mask


class PathsDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, masks_dir=None, sample_ids=None):
    super().__init__(root_dir, domain, masks_dir=masks_dir, with_id=True, with_image=False, sample_ids=sample_ids)

  def __getitem__(self, index):
    image_id, = super().__getitem__(index)
    image_path = self.get_image_path(image_id)
    mask_path = os.path.join(self.mask_dir, image_id + '.png')

    return image_id, image_path, mask_path


class InferenceDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None, sample_ids=None):
    super().__init__(root_dir, domain, with_id=True, with_tags=True, sample_ids=sample_ids)
    self.transform = transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

  def __getitem__(self, index):
    image, image_id, tags = super().__getitem__(index)

    if self.transform is not None:
      image = self.transform(image)

    try:
      label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    except TypeError:
      label = None

    return image, image_id, label


class AffinityDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, path_index, label_dir, transform=None, sample_ids=None):
    super().__init__(root_dir, domain, with_id=True, sample_ids=sample_ids)

    self.transform = transform

    self.label_dir = label_dir
    self.path_index = path_index

    self.extract_aff_lab_func = GetAffinityLabelFromIndices(
      self.path_index.src_indices, self.path_index.dst_indices, classes=21
    )

  def __getitem__(self, idx):
    image, image_id = super().__getitem__(idx)

    label = imageio.imread(os.path.join(self.label_dir, image_id + '.png'))
    label = Image.fromarray(label)

    entry = self.transform({'image': image, 'mask': label})
    image, label = entry['image'], entry['mask']

    return image, self.extract_aff_lab_func(label)


class HRCAMsDataset(VOC12Dataset):

  IGNORE_BG_IMAGES = True

  def __init__(self, root_dir, domain, cams_dir, transform, sample_ids=None):
    super().__init__(root_dir, domain, with_id=True, with_tags=True, sample_ids=sample_ids)
    self.cams_dir = cams_dir
    self.transform = transform

  def __getitem__(self, index):
    image, image_id, tags = super().__getitem__(index)
    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)

    mask_path = os.path.join(self.cams_dir, f'{image_id}.npy')
    mask_pack = np.load(mask_path, allow_pickle=True).item()
    cams = torch.from_numpy(mask_pack['hr_cam'].max(0, keepdims=True))

    data = self.transform({'image': image, 'mask': cams})
    image, cams = data['image'], data['mask']

    return image, label, cams
