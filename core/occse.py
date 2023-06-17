import torch
import torch.nn.functional as F


def random_label_split(label, k, choices):
  bs = label.shape[0]
  y_mask = torch.zeros_like(label)
  indices = []

  for i in range(bs):
    label_idx = torch.nonzero(label[i] > 0.5)  # [0, 1, 14]
    rand_idx = torch.randperm(len(label_idx))[:k]  # [2]
    target = label_idx[rand_idx]  # [14]
    y_mask[i, target] = 1
    choices[target] += 1
    indices.append(target)

  return y_mask, indices


def balanced_label_split(label, k, choices, gamma=1.0):
  bs = label.shape[0]
  y_mask = torch.zeros_like(label)
  indices = []

  for i in range(bs):
    di = label[i] > 0.5
    ci = choices[di]
    p = 1 / ci.clip(min=1)**gamma  # inv. prop. to the number of times chosen
    p /= p.sum()

    targets = torch.where(di)[0]
    target = targets[torch.multinomial(p, k)]

    y_mask[i, target] = 1
    choices[target] += 1

    indices.append(target)

  return y_mask, indices


def least_chosen_label_split(label, k, choices, gamma=1.0):
  bs = label.shape[0]
  y_mask = torch.zeros_like(label)
  indices = []

  for i in range(bs):
    label_i = torch.nonzero(label[i] > 0.5).ravel()
    min_i = choices[label_i].argsort()[:k]
    target = label_i[min_i]

    y_mask[i, target] = 1
    choices[target] += 1

    indices.append(target)

  return y_mask, indices


def focal_label_split(label, k, choices, focal_factor):
  bs = label.shape[0]
  y_mask = torch.zeros_like(label)
  indices = []

  for i in range(bs):
    p = focal_factor  # prop. to the focal factor
    p = p * label[i]  # suppress if label not present

    target = torch.multinomial(p, k)
    y_mask[i, target] = 1
    choices[target] += 1

    indices.append(target)

  return y_mask, indices


STRATEGIES = ['random', 'balanced', 'focal', 'least_chosen']


def split_label(
  label,
  k,
  choices,
  focal_factor=None,
  strategy='random',  # args.label_split
):
  if strategy == 'random':
    y_mask, indices = random_label_split(label, k, choices)
  elif strategy == 'balanced':
    y_mask, indices = balanced_label_split(label, k, choices)
  elif strategy == 'focal':
    y_mask, indices = focal_label_split(label, k, choices, focal_factor)
  elif strategy == 'least_chosen':
    y_mask, indices = least_chosen_label_split(label, k, choices, focal_factor)
  else:
    raise ValueError('Only `random` and `focus` are available.')

  return y_mask, torch.cat(indices, dim=0)


def calculate_focal_factor(target, output, gamma=2.0, alpha=0.25, apply_class_balancing=False):
  output = torch.sigmoid(output)
  factor = target * output + (1 - target) * (1 - output)  # high if certain
  factor = (1.0 - factor)**gamma  # low if certain

  if apply_class_balancing:
    weight = target * alpha + (1 - target) * (1 - alpha)
    factor = weight * factor

  return factor.detach().float()


def update_focal(
  labels,
  labels_oc,
  cl_logits,
  focal_factor,
  momentum=0.9,
  gamma=2.0,
):
  samples = labels.sum(axis=0)
  valid = labels.any(axis=0)

  cf = calculate_focal_factor(labels_oc, cl_logits, gamma).sum(axis=0) / samples
  focal_factor[valid] = (momentum * focal_factor + (1 - momentum) * cf)[valid]

  return focal_factor


def _class_specific_mask(features, label_mask, sizes, eps=1e-5, globalnorm=True):
  mask = features[label_mask == 1, :, :].unsqueeze(1)
  mask = F.relu(mask)
  mask = F.interpolate(mask, sizes, mode='bilinear', align_corners=False)

  if globalnorm:
    # Normalize considering all samples/channels because that's OC-CSE's default policy.
    return mask / (mask.max() + eps)

  b, c = mask.size()[:2]
  mask_max = mask.view(b, c, -1).max(dim=-1)[0].view(b, c, 1, 1)
  return mask / (mask_max + eps)


def soft_mask_images(images, features, label_mask, eps=1e-5, globalnorm=True):
  sizes = images.size()[2:]
  mask = _class_specific_mask(features, label_mask, sizes, eps, globalnorm)

  return images * (1 - mask)


def hard_mask_images(images, features, label_mask, t=0.3, eps=1e-5, globalnorm=False):
  sizes = images.size()[2:]
  mask = _class_specific_mask(features, label_mask, sizes, eps, globalnorm)

  return images * (mask <= t)
