import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def get_world_size():
  if not dist.is_available():
    return 1
  if not dist.is_initialized():
    return 1
  return dist.get_world_size()


@torch.no_grad()
def gather_together(data):
  dist.barrier()

  world_size = get_world_size()
  gather_data = [None for _ in range(world_size)]
  dist.all_gather_object(gather_data, data)

  return gather_data


def get_rank():
  if not dist.is_available():
    return 0
  if not dist.is_initialized():
    return 0
  return dist.get_rank()


def is_main_process():
  return get_rank() == 0


def synchronize():
  """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
  if not dist.is_available():
    return
  if not dist.is_initialized():
    return
  world_size = dist.get_world_size()
  if world_size == 1:
    return
  dist.barrier()


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
  # gather keys before updating queue
  keys = keys.detach().clone().cpu()
  gathered_list = gather_together(keys)
  keys = torch.cat(gathered_list, dim=0).cuda()

  batch_size = keys.shape[0]

  ptr = int(queue_ptr)

  queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
  if queue[0].shape[0] >= queue_size:
    queue[0] = queue[0][-queue_size:, :]
    ptr = queue_size
  else:
    ptr = (ptr + batch_size) % queue_size  # move pointer

  queue_ptr[0] = ptr

  return batch_size


def compute_unsupervised_loss(predict, target, percent, pred_teacher):
  batch_size, num_class, h, w = predict.shape

  with torch.no_grad():
    # drop pixels with high entropy
    prob = torch.softmax(pred_teacher, dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

    thresh = np.percentile(entropy[target != 255].detach().cpu().numpy().flatten(), percent)
    thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

    target[thresh_mask] = 255
    weight = batch_size * h * w / torch.sum(target != 255)

  loss = weight * F.cross_entropy(predict, target, ignore_index=255)  # [10, 321, 321]

  return loss


def compute_contra_memobank_loss(
  rep,
  label_l,
  label_u,
  prob_l,
  prob_u,
  low_mask,
  high_mask,
  cfg,
  memobank,
  queue_ptrlis,
  queue_size,
  rep_teacher,
  momentum_prototype=None,
  i_iter=0,
):
  # current_class_threshold: delta_p (0.3)
  # current_class_negative_threshold: delta_n (1)
  current_class_threshold = cfg["current_class_threshold"]
  current_class_negative_threshold = cfg["current_class_negative_threshold"]
  low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]
  temp = cfg["temperature"]
  num_queries = cfg["num_queries"]
  num_negatives = cfg["num_negatives"]

  num_feat = rep.shape[1]
  num_labeled = label_l.shape[0]
  num_segments = label_l.shape[1]

  low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
  high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

  rep = rep.permute(0, 2, 3, 1)
  rep_teacher = rep_teacher.permute(0, 2, 3, 1)

  seg_feat_all_list = []
  seg_feat_low_entropy_list = []  # candidate anchor pixels
  seg_num_list = []  # the number of low_valid pixels in each class
  seg_proto_list = []  # the center of each class

  _, prob_indices_l = torch.sort(prob_l, 1, True)
  prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

  _, prob_indices_u = torch.sort(prob_u, 1, True)
  prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

  prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

  valid_classes = []
  new_keys = []
  for i in range(num_segments):
    low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
    high_valid_pixel_seg = high_valid_pixel[:, i]

    prob_seg = prob[:, i, :, :]
    rep_mask_low_entropy = (prob_seg > current_class_threshold) * low_valid_pixel_seg.bool()
    rep_mask_high_entropy = (prob_seg < current_class_negative_threshold) * high_valid_pixel_seg.bool()

    seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
    seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

    # positive sample: center of the class
    seg_proto_list.append(torch.mean(rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True))

    # generate class mask for unlabeled data
    # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
    class_mask_u = torch.sum(prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3).bool()

    # generate class mask for labeled data
    # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
    # prob_i_classes = prob_indices_l[label_l_mask]
    class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

    class_mask = torch.cat((class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0)

    negative_mask = rep_mask_high_entropy * class_mask

    keys = rep_teacher[negative_mask].detach()
    new_keys.append(
      dequeue_and_enqueue(
        keys=keys,
        queue=memobank[i],
        queue_ptr=queue_ptrlis[i],
        queue_size=queue_size[i],
      )
    )

    if low_valid_pixel_seg.sum() > 0:
      seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
      valid_classes.append(i)

  if (len(seg_num_list) <= 1):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
    if momentum_prototype is None:
      return new_keys, torch.tensor(0.0) * rep.sum()
    else:
      return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

  else:
    reco_loss = torch.tensor(0.0).cuda()
    seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
    valid_seg = len(seg_num_list)  # number of valid classes

    prototype = torch.zeros((prob_indices_l.shape[-1], num_queries, 1, num_feat)).cuda()

    for i in range(valid_seg):
      if (len(seg_feat_low_entropy_list[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0):
        # select anchor pixel
        seg_low_entropy_idx = torch.randint(len(seg_feat_low_entropy_list[i]), size=(num_queries,))
        anchor_feat = (seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda())
      else:
        # in some rare cases, all queries in the current query class are easy
        reco_loss = reco_loss + 0 * rep.sum()
        continue

      # apply negative key sampling from memory bank (with no gradients)
      with torch.no_grad():
        negative_feat = memobank[valid_classes[i]][0].clone().cuda()

        high_entropy_idx = torch.randint(len(negative_feat), size=(num_queries * num_negatives,))
        negative_feat = negative_feat[high_entropy_idx]
        negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)
        positive_feat = (
          seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1).cuda()
        )  # (num_queries, 1, num_feat)

        if momentum_prototype is not None:
          if not (momentum_prototype == 0).all():
            ema_decay = min(1 - 1 / i_iter, 0.999)
            positive_feat = (1 - ema_decay) * positive_feat + ema_decay * momentum_prototype[valid_classes[i]]

          prototype[valid_classes[i]] = positive_feat.clone()

        all_feat = torch.cat((positive_feat, negative_feat), dim=1)  # (num_queries, 1 + num_negative, num_feat)

      seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

      reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())

    if momentum_prototype is None:
      return new_keys, reco_loss / valid_seg
    else:
      return prototype, new_keys, reco_loss / valid_seg
