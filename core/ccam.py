import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_simi(embedded_fg, embedded_bg):
  embedded_fg = F.normalize(embedded_fg, dim=1)
  embedded_bg = F.normalize(embedded_bg, dim=1)
  sim = torch.matmul(embedded_fg, embedded_bg.T)
  return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
  embedded_fg = F.normalize(embedded_fg, dim=1)
  embedded_bg = F.normalize(embedded_bg, dim=1)
  sim = torch.matmul(embedded_fg, embedded_bg.T)

  return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
  N, C = embedded_fg.size()

  # embedded_fg = F.normalize(embedded_fg, dim=1)
  # embedded_bg = F.normalize(embedded_bg, dim=1)

  embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
  embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

  return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C


# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):

  def __init__(self, metric='cos', reduction='mean'):
    super(SimMinLoss, self).__init__()
    self.metric = metric
    self.reduction = reduction

  def forward(self, embedded_bg, embedded_fg):
    """
    :param embedded_fg: [N, C]
    :param embedded_bg: [N, C]
    :return:
    """
    if self.metric == 'l2':
      raise NotImplementedError
    elif self.metric == 'cos':
      sim = cos_simi(embedded_bg, embedded_fg)
      loss = -torch.log(1 - sim)
    else:
      raise NotImplementedError

    if self.reduction == 'mean':
      return torch.mean(loss)
    elif self.reduction == 'sum':
      return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):

  def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
    super(SimMaxLoss, self).__init__()
    self.metric = metric
    self.alpha = alpha
    self.reduction = reduction

  def forward(self, embedded_bg):
    """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
    if self.metric == 'l2':
      raise NotImplementedError

    elif self.metric == 'cos':
      sim = cos_simi(embedded_bg, embedded_bg)
      loss = -torch.log(sim)
      loss[loss < 0] = 0
      _, indices = sim.sort(descending=True, dim=1)
      _, rank = indices.sort(dim=1)
      rank = rank - 1
      rank_weights = torch.exp(-rank.float() * self.alpha)
      loss = loss * rank_weights
    else:
      raise NotImplementedError

    if self.reduction == 'mean':
      return torch.mean(loss)
    elif self.reduction == 'sum':
      return torch.sum(loss)


class Disentangler(nn.Module):

  def __init__(self, cin):
    super(Disentangler, self).__init__()

    self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
    self.bn_head = nn.BatchNorm2d(1)

  def forward(self, x):
    N, C, H, W = x.size()
    ccam = self.bn_head(self.activation_head(x))
    ccam_ = torch.sigmoid(ccam).reshape(N, 1, H * W)  # [N, 1, H*W]
    x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()  # [N, H*W, C]
    fg_feats = torch.matmul(ccam_, x) / (H * W)  # [N, 1, C]
    bg_feats = torch.matmul(1 - ccam_, x) / (H * W)  # [N, 1, C]

    return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam
