import torch

from .torch_utils import *


class PolyOptimizer(torch.optim.SGD):

  def __init__(self, params, lr, max_step, momentum=0.9):
    super().__init__(params, lr)

    self.global_step = 0
    self.max_step = max_step
    self.momentum = momentum

    self.__initial_lr = [group['lr'] for group in self.param_groups]

  def step(self, closure=None):
    if self.global_step < self.max_step:
      lr_mult = (1 - self.global_step / self.max_step)**self.momentum

      for i in range(len(self.param_groups)):
        self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

    super().step(closure)

    self.global_step += 1


def get_optimizer(lr, wd, max_step, param_groups):
  return PolyOptimizer(
    [
      {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
      {'params': param_groups[1], 'lr': 2 * lr, 'weight_decay': 0},
      {'params': param_groups[2], 'lr': 10 * lr, 'weight_decay': wd},
      {'params': param_groups[3], 'lr': 20 * lr, 'weight_decay': 0},
    ],
    lr=lr,
    momentum=0.9,
    max_step=max_step,
  )


def linear_schedule(step, max_step, a_0=0., a_n=1.0, schedule=1.0, contraint=min):
  if schedule == 0:
    return a_n

  rate = a_0 + (a_n - a_0) * step / (max_step * schedule)
  rate = contraint(rate, a_n)
  return rate
