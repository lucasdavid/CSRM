# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np

from tools.ai.torch_utils import calculate_parameters
from tools.general.txt_utils import add_txt


def log_config(args, title=None, print_fn=print):
  margin = '=' * 80
  print_fn(margin)
  if title:
    print_fn(title)
    print_fn()

  pad = max(map(len, args)) + 1
  for k, v in args.items():
    print_fn(f'{k.ljust(pad)}: {v}')

  print_fn(margin)


def log_dataset(name, dataset, tt, tv):
  print(
    f"Dataset {name}\n"
    f"  samples={len(dataset)}"
    f"  classes={dataset.info.num_classes}\n"
    f"  train transforms={tt}\n"
    f"  valid transforms={tv}\n\n"
  )


def log_model(name, model, args):
  properties = "architecture mode dilated regularization trainable_stem use_gn".split()
  config = [(p, getattr(args, p)) for p in properties if hasattr(args, p)]

  print(
    f"{name}\n",
    *(f"{p:>14} = {v}\n" for p, v in config),
    f"  total params   = {calculate_parameters(model):.2f}M\n"
  )


def log_opt_params(name, params):
  print(
    f"{name} Parameters\n"
    f"  pretrained:\n    w={len(params[0])} b={len(params[1])}\n"
    f"  scratch:\n    w={len(params[2])} b={len(params[3])}\n"
  )


def log_print(message='', path=None):
  """This function shows message and saves message.
    
    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.
        
        gt_tags:
            The type of variable is list.
            the type of each element is string.
    """
  print(message, flush=True)
  if path:
    add_txt(path, message)


class Logger:

  def __init__(self):
    pass


class MetricsContainer:

  def __init__(self, metric_names):
    self.metric_names = metric_names
    self.clear()

  def update(self, data):
    for n, v in data.items():
      self._metrics_history[n].append(v)

  def get(self, metrics=None, clear=False):
    if metrics is None:
      metrics = self.metric_names

    metric_values = [float(np.mean(self._metrics_history[k])) for k in metrics]

    if clear:
      self.clear()

    if len(metric_values) == 1:
      metric_values = metric_values[0]

    return metric_values

  def clear(self):
    self._metrics_history = {n: [] for n in self.metric_names}
