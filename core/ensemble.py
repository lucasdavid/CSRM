import numpy as np


def sum_st(parts, targets=None, weights=None, alpha=None):
  cam = np.sum([e[0] for e in parts], axis=0)
  return cam


def avg_st(parts, targets=None, weights=None, alpha=None):
  cam = np.mean([e[0] for e in parts], axis=0)
  return cam


def max_st(parts, targets=None, weights=None, alpha=None):
  cam = np.max([e[0] for e in parts], axis=0)
  return cam


def weighted_st(parts, targets, weights, alpha=0.25):
  if alpha is None: alpha = 0.25

  cam = []

  for ic, c in enumerate(np.where(targets > 0.5)[0]):
    wc = weights.loc[c + 1]

    wc = np.exp(wc * alpha)
    wc /= wc.sum()
    cam_c = np.sum([e[0][ic] * w for e, w in zip(parts, wc)], axis=0)

    cam.append(cam_c)

  cam = np.stack(cam, axis=0)
  return cam


def ranked_st(parts, targets, weights, alpha=2.0):
  if alpha is None: alpha = 2.0

  cam = []

  for ic, c in enumerate(np.where(targets > 0.5)[0]):
    wc = weights.loc[c + 1]

    wc = wc.argsort()[::-1]
    wc = wc.argsort()
    wc = np.exp(-wc * alpha)
    wc /= wc.sum()

    cam_c = np.sum([e[0][ic] * w for e, w in zip(parts, wc)], axis=0)

    cam.append(cam_c)

  cam = np.stack(cam, axis=0)
  return cam


def highest_st(parts, targets, weights, alpha=None):
  cam = []
  for ic, c in enumerate(np.where(targets > 0.5)[0]):
    wc = weights.loc[c + 1]
    cam_c = parts[wc.argmax()][0][ic]

    cam.append(cam_c)

  cam = np.stack(cam, axis=0)
  return cam


def learned_st(parts, targets, weights=None, alpha=None):
  """Learned strategy based on VOC 2012 results.
  """
  if alpha is None: alpha = 2.0
  cam = []

  # ra, oc, p, poc, pnoc = parts
  labels = targets.sum()

  if labels < 1:
    return None

  if labels == 1:
    # Singletons: p, pnoc, poc, oc, ra
    wc = np.asarray([.1, .1, .5, .15, .15])
    cam = np.sum([e[0] * w for e, w in zip(parts, wc)], axis=0)
    return cam

  for ic, c in enumerate(np.where(targets > 0.5)[0]):
    if c == 14:
      # Person: poc, oc, ra, pnoc, p
      wc =  np.asarray([.1, .2, .1, .5, .1])
    else:
      # General: pnoc, poc, oc, p, ra
      ranks = np.asarray([3, 2, 3, 1, 0])
      wc = np.exp(-ranks * alpha)
      wc /= wc.sum()
    cam_c = np.sum([e[0][ic] * w for e, w in zip(parts, wc)], axis=0)
    cam.append(cam_c)

  cam = np.stack(cam, axis=0)
  return cam


STRATEGIES = {
  "sum": sum_st,
  "avg": avg_st,
  "max": max_st,
  "weighted": weighted_st,
  "ranked": ranked_st,
  "highest": highest_st,
  "learned": learned_st,
}


def merge(parts, targets, strategy, weights=None, alpha=None):
  if strategy not in STRATEGIES:
    raise ValueError(
      f"Unknown merge strategy {strategy}. Available strategies are: {list(STRATEGIES)}"
    )

  return STRATEGIES[strategy](
    parts=parts,
    targets=targets,
    weights=weights,
    alpha=alpha,
  )
