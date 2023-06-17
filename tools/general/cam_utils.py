import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from torchvision import utils

from tools.ai.demo_utils import *
from tools.ai.torch_utils import *


def check_positive(am):
  am[am > 0.5] = 1
  am[am <= 0.5] = 0
  edge_mean = (am[0, 0, 0, :].mean() + am[0, 0, :, 0].mean() + am[0, 0, -1, :].mean() + am[0, 0, :, -1].mean()) / 4
  return edge_mean > 0.5


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def visualize_heatmap(experiments, images, attmaps, epoch, cnt, phase='train', seg=False):
  import cmapy
  n = int(np.sqrt(images.shape[0]))
  utils.save_image(
    images, './experiments/images/{}/{}/{}-{}-pri.jpg'.format(experiments, phase, epoch, cnt), nrow=n, normalize=True
  )
  _, c, h, w = images.shape
  fig, axes = plt.subplots(n, n, figsize=(21, 21))
  for j in range(axes.shape[0]):
    for k in range(axes.shape[1]):
      temp = attmaps[j * n + k, 0, :, :]
      temp = temp.cpu().detach().numpy()
      if temp.shape[0] != h:
        temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_CUBIC)
      axes[j, k].imshow(temp)
      axes[j, k].axis('off')
  plt.savefig('./experiments/images/{}/{}/{}-{}-att.jpg'.format(experiments, phase, epoch, cnt))
  plt.close()

  fig, axes = plt.subplots(n, n, figsize=(21, 21))
  for j in range(axes.shape[0]):
    for k in range(axes.shape[1]):
      temp = attmaps[j * n + k, 0, :, :]
      temp = temp.cpu().detach().numpy()
      axes[j, k].hist(temp.ravel(), bins=50, range=(0, 1), color='cornflowerblue')
      axes[j, k].set_xlabel('Intensity')
      axes[j, k].set_ylabel('Density')
      # axes[j, k].axis('off')
  plt.savefig('./experiments/images/{}/{}/{}-{}-hist.jpg'.format(experiments, phase, epoch, cnt))
  plt.close()

  attmaps = attmaps.squeeze().to('cpu').detach().numpy()

  for i in range(images.shape[0]):
    attmap = attmaps[i]
    attmap = attmap / np.max(attmap)
    attmap = np.uint8(attmap * 255)
    colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cmapy.cmap('seismic'))
    # save .npy data
    # np.save('images/{}/{}/colormaps/{}-{}-image.npy'.format(experiments, phase, cnt, i), attmap)

    grid = utils.make_grid(images[i].unsqueeze(0), nrow=1, padding=0, pad_value=0, normalize=True, value_range=None)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    image = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]
    # print(image.shape, colormap.shape)
    cam = colormap + 0.4 * image
    cam = cam / np.max(cam)
    cam = np.uint8(cam * 255).copy()

    cv2.imwrite('./experiments/images/{}/{}/colormaps/{}-{}-image.jpg'.format(experiments, phase, cnt, i), image)
    cv2.imwrite('./experiments/images/{}/{}/colormaps/{}-{}-colormap.jpg'.format(experiments, phase, cnt, i), cam)

    if seg:
      h, w, c = image.shape
      attmap = cv2.resize(attmap, (h, w), interpolation=cv2.INTER_LINEAR)
      mask = np.where(attmap > 150, 1, 0).reshape(h, w, 1)
      temp = np.uint8(image * mask)
      cv2.imwrite('./experiments/images/{}/{}/colormaps/{}-{}-seg.jpg'.format(experiments, phase, cnt, i), temp)
