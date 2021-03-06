
import numpy as np
import sklearn.datasets

import torch
import torch.utils.data
import torch.nn as nn

from wgan_trainer import WGANTrainer
from utils import get_config


# Networks
class Generator(nn.Module):
  def __init__(self, args):
    super(Generator, self).__init__()

    self.batch_size = args.batch_size
    self.device = args.device
    self.net = nn.Sequential(
      nn.Linear(2, args.dim),
      nn.ReLU(inplace=True),
      nn.Linear(args.dim, args.dim),
      nn.ReLU(inplace=True),
      nn.Linear(args.dim, args.dim),
      nn.ReLU(inplace=True),
      nn.Linear(args.dim, 2),
    )
  
  def forward(self):
    x = torch.randn([self.batch_size, 2]).to(self.device)
    return self.net(x)


class Discriminator(nn.Module):
  def __init__(self, args):
    super(Discriminator, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(2, args.dim),
      nn.ReLU(inplace=True),
      nn.Linear(args.dim, args.dim),
      nn.ReLU(inplace=True),
      nn.Linear(args.dim, args.dim),
      nn.ReLU(inplace=True),
      nn.Linear(args.dim, 1),
    )

  def forward(self, x):
    return self.net(x)


class ToyNet(nn.Module):
  def __init__(self, args):
    super(ToyNet, self).__init__()
    self.gen = Generator(args)
    self.dis = Discriminator(args)


# Datasets:
class Gaussians25(torch.utils.data.Dataset):
  def __init__(self, num_points=100000, is_inf=True):
    self.num_points = num_points
    self.is_inf = is_inf
    self.dataset = []
    for i in range(self.num_points//25):
      for x in range(-2, 3):
        for y in range(-2, 3):
          point = np.random.randn(2)*0.05
          point[0] += 2*x
          point[1] += 2*y
          self.dataset.append(point)
    self.dataset = np.array(self.dataset, dtype='float32')
    self.dataset /= 2.828 # stdev

  def __len__(self):
    if self.is_inf:
      return 100000000
    else:
      return self.num_points

  def __getitem__(self, index):
    return self.dataset[index % self.num_points]


class Swissroll(torch.utils.data.Dataset):
  def __init__(self, num_points=100000, is_inf=True):
    self.num_points = num_points
    self.is_inf = is_inf
    self.dataset = sklearn.datasets.make_swiss_roll(
      n_samples=num_points,
      noise=0.25
    )[0]
    self.dataset = self.dataset.astype('float32')[:, [0, 2]]
    self.dataset /= 7.5  # stdev plus a little
    self.dataset = np.array(self.dataset, dtype='float32')
    self.dataset /= 2.828 # stdev

  def __len__(self):
    if self.is_inf:
      return 100000000
    else:
      return self.num_points

  def __getitem__(self, index):
    return self.dataset[index % self.num_points]


class Gaussians8(torch.utils.data.Dataset):
  def __init__(self, num_points=100000, is_inf=True):
    self.num_points = num_points
    self.is_inf = is_inf
    self.dataset = []
    scale = 2.
    centers = [
      (1, 0),
      (-1, 0),
      (0, 1),
      (0, -1),
      (1. / np.sqrt(2), 1. / np.sqrt(2)),
      (1. / np.sqrt(2), -1. / np.sqrt(2)),
      (-1. / np.sqrt(2), 1. / np.sqrt(2)),
      (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    self.dataset = []
    for i in range(self.num_points // 8):
      for j in range(8):
        point = np.random.randn(2) * .02
        center = centers[j]
        point[0] += center[0]
        point[1] += center[1]
        self.dataset.append(point)
    self.dataset = np.array(self.dataset, dtype='float32')
    self.dataset /= 1.414  # stdev

  def __len__(self):
    if self.is_inf:
      return 100000000
    else:
      return self.num_points

  def __getitem__(self, index):
    return self.dataset[index % self.num_points]


def main():
  args = get_config('config/toy.yaml')

  if torch.cuda.is_available():
    args.device = torch.device(args.gpu)
  else:
    print("Using CPU")
    args.device = torch.device("cpu")

  net = ToyNet(args).to(args.device)
  trainer = WGANTrainer(net, args)

  if args.dataset == "gaussians25":
    dataset = Gaussians25()
  elif args.dataset == "gaussians8":
    dataset = Gaussians8()
  elif args.dataset == "swissroll":
    dataset = Swissroll()

  train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size, shuffle=True, 
    pin_memory=True, drop_last=True)

  trainer.train(train_loader, args)


if __name__ == "__main__":
  main()