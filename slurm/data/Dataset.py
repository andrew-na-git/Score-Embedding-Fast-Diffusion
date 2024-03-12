import numpy as np
import os
import pickle
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, Flowers102
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class CIFARDataset(Dataset):
  def __init__(self, H, W, n=3, seed=10, repeat=1):
    np.random.seed(seed)
    self.repeat = repeat
    
    self.transform = v2.Compose([
      v2.ToImageTensor(),
      v2.CenterCrop((W, H))
    ])
    
    cifar = CIFAR10(dir_path, download=False)
    cifar_data = cifar.data[np.array(cifar.targets) == 5]
    
    choices = np.random.choice(np.arange(len(cifar_data)), n, replace=False)

    self.data = [self.transform(x) for x in cifar_data[choices]]

    self.data = [(raw - raw.min())/(raw.max() - raw.min()) for raw in self.data]

    self.channels = 3
  def __len__(self):
    return len(self.data) * self.repeat
  
  def __getitem__(self, idx):
    if idx >= len(self.data) * self.repeat:
      raise StopIteration()
    return self.data[idx % len(self.data)]

class FlowersDataset(Dataset):
  def __init__(self, H, W, n=3):
    np.random.seed(37)
    
    self.transform = v2.Compose([
      v2.ToImageTensor(),
      v2.Resize((W, H), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    ])
    
    flowers = Flowers102(dir_path, transform = self.transform, download=True)
    
    choices = np.random.choice(np.arange(len(flowers)), n, replace=False)
    choices[-1] += 1
    self.data = Subset(flowers, choices)
    self.channels = 3
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    raw = self.data[idx][0].data
    return (raw - raw.min())/(raw.max() - raw.min())

class ImageNetDataset(Dataset):
  def __init__(self, H, W, n=3, seed=None, repeat = 1):
    self.repeat = repeat
    with open(os.path.join(dir_path, 'imagenet_data.pkl'), 'br') as f:
      data = pickle.load(f)
    self.transform = v2.Compose([
      v2.ToImageTensor(),
      v2.Resize((W, H), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    ])
    if n == 1:
      self.data = [self.transform(data[1])]
    else:
      self.data = [self.transform(img) for img in data[:n]]
    self.channels = 3

    self.data = [(raw - raw.min())/(raw.max() - raw.min()) for raw in self.data]
  
  def __len__(self):
    return len(self.data) * self.repeat
  
  def __getitem__(self, idx):
    return self.data[idx % len(self.data)]
  
class CelebDataset(Dataset):
  def __init__(self, H, W, n=3, seed=None, repeat=1):
    self.repeat = repeat
    with open(os.path.join(dir_path, 'celeb_data.pkl'), 'br') as f:
      data = pickle.load(f)
    self.transform = v2.Compose([
      v2.ToImageTensor(),
      v2.Resize((W, H), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    ])
    if n == 1:
      self.data = [self.transform(data[-3])]
    else:
      self.data = [self.transform(img) for img in data[:n]]
    self.channels = 3

    self.data = [(raw - raw.min())/(raw.max() - raw.min()) for raw in self.data]
  
  def __len__(self):
    return len(self.data) * self.repeat
  
  def __getitem__(self, idx):
    if idx >= len(self.data) * self.repeat:
      raise StopIteration()
    return self.data[idx % len(self.data)]
    