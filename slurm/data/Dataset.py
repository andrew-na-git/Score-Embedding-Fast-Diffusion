import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, Flowers102
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class CIFARDataset(Dataset):
  def __init__(self, H, W, n=4):
    np.random.seed(2)
    
    self.transform = v2.Compose([
      v2.ToImageTensor(),
      v2.CenterCrop((W, H))
    ])
    
    cifar = CIFAR10(dir_path, download=True)
    cifar_data = cifar.data[np.array(cifar.targets) == 5]
    
    choices = np.random.choice(np.arange(len(cifar_data)), n, replace=False)
    self.data = cifar_data[choices]
    self.channels = 3
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    raw = self.transform(self.data[idx]).unsqueeze(0)[0]
    return (raw - raw.min())/(raw.max() - raw.min())

class FlowersDataset(Dataset):
  def __init__(self, H, W, n=1):
    np.random.seed(2)
    
    self.transform = v2.Compose([
      v2.ToImageTensor(),
      v2.Resize((W, H), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    ])
    
    flowers = Flowers102(dir_path, transform = self.transform, download=True)
    
    choices = np.random.choice(np.arange(len(flowers)), n, replace=False)
    self.data = Subset(flowers, choices)
    self.channels = 3
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    raw = self.data[idx][0].data
    return (raw - raw.min())/(raw.max() - raw.min())