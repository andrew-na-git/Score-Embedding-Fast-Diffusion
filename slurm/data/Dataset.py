import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class CIFARDataset(Dataset):
  def __init__(self, H, W, n=1):
    np.random.seed(3)
    
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