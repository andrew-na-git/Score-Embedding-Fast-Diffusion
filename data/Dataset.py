import numpy as np
import os
import pickle
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, Flowers102
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import math

from abc import ABC, abstractmethod

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class Dataset(ABC):
  def __init__(self, image_res=32, n_data=1, repeat_dataset=1, sample=True):
    self.repeat = repeat_dataset
    self.image_res = image_res
    self.n_data = n_data
    
    transform = v2.Compose([
      v2.ToImageTensor(),
      v2.Resize((image_res, image_res), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    ])
    
    raw_data = self.get_raw_data()
    
    # get n_data images and transform them
    if sample:
      choices = np.random.choice(np.arange(len(raw_data)), n_data, replace=False)
      self.data = [transform(x) for x in raw_data[choices]]
    else:
      self.data = [transform(x) for x in raw_data[:n_data]]

    # normalize to [0, 1]
    self.data = [(raw - raw.min())/(raw.max() - raw.min()) for raw in self.data]

  @abstractmethod
  def get_raw_data(self):
    pass
  
  def plot(self, n_cols = 3):
    n_cols = min(n_cols, self.n_data)
    n_rows = math.ceil(self.n_data / n_cols)
    fig, axes = plt.subplots(n_rows, 
                             n_cols, 
                             figsize=(3 * n_cols, 3 * n_rows + 0.2), 
                             sharex=True, 
                             sharey=True,
                             tight_layout=True);
    if self.n_data == 1:
      axes.imshow(self.data[0].moveaxis(0, 2), aspect="auto");
    else:
      for i, ax in enumerate(axes.flatten()):
        if i >= len(self.data):
          fig.delaxes(ax);
          continue
        ax.imshow(self.data[i].moveaxis(0, 2), aspect="auto");
      
    fig.suptitle(f"Ground Truth (Image Size = {self.image_res})");

    return fig;
  
  def __len__(self):
    return self.n_data * self.repeat
  
  def __getitem__(self, idx):
    if idx >= self.n_data * self.repeat:
      raise StopIteration()
    return self.data[idx % self.n_data]
    
class CIFARDataset(Dataset):
  def __init__(self, image_res=32, n_data=1, seed=9, repeat_dataset=1):
    np.random.seed(seed)
    super().__init__(image_res, n_data, repeat_dataset)
    self.channels = 3
    
  def get_raw_data(self):
    cifar = CIFAR10(dir_path, download=True)
    # choose dogs
    return cifar.data[np.array(cifar.targets) == 5]

class ImageNetDataset(Dataset):
  def __init__(self, image_res=64, n_data=1, seed=9, repeat_dataset=1):
    super().__init__(image_res, n_data, repeat_dataset, sample=False)
    self.channels = 3
    
  def get_raw_data(self):
    with open(os.path.join(dir_path, 'imagenet_data.pkl'), 'br') as f:
      data = pickle.load(f)
    
    if self.n_data == 1:
      raw_data = [data[1]]
    else:
      raw_data = data[:self.n_data]
    
    return raw_data

class CelebDataset(Dataset):
  def __init__(self, image_res=64, n_data=1, seed=9, repeat_dataset=1):
    super().__init__(image_res, n_data, repeat_dataset, sample=False)
    self.channels = 3
    
  def get_raw_data(self):
    with open(os.path.join(dir_path, 'celeb_data.pkl'), 'br') as f:
      data = pickle.load(f)
    
    if self.n_data == 1:
      raw_data = [data[-3]]
    else:
      raw_data = data[:self.n_data]
      
    return raw_data
  
def get_dataset(config, repeat = 1):
  
  dataset = config["data_loader"]["dataset"]
  n_data = config["data_loader"]["num_images"]
  img_size = config["data_loader"]["image_size"]
  seed = config["data_loader"]["seed"]
  
  if dataset == "cifar":
    dataset_class = CIFARDataset
  elif dataset == "celeb":
    dataset_class = CelebDataset
  elif dataset == "imagenet":
    dataset_class = ImageNetDataset
  else:
    raise NotImplementedError(f"Dataset {dataset} is not a valid dataset")

  return dataset_class(img_size, n_data, seed, repeat)
  
