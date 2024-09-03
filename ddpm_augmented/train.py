# %%
from diffusers import UNet2DModel

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

import torchvision
from torchvision import transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

local_rank = int(os.environ["LOCAL_RANK"])

global_rank = int(os.environ["RANK"])

print(f'Worldsize = {os.environ["WORLD_SIZE"]}')
print(f"Global rank = {global_rank}")

checkpoint_dir = "/home/w56gao/scratch/test"

def ddp_setup():
  init_process_group(backend="nccl")

beta_min = 0.0001
beta_max = 0.02

num_timesteps = 1000

# %%
def forward_pass(images, timesteps):
  beta_samples = torch.linspace(beta_min, beta_max, steps=num_timesteps, device=local_rank)

  alphas = torch.cumprod(1 - beta_samples, dim=0)
  images = images.to(local_rank)
  timesteps = timesteps.to(local_rank)
  alpha_timesteps = alphas[timesteps].to(local_rank)
  sqrt_alphas = torch.sqrt(alpha_timesteps) # to scale the image

  sqrt_one_minus_alphas = torch.sqrt(1 - alpha_timesteps) # std of noise

  # sample our noise
  noise = torch.randn_like(images) 

  return noise * sqrt_one_minus_alphas[:, None, None, None] + images * sqrt_alphas[:, None, None, None], noise



# %% [markdown]
# ## Main

# %%
def train(model,data_loader,criterion,optimizer, epochs):

  for epoch in tqdm(range(epochs)):
    data_loader.sampler.set_epoch(epoch)
    for batch_idx, inputs in enumerate(data_loader):
      model.train()
      inputs = inputs.to(local_rank)
      
      # sample some random timesteps
      timesteps = torch.randint(low=0, high=num_timesteps, size=[len(inputs)], device=local_rank)
      peturbed, noise = forward_pass(inputs, timesteps)

      y = model(peturbed, timesteps.float()).sample

      optimizer.zero_grad()
      loss = criterion(y, noise)
      loss.backward()
      optimizer.step()
  
      tqdm.write(f"Batch {batch_idx + 1}, Loss {loss.item()}", end= "\r")
    
    if epoch % 5 == 0 and local_rank == 0:
      torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, f"{epoch}.pt"))

class CustomDataset(Dataset):
  def __init__(self, data, transform=None):
    self.data = data
    self.transform = transform
      
  def __getitem__(self, index):
    x = self.data[index]
    
    if self.transform:
      x = self.transform(x)
    
    return x
  
  def __len__(self):
    return len(self.data)

# %%
def main():
  ddp_setup()
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
  ])

  #Load the data and choose one class, then create the data loader.
  cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)

  dogs = cifar_dataset.data

  dataset = CustomDataset(dogs, preprocess)
  
  epochs = 100
  
  model = UNet2DModel(sample_size=32, block_out_channels=[256, 256, 512, 512]).to(local_rank)
  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Number of parameters: {total_params}")
  
  model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank])
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  
  dataloader = DataLoader(dataset, 64, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))
  
  train(model, dataloader, nn.MSELoss(), optimizer, epochs)
  
  if local_rank == 0:
    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'final.pt'))
  destroy_process_group()

# %%
if __name__ == "__main__":
  main()


