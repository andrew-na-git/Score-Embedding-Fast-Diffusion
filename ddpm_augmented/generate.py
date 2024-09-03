from diffusers import UNet2DModel

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, new_group, gather
import os

print(torch.cuda.device_count())

beta_min = 0.0001
beta_max = 0.02

num_timesteps = 1000


def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_next(model, x, i, rank):
  beta_samples = torch.linspace(beta_min, beta_max, steps=num_timesteps, device=rank)
  alphas = torch.cumprod(1 - beta_samples, dim=0)
  beta_i = beta_samples[i]
  alpha_timesteps = alphas[i].to(rank)
  
  sqrt_one_minus_alphas = torch.sqrt(1 - alpha_timesteps) # std of noise
  score = -model(x, i).sample.detach() / sqrt_one_minus_alphas[:, None, None, None]


  return (1 / torch.sqrt(1 - beta_i))[:, None, None, None] * (x + beta_i[:, None, None, None] * score) + torch.sqrt(beta_i)[:, None, None, None] * torch.randn_like(x)

def generate_sample(model, batch_size, rank):
  sample = torch.randn(batch_size, 3, 32, 32).to(rank)
  with torch.no_grad():
    for i in tqdm(reversed(range(0, num_timesteps)), total=num_timesteps):
      sample = get_next(model, sample, torch.Tensor([i] * sample.shape[0]).long().to(rank), rank)
  
  return sample.cpu().numpy()

def main(rank, world_size):
  ddp_setup(rank, world_size)
  group = new_group(list(range(world_size)), backend='gloo')

  
  #model_path = "/home/w56gao/scratch/ddpm_checkpoints/final.pt"
  model_path = "/home/w56gao/src/Stable-Diffusion/ddpm_augmented/final.pt"
  
  model = UNet2DModel(sample_size=32, block_out_channels=[256, 256, 512, 512]).to(rank)
  model.load_state_dict(torch.load(model_path))
  model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[rank])

  model.eval()
  
  num_samples = 5000
  batch_size = 64
  
  samples = []
  
  for _ in tqdm(range((num_samples // batch_size) + 1)):
    samples.append(generate_sample(model, batch_size, rank))
  
  samples = np.concatenate(np.array(samples))
  
  print(samples.shape)
  
  samples = torch.tensor(samples)
 
  # gather
  if rank == 0:
    images = [torch.empty(len(samples), 3, 32, 32) for _ in range(world_size)]
    gather(samples, gather_list=images, dst=0, group=group)
  else:
    gather(samples, gather_list=[], dst=0, group=group)
  
  if rank == 0:
    np.save('images.npy', np.concatenate(np.array([i.cpu() for i in images])))
    print('saved')
  

if __name__ == "__main__":
  world_size = torch.cuda.device_count()
  mp.spawn(main, args=(world_size,), nprocs=world_size)