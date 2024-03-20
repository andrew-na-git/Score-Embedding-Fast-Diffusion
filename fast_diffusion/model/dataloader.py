import torch
import numpy as np
from .kfp import diffusion_coeff, marginal_prob_std
import math

def create_batch(x, scores, config, idx):
  # create a batch by embedding score into image

  eps = float(config["misc"]["eps"])
  N = config["diffusion"]["num_timesteps"]
  sigma = config["diffusion"]["sigma"]
  time_embed_method = config["model"]["embedding_method"]
  n_data = config["data_loader"]["num_images"]
  batch_size = config["data_loader"]["batch_size"]
  timestep_multiplier = config["training"]["timestep_multiplier"]

  dt = 1/N

  perturbed_x = []
  t = []
  stds = []
  diff_stds = []
  zs = []
  
  for i in range(math.ceil(batch_size / N)):
    random_t = torch.tensor((np.linspace(eps, 1.-dt/2, N, endpoint=False) + (np.random.rand(N)-eps)*dt/2)).float()
    # we encode the label into the initial data using the reverse ODE
    diff_std2 = diffusion_coeff(2 * random_t, sigma)

    for i in range(1, N):
      x[i] = x[i-1] - 0.5 * scores[idx][i] * diff_std2[i] * dt
    std = marginal_prob_std(random_t, sigma)

    z = torch.randn_like(x)

    # we perturb the image by the forward SDE conditional distribution
    perturbed_x.append(x + z * std[:, None, None, None])

    if time_embed_method == "linear":
      t.append(timestep_multiplier * (random_t + idx - n_data/2))
    else:
      t.append(random_t * (timestep_multiplier - 1))

    stds.append(std)
    diff_stds.append(diff_std2)
    zs.append(z)
  
  perturbed_x = torch.concatenate(perturbed_x)
  t = torch.concatenate(t)
  diff_stds = torch.concatenate(diff_stds)
  stds = torch.concatenate(stds)
  zs = torch.concatenate(zs)

  indices = np.sort(np.random.choice(len(perturbed_x), batch_size, replace=False))
  return perturbed_x[indices], t[indices], diff_stds[indices], stds[indices], zs[indices]
