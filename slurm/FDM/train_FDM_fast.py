import os
import threading
import torch.multiprocessing as multiprocessing
import time
import functools
import time

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
import scipy.stats as stats
import pandas as pd

from utils.kfp import get_B_block, diffusion_coeff, solve_pde, get_sparse_A_block, marginal_prob_std
from network.network import ScoreNet

from data.Dataset import CIFARDataset

torch.manual_seed(2);
torch.set_num_threads(1)

# Parameters
eps = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"


#@title Defining pde diffusion for multi-threading
def diffuse(x, m, dm, channel, time_, g, scores, dt, H, W, N):

  y_train = []
  for j in range(H):
    y_train.append(x[channel][j, :])
  y_train = np.concatenate(y_train)
  x_train = []
  for l in range(W):
    x_train.append(x[channel][:, l])
  x_train = np.concatenate(x_train)

  xy_train = np.vstack([x_train, y_train])
  kde_kernel = stats.gaussian_kde(xy_train)
  xy_sample = kde_kernel.resample(seed=0)
  m[0, channel] = kde_kernel.logpdf(xy_train)
  dx = xy_train.max()/H*W
  
  sparse_A_block = get_sparse_A_block(dx, dt, g(time_[1:]), scores[1:, channel], H, W, N)

  B_block = get_B_block(dx, dt, m, channel, H, W, N)

  m[1:, channel] = solve_pde(sparse_A_block, B_block, mode='sp_sparse').reshape((-1, H*W))
  img_log_prob = m[:, channel]
  for i in range(N):
    dm[i, channel, 1:-1] = (img_log_prob[i,:-2] - img_log_prob[i,2:])/(2*dx)
    dm[i, channel, 0] = dm[i, channel, 1]
    dm[i, channel, -1] = dm[i, channel, -2]
    
# do training

def loss_fn(model, x, label, diffusion_coeff, marginal_prob_std, dt, N, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.tensor(np.sort(np.random.uniform(eps, 1., N)).astype(np.float32))
  # we encode the label into the initial data using the reverse ODE
  diff_std2 = diffusion_coeff(2 * random_t)
  for i in range(1, N):
    x[i] = x[i-1] - 0.5 * label[i] * diff_std2[i] * dt
  std = marginal_prob_std(random_t)
  z = torch.randn_like(x)
  # we perturb the image by the forward SDE conditional distribution
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  # loss = torch.mean(torch.sum((score * std[:, None, None, None] - label)**2, dim=(1, 2, 3)) / (2 * diff_std2))
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3))) # original loss from tutorial
  return loss

def diffuse_train(channel, init_x, epoch, diffusion_coeff, marginal_prob_std, label, dt, N, loss_hist):
  
  model_score = ScoreNet(marginal_prob_std=marginal_prob_std)
  optimizer = Adam(model_score.parameters(), lr=1e-4)
  model_score.train();

  scores_label = torch.tensor(label)[:, channel][:, None]
  for e in tqdm(range(epoch)):
    loss = loss_fn(model_score, init_x[:, channel][:, None], scores_label, diffusion_coeff, marginal_prob_std, dt, N)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_hist[e] = loss.item()

  print(f'\nloss at channel {channel}: {loss}')
  file = f'model_cifar_thread_{channel}.pth'
  torch.save(model_score.state_dict(), file)
  print(f"model for thread {channel} has been saved\n")

def train(dataset, N=10, H=28, W=28, channels=3, epochs=1000, sigma=2):
  print(f"Training using device: {device}")

  dt=1/N
    
  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
  start_time = time.time()
  m = np.zeros((N, channels, H*W), dtype=np.float32)
  m_prev = np.ones((N, channels, H*W), dtype=np.float32)
  scores = np.ones((N, channels, H*W), dtype=np.float32) # initial scores guess
  dm = np.zeros_like(scores, dtype=np.float32)

  # we want to sample from random time steps to construct training samples
  time_ = np.linspace(eps, 1, N).astype(np.float32)

  # create training data
  res = 1
  e = 0

  while res > 0.005:
    for idx, data in tqdm(enumerate(dataset)):
      

      # diffuse all three channels concurrently
      for ch in range(channels):
          diffuse(data, m, dm, ch, time_, diffusion_coeff_fn, scores, dt, H, W, N)

      scores = dm.copy()

      if e == 1000:
        print(f'No convergence')
        exit(1)

      res = np.linalg.norm(m - m_prev)/np.linalg.norm(m_prev)

      m_prev = m.copy()
      e += 1

  scores_label = scores.copy().reshape((-1, channels, H, W))

  processes = [None] * channels
  losses = [multiprocessing.Array('f', range(epochs))] * channels
  init_x = torch.zeros((N, channels, H, W))

  for idx, data in enumerate(dataset):
    for ch in range(channels):
      init_x[:, ch] = data[ch]

    # train all three channels concurrently
    for ch in range(channels):
      processes[ch] = multiprocessing.Process(target=diffuse_train, args=[ch, init_x, epochs, diffusion_coeff_fn, marginal_prob_std_fn, scores_label, dt, N, losses[ch]])
      processes[ch].start()

    for p in processes:
      p.join()

  end_time = time.time() - start_time
  log_df = pd.DataFrame(data={"time": [end_time] * epochs, "first_loss": losses[0][:], "second_loss": losses[1][:], "third_loss": losses[2][:]})
  log_df.to_csv("loss.log", index_label="epoch")
