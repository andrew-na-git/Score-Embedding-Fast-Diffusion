import torch.multiprocessing as multiprocessing
import time
import functools
import time

import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
import scipy.stats as stats
import pandas as pd

from utils.kfp import get_B_block, diffusion_coeff, solve_pde, get_sparse_A_block, marginal_prob_std
from network.network import ScoreNet

torch.manual_seed(2);

# Parameters
eps = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"


#@title Defining pde diffusion for multi-threading
def diffuse(x, m, dm, channel, time_, g, scores, dt, H, W, N, kdes):

  y_train = []
  for j in range(H):
    y_train.append(x[channel][j, :])
  y_train = np.concatenate(y_train)
  x_train = []
  for l in range(W):
    x_train.append(x[channel][:, l])
  x_train = np.concatenate(x_train)

  xy_train = np.vstack([x_train, y_train])

  if kdes[channel] is None:
    kde_kernel = stats.gaussian_kde(xy_train)
    kdes[channel] = kde_kernel.logpdf(xy_train)
  m[0, channel] = kdes[channel]
  dx = 256/H*W
  
  sparse_A_block = get_sparse_A_block(dx, dt, g(time_[1:]), scores[1:, channel], H, W, N)

  B_block = get_B_block(dx, dt, m, channel, H, W, N)

  m[1:, channel] = solve_pde(sparse_A_block, B_block, mode='sp_sparse').reshape((-1, H*W))
  img_log_prob = m[:, channel]#.reshape((-1, H, W))

  dm[:, channel, 1:-1] = (img_log_prob[:,:-2] - img_log_prob[:,2:])/(2*dx)
  dm[:, channel, 0] = dm[:, channel, 1]
  dm[:, channel, -1] = dm[:, channel, -2]
  # dm[:, channel, 1:-1 , 1:-1] = (img_log_prob[:, 2:, 1:-1] - img_log_prob[:, :-2, 1:-1])/(2*dx) + (img_log_prob[:, 1:-1, 2:] - img_log_prob[:, 1:-1, :-2])/(2*dy)
  # dm[:, channel, 1:-1 , 0] = (img_log_prob[:, 2:, 0] - img_log_prob[:, :-2, 0])/(2*dx) + (img_log_prob[:, 1:-1, 0])/(dy)
  # dm[:, channel, 1:-1 , -1] = (img_log_prob[:, 2:, -1] - img_log_prob[:, :-2, -1])/(2*dx) + (img_log_prob[:, 1:-1, -1])/(dy)
  # dm[:, channel, 0 , 1:-1] = (img_log_prob[:, 0, 2:] - img_log_prob[:, 0, :-2])/(2*dy) + (img_log_prob[:, 0, 1:-1])/(dx)
  # dm[:, channel, -1 , 1:-1] = (img_log_prob[:, -1, 2:] - img_log_prob[:, -1, :-2])/(2*dy) + (img_log_prob[:, -1, 1:-1])/(dx)
    
# do training

def loss_fn(model, x, label, diffusion_coeff, marginal_prob_std, dt, N, idx, eps=1e-5):
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
    x[i] = x[i-1] - 0.5 * label[i-1] * diff_std2[i-1] * dt
  std = marginal_prob_std(random_t)
  z = torch.randn_like(x)
  # we perturb the image by the forward SDE conditional distribution
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x.to(device), (random_t + idx).to(device)).cpu()
  # loss = torch.mean(torch.sum((score * std[:, None, None, None] - label)**2, dim=(1, 2, 3)) / (2 * diff_std2))
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3))) # original loss from tutorial
  return loss

def diffuse_train(init_x, epoch, diffusion_coeff, marginal_prob_std, label, dt, N, loss_hist):
  
  model_score = ScoreNet(marginal_prob_std=marginal_prob_std).to(device)
  optimizer = Adam(model_score.parameters(), lr=1e-3)
  model_score.train();

  scores_label = torch.tensor(label)
  for e in tqdm(range(epoch)):
    total_loss = 0
    for i in range(init_x.shape[0]):
      loss = loss_fn(model_score, init_x[i], scores_label[i], diffusion_coeff, marginal_prob_std, dt, N, i)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    loss_hist[e] = total_loss / init_x.shape[0]

  print(f'\nloss at all channels: {loss}')
  file = f'model_cifar_thread_all.pth'
  torch.save(model_score.state_dict(), file)
  print(f"model has been saved\n")

def train(dataset, N=10, H=28, W=28, channels=3, epochs=1000, sigma=2):
  print(f"Training using device: {device}")

  dt=1/N

  n_data = len(dataset)
    
  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
  start_time = time.time()
  m = np.zeros((n_data, N, channels, H*W), dtype=np.float32)
  m_prev = np.ones((n_data, N, channels, H*W), dtype=np.float32)
  scores = np.ones((n_data, N, channels, H*W), dtype=np.float32) # initial scores guess
  dm = np.zeros_like(scores, dtype=np.float32)
  kdes = np.full((n_data, channels), None)

  # we want to sample from random time steps to construct training samples
  time_ = np.linspace(eps, 1, N).astype(np.float32)
  time_multiple = np.linspace([i + eps for i in range(n_data)], [i + 1 for i in range(n_data)], N, axis=1).astype(np.float32)

  # create training data
  res = 1
  e = 0

  while res > 1e-6:
    for idx, data in tqdm(enumerate(dataset)):
      # diffuse all three channels
      for ch in range(channels):
        diffuse(data, m[idx], dm[idx], ch, time_, diffusion_coeff_fn, scores[idx], dt, H, W, N, kdes[idx])

      scores[idx] = dm[idx].copy()

      if e == 1000:
        print(f'No convergence')
        exit(1)

      res = np.linalg.norm(m[idx] - m_prev[idx])#/np.linalg.norm(m_prev[idx])

      m_prev[idx] = m[idx].copy()
      e += 1

  scores_label = scores.copy().reshape((n_data, -1, channels, H, W))

  losses = multiprocessing.Array('f', range(epochs))
  init_x = torch.zeros((n_data, N, channels, H, W))

  for idx, data in enumerate(dataset):
    init_x[idx] = data

  diffuse_train(init_x, epochs, diffusion_coeff_fn, marginal_prob_std_fn, scores_label, dt, N, losses)
    

  end_time = time.time() - start_time
  log_df = pd.DataFrame(data={"time": [end_time] * epochs, "loss": losses[:]})
  log_df.to_csv("loss.log", index_label="epoch")
