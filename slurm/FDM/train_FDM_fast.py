import os
import threading

import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time

import torch
from torch.optim import Adam

from utils.kfp import get_B_block, diffusion_coeff, solve_pde, logsumexp, get_sparse_A_block
from network.network import ScoreNet

mm_scaler = MinMaxScaler();

# Parameters

device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-6

params = {"bandwidth": np.logspace(-1, 1, 20)}

# diffuse function for 1 channel
def get_init_score(data, data_idx, channel, kdes):
  if not kdes[data_idx, channel]:
      grid = GridSearchCV(KernelDensity(), params)
      kdes[data_idx, channel] = grid.fit(data.ravel()[:, None]) # 0.7 sec
  return kdes[data_idx, channel].score_samples(data.ravel()[:, None])
  
def construct_samples(random_t, del_m, m, sigma_, train_x_data, train_y_data, channel, N, H, W):
  # constructing the training data and labels
  for i, _ in enumerate(random_t, 1):
    del_m[channel][i] = np.diff(m[channel][i].ravel(), axis=0, prepend=m[channel][i, 0])

  x = torch.tensor(mm_scaler.fit_transform(np.exp((-m[channel].ravel() - logsumexp(-m[channel].ravel())))[:, None])).reshape((N, 1, H, W))
  perturbed_x = x + torch.randn_like(x) * torch.sqrt(2 * torch.tensor(sigma_)**2)[:, None, None, None]
  train_x_data[:, channel] = perturbed_x[:, 0]
  train_y_data[:, channel] = torch.tensor(del_m[channel].astype(np.float32)).reshape((N, H, W))

def diffuse(data, x, m , del_m, channel, train_x_data, train_y_data, random_t, time_, sigma_, data_idx, scores, kdes, N, H, W):
    data = data[channel]
    dx = data.detach().numpy().max()/H
    dy = data.detach().numpy().max()/W
    
    x[channel][0] = torch.tensor(mm_scaler.fit_transform(data.ravel()[:, None]).astype(np.float32).reshape((1, 1, H, W)))
    m[channel][0] = get_init_score(data, data_idx, channel, kdes)
    del_m[channel][0] = np.diff(m[channel][0].ravel(), axis=0, prepend=m[channel][0,0])
    
    A_block = get_sparse_A_block(dx, dy, random_t - time_[:N-1], np.sqrt(2) * sigma_[1:], scores[1:, channel], H, W, N)
    
    B_block = get_B_block(dx, dy, time_, m, sigma_, channel, scores, H, W, N)
    
    m[channel][1:] = solve_pde(A_block, B_block, mode="sp_sparse").reshape((N-1, H*W))

    construct_samples(random_t, del_m, m, sigma_, train_x_data, train_y_data, channel, N, H, W)
    
# do training

def train(dataset, N=10, H=28, W=28, channels=3, epochs=60, sigma=25):
  print(f"Training using device: {device}")
  threads = [None] * channels
  loss_hist = []
  epoch_times = []

  dt = 1/N
  n_data = len(dataset)

  scores = np.zeros((N, channels, H, W), dtype=np.float32)
  kdes = np.full((n_data, channels), None)

  model_score = ScoreNet(H=H, W=W, in_channels=channels).to(device)
  loss_fn = torch.nn.MSELoss()
  optimizer = Adam(model_score.parameters(), lr=1e-3)

  model_score.train();

  for e in tqdm(range(epochs)):
    x = torch.zeros((channels, N, H, W))
    m = np.zeros((channels, N, H*W), dtype=np.float32)
    del_m = np.zeros_like(m, dtype=np.float32)
    # we want to sample from random time steps to construct training samples
    random_t = np.linspace([dt + eps] * n_data, [1] * n_data, N-1, axis=1)
    random_t += np.random.uniform(-dt/2, dt/2, random_t.shape)
    random_t[:, -1] = 1
    time_ = np.sort(np.insert(random_t, 0, eps, axis=1), axis=1).astype(np.float32)
    sigma_ = diffusion_coeff(torch.tensor(time_), sigma).detach().cpu().numpy()
    for idx, x_ in enumerate(dataset):
      epoch_start_time = time.time()

      data = x_
      train_x_data = torch.zeros((N, channels, H, W))
      train_y_data = torch.zeros_like(train_x_data)
      
      # diffuse all three channels concurrently
      for i in range(channels):
        threads[i] = threading.Thread(target=diffuse, args=[data, x, m, del_m, i, train_x_data, train_y_data, random_t[idx], time_[idx], sigma_[idx], idx, scores, kdes, N, H, W])
        threads[i].start()
      for thread in threads:
        thread.join()

      y_pred = model_score(train_x_data.to(device), torch.tensor(time_[idx]).to(device))
      lm = (2*torch.tensor(sigma_[idx])**2)[:, None, None, None].to(device)
      loss = loss_fn(y_pred/lm, train_y_data.to(device))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses = loss.item()
      loss_hist.append(losses)
      scores = (y_pred/lm).clone().detach().cpu().numpy().reshape((N, channels, H, W)) # we normalize before fedding back into PDE

      epoch_times.append(time.time() - epoch_start_time)

  log_df = pd.DataFrame(data={"time": epoch_times, "loss": loss_hist})
  log_df.to_csv("loss.log", index_label="epoch")

  torch.save(model_score.state_dict(), 'model_FDM.pth')
