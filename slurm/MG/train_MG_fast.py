import os
import threading
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import torch
from torch.optim import Adam

from utils.kfp import get_B_block, construct_B, diffusion_coeff, construct_R, construct_P, gauss_seidel, solve_pde, logsumexp, get_sparse_A_block
from network.network import ScoreNet


mm_scaler = MinMaxScaler();

# parameters
device = "gpu" if torch.cuda.is_available() else "cpu"
eps = 1e-6
params = {"bandwidth": np.logspace(-1, 1, 20)}



# diffuse one channel
def get_init_score(data, data_idx, channel, kdes):
  if kdes[data_idx, channel] is None:
      grid = GridSearchCV(KernelDensity(), params)
      kdes[data_idx, channel] = grid.fit(data.ravel()[:, None]).score_samples(data.ravel()[:, None])
  return kdes[data_idx, channel]

def get_A_block(dx, dy, random_t, time_, sigma_, scores, channel):
  # we normalize for sigma to ensure the dynamics doesn't blow up
  As = []
  for i, t_ in enumerate(random_t, 1):
    A = construct_A(dx, dy, t_ - time_[i-1], np.sqrt(2) * sigma_[i], scores[i][channel].ravel(), H, W)
    As.append(A)

  A_block = sp.linalg.block_diag(*As)
  for i, t_ in enumerate(random_t, 1):
    if i == 1:
      continue
    A_block[(i-1)*H*W:i*H*W, (i-2)*H*W:(i-1)*H*W] = -np.eye((H*W))/(t_ - time_[i-1])
  
  return A_block
  
def construct_samples(random_t, del_m, m, m_c, sigma_, train_x_data, train_y_data, train_xc_data, channel, N, H, W):
  # constructing the training data and labels
  for i, t_ in enumerate(random_t, 1):
    del_m[channel][i] = np.diff(m[channel][i].ravel(), axis=0, prepend=m[channel][i, 0])

  x = torch.tensor(mm_scaler.fit_transform(np.exp((-m[channel].ravel() - logsumexp(-m[channel].ravel())))[:, None])).reshape((N, 1, H, W))
  perturbed_x = x + torch.randn_like(x) * torch.sqrt(2 * torch.tensor(sigma_)**2)[:, None, None, None]
  train_x_data[:, channel] = perturbed_x[:, 0]
  train_y_data[:, channel] = torch.tensor(del_m[channel].astype(np.float32)).reshape((N, H, W))

  # generate coarse dataset
  x_c = torch.tensor(mm_scaler.fit_transform(np.exp((-m_c[channel].ravel() - logsumexp(-m_c[channel].ravel())))[:, None])).reshape((N, int(H/2), int(W/2)))
  perturbed_xc = x_c + torch.randn_like(x_c) * torch.sqrt(2 * torch.tensor(sigma_)**2)[:, None, None]
  train_xc_data[:, channel] = perturbed_xc

def diffuse(data, x, m , del_m, m_c, channel, train_xc_data, train_x_data, train_y_data, random_t, time_, sigma_, data_idx, scores, kdes, N, H, W, R_block, P_block, sparse_R_block, sparse_P_block):
    data = data[channel]
    dx = data.detach().numpy().max()/H
    dy = data.detach().numpy().max()/W
    
    x[channel][0] = torch.tensor(mm_scaler.fit_transform(data.ravel()[:, None]).astype(np.float32).reshape((1, 1, H, W)))
    m[channel][0] = get_init_score(data, data_idx, channel, kdes)
    del_m[channel][0] = np.diff(m[channel][0].ravel(), axis=0, prepend=m[channel][0,0])
    
    A_block = get_sparse_A_block(dx, dy, random_t - time_[:N-1], np.sqrt(2) * sigma_[1:], scores[1:, channel], H, W, N)
    
    B_block = get_B_block(dx, dy, time_, m, sigma_, channel, scores, H, W, N)
    
    # update m (pre-smoothing)
    m[channel][1:] = gauss_seidel(A_block, B_block, scores[1:, channel].flatten(), A_block.shape[0]).reshape(((N-1), H*W))
    
    # we want to perform the coarse grid
    # compute residual r = b - Am[1:]
    r = B_block - A_block@(m[channel][1:]).flatten()
    # coursening step 1: r_c = R_c@r
    r_c = R_block@r
    # coursening A_c = R_c@A@P_c (Petrov-Galerkin Coursening)
    
    A_c = (sparse_R_block@A_block@sparse_P_block)
    
    # compute course err: err_c = solve_pde(A_c,r_c)
    err_c = solve_pde(A_c, r_c, mode='sp_sparse')
    
    # interpolate to fine grid: err = P_c@err_c
    err = P_block@err_c
    # we apply fine grid-correction
    m[channel][1:] = (m[channel][1:].flatten() + err).reshape((N-1, H*W))
    # post smoothing
    m[channel][1:] = gauss_seidel(A_block, B_block, m[channel][1:].flatten(), A_block.shape[0]).reshape(((N-1), H*W))
    # we want to coarsen the score function to train on coarse data
    m_c[channel][1:] = (R_block@m[channel][1:].flatten()).reshape((-1, int(H*W/4)))
    
    construct_samples(random_t, del_m, m, m_c, sigma_, train_x_data, train_y_data, train_xc_data, channel, N, H, W)
    
# do training

def train(dataset, N=10, H=28, W=28, channels=3, epochs=60, sigma=25):
  print(f"Training using device: {device}")
  threads = [None] * channels
  loss_c_hist = []
  loss_hist = []
  epoch_times = []
  
  n_data = len(dataset)
  dt = 1/N


  # create memory buffers
  scores = np.zeros((N, channels, H, W), dtype=np.float32)
  kdes = np.full((n_data, channels), None)

  params = {"bandwidth": np.logspace(-1, 1, 20)}
  P = construct_P(int(W), int(H/2))
  P_block = sp.linalg.block_diag(*([P] * (N-1)))
  sparse_P_block = sp.sparse.block_diag(([P] * (N-1)))

  R = construct_R(P)
  R_block = sp.linalg.block_diag(*([R] * (N-1)))
  sparse_R_block = sp.sparse.block_diag(([R] * (N-1)))

  model_score = ScoreNet(H=H, W=W, in_channels=channels).to(device)
  loss_fn = torch.nn.MSELoss()
  optimizer = Adam(model_score.parameters(), lr=1e-3)

  model_score.train();

  for e in tqdm(range(epochs)):
    epoch_start_time = time.time()

    x = torch.zeros((channels, N, H, W))
    m = np.zeros((channels, N, H*W), dtype=np.float32)
    del_m = np.zeros_like(m, dtype=np.float32)
    m_c = np.zeros((channels, N, int((H*W/4))), dtype=np.float32)
    # we want to sample from random time steps to construct training samples
    random_t = np.linspace([dt + eps] * n_data, [1] * n_data, N-1, axis=1)
    random_t += np.random.uniform(-dt/2, dt/2, random_t.shape)
    random_t[:, -1] = 1
    time_ = np.sort(np.insert(random_t, 0, eps, axis=1), axis=1).astype(np.float32)
    sigma_ = diffusion_coeff(torch.tensor(time_), sigma).detach().cpu().numpy()
    for idx, x_ in enumerate(dataset):
      data = x_
      train_xc_data = torch.zeros((N, channels, int(H/2), int(W/2)))
      train_x_data = torch.zeros((N, channels, H, W))
      train_y_data = torch.zeros_like(train_x_data)
      
      # diffuse all three channels concurrently
      for i in range(channels):
        threads[i] = threading.Thread(target=diffuse, args=[data, x, m, del_m, m_c, i, train_xc_data, train_x_data, train_y_data, random_t[idx], time_[idx], sigma_[idx], idx, scores, kdes, N, H, W, R_block, P_block, sparse_R_block, sparse_P_block])
        threads[i].start()
      for thread in threads:
        thread.join()

      yc_pred = model_score(train_xc_data.to(device), torch.tensor(time_[idx]).to(device), coarse=True)
      lm = (2*torch.tensor(sigma_[idx])**2)[:, None, None, None]
      loss = loss_fn(yc_pred/lm, train_y_data)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses = loss.item()
      loss_c_hist.append(losses)

      y_pred = model_score(train_x_data.to(device), torch.tensor(time_[idx]).to(device))
      lm = (2*torch.tensor(sigma_[idx])**2)[:, None, None, None]
      loss = loss_fn(y_pred/lm, train_y_data)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses = loss.item()
      loss_hist.append(losses)
      scores = (y_pred/lm).clone().detach().cpu().numpy().reshape((N, channels, H, W)) # we normalize before feeding back into PDE

      epoch_times.append(time.time() - epoch_start_time)

  loss_df = pd.DataFrame(data = {"time": epoch_times, "coarse_loss": loss_c_hist, "loss": loss_hist})
  loss_df.to_csv("loss.log", index_label="epoch")

  torch.save(model_score.state_dict(), 'model_MG.pth')
  print(f"\nmodel has been saved")