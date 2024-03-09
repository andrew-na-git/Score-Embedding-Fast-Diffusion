import time
import functools
import time
import os
import multiprocessing

import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

import torch
from torch.optim import Adam, SGD
import scipy.stats as stats
from scipy import sparse
import pandas as pd

from sample.sample import ode_sampler
from utils.kfp import get_B_block, diffusion_coeff, solve_pde, get_sparse_A_block, marginal_prob_std, construct_A, construct_B
from network.network import ScoreNet

# Parameters
eps = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Found {multiprocessing.cpu_count()} cores")

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


#@title Defining pde diffusion for multi-threading
def diffuse(x, m, dm, channel, time_, g, scores, dt, H, W, N, kdes, dx_num):

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
    # kde_kernel = stats.gaussian_kde(xy_train)
    # kdes[channel] = kde_kernel.logpdf(xy_train)

    kde_kernel = KernelDensity(kernel="linear").fit(xy_train.T)
    # kdes[channel] = parrallel_score_samples(kde_kernel, xy_train.T)
    kdes[channel] = kde_kernel.score_samples(xy_train.T)
  m[0, channel] = kdes[channel]
  dh = dx_num

  for i in range(1, N):
    A_block = construct_A(H, W, dt/(2*dh), dt/(dh**2), 0, 0, g(time_[i]), scores[i, channel]) # A_block.append(sparse.csr_matrix(A))
    B_block = construct_B(H, W, m[i-1, channel])
    m[i, channel] = sparse.linalg.spsolve(A_block, B_block).reshape((-1, H*W))

  
  #sparse_A_block = get_sparse_A_block(dx, dt, g(time_[1:]), scores[1:, channel], H, W, N)

  #B_block = get_B_block(dx, dt, m, channel, H, W, N)

  #[1:, channel] = solve_pde(sparse_A_block, B_block, mode='sp_sparse').reshape((-1, H*W))
  
  img_log_prob = m[:, channel]#.reshape((-1, H, W))

  dm[:, channel, 1:-1] = (img_log_prob[:, 2:] - img_log_prob[:, :-2])/(2*dh) # (img_log_prob[:, 1:-1, 2:] - img_log_prob[:, 1:-1, :-2])/(2*dy)
  dm[:, channel, 0] = (img_log_prob[:, 1] - 0)/(2*dh) #+ (img_log_prob[:, 1:-1, 0] - img_log_prob[:, 1:-1, -1])/dy
  dm[:, channel, -1] = (0 - img_log_prob[:, -2])/(2*dh) #+ (img_log_prob[:, 1:-1, -1] - img_log_prob[:, 1:-1, 0])/dy


  # dm[:, channel, 1:-1] = (img_log_prob[:,:-2] - img_log_prob[:,2:])/(2*dx)
  # dm[:, channel, 0] = dm[:, channel, 1]
  # dm[:, channel, -1] = dm[:, channel, -2]
  
  
  # dm[:, channel, 1:-1 , 1:-1] = (img_log_prob[:, 2:, 1:-1] - img_log_prob[:, :-2, 1:-1])/(2*dx) + (img_log_prob[:, 1:-1, 2:] - img_log_prob[:, 1:-1, :-2])/(2*dy)
  # dm[:, channel, 1:-1 , 0] = (img_log_prob[:, 2:, 0] - img_log_prob[:, :-2, 0])/(2*dx) + (img_log_prob[:, 1:-1, 0])/(dy)
  # dm[:, channel, 1:-1 , -1] = (img_log_prob[:, 2:, -1] - img_log_prob[:, :-2, -1])/(2*dx) + (img_log_prob[:, 1:-1, -1])/(dy)
  # dm[:, channel, 0 , 1:-1] = (img_log_prob[:, 0, 2:] - img_log_prob[:, 0, :-2])/(2*dy) + (img_log_prob[:, 0, 1:-1])/(dx)
  # dm[:, channel, -1 , 1:-1] = (img_log_prob[:, -1, 2:] - img_log_prob[:, -1, :-2])/(2*dy) + (img_log_prob[:, -1, 1:-1])/(dx)
    
# do training

def loss_fn(model, optimizer, x, label, diffusion_coeff, marginal_prob_std, dt, N, idx, n_data, temb_method, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  #random_t = torch.tensor(np.sort(np.random.uniform(eps, 1., N)).astype(np.float32))
  
  random_t_sample = (np.linspace(dt + eps, 1., N, endpoint=False) + (np.random.rand(N) - 0.5) * dt).astype(np.float32)
  random_t = torch.tensor(random_t_sample)

  # we encode the label into the initial data using the reverse ODE
  diff_std2 = diffusion_coeff(2 * random_t)
  #print(random_t, random_t_test, flush=True)

  for i in range(1, N):
    x[i] = x[i-1] - 0.5 * label[i] * diff_std2[i] * dt
  std = marginal_prob_std(random_t)
  z = torch.randn_like(x)
  # we perturb the image by the forward SDE conditional distribution
  perturbed_x = x + z * std[:, None, None, None]
  scores = []

  batch_size = N
  total_loss = 0
  for i in range(N//batch_size):
    
    if temb_method == "linear":
        score = model(perturbed_x[i * batch_size:(i+1) * batch_size].to(device), (random_t + idx * 2 - n_data + 0.5)[i * batch_size: (i+1) * batch_size].to(device)).cpu()
    else:
      score = model(perturbed_x[i * batch_size:(i+1) * batch_size].to(device), (random_t * 9)[i * batch_size: (i+1) * batch_size].to(device)).cpu()
  # loss = torch.mean(torch.sum((score * std[:, None, None, None] - label)**2, dim=(1, 2, 3)) / (2 * diff_std2))
    loss = torch.mean(torch.sum((score * std[i * batch_size:(i+1) * batch_size][:, None, None, None] + z[i * batch_size:(i+1) * batch_size])**2, dim=(1, 2, 3))) # original loss from tutorial
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1/(N/batch_size))
    optimizer.step()
    total_loss += loss.item()

  return total_loss / (N//batch_size)

def diffuse_train(model, init_x, epoch, diffusion_coeff, marginal_prob_std, label, dt, N, loss_hist, lost_hist_per_image, comparison=False, model_save_folder = None, temb_method="linear", time_between_samples=30):
  last_epoch_time = time.time()
  model_score = model.to(device)
  optimizer = Adam(model_score.parameters(), lr=1e-4)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001)
  model_score.train();

  cur_running_time = 0
  last_save_time = 0
  scores_label = torch.tensor(label)
  for e in tqdm(range(epoch)):
    total_loss = 0
    for i in range(init_x.shape[0]):
      loss = loss_fn(model_score, optimizer, init_x[i], scores_label[i], diffusion_coeff, marginal_prob_std, dt, N, i, init_x.shape[0], temb_method)
      lost_hist_per_image[i, e] = loss
      # optimizer.zero_grad()
      # loss.backward()
      # torch.nn.utils.clip_grad_norm_(model_score.parameters(), 1)
      # optimizer.step()
      total_loss += loss#.item()
    loss_hist[e] = total_loss / init_x.shape[0]

    #scheduler.step(loss_hist[e])
    tqdm.write(f'Epoch {e + 1}/{epoch}, Avg Loss: {loss_hist[e]:.4f}')
    # if loss_hist[e] < 40:
    #   print("Early stopping")
    #   break

    cur_running_time += time.time() - last_epoch_time
    
    ## time between here is not counted ##
    if comparison and cur_running_time - last_save_time > time_between_samples: # sample every n seconds
      print("Saving sample checkpoint...", flush=True)
      samples, iters = ode_sampler(model_score, marginal_prob_std, diffusion_coeff, batch_size=init_x.shape[0], device=device, H = init_x.shape[-2], W=init_x.shape[-1], N=N, temb_method=temb_method)
      #torch.save(model_score.state_dict(), os.path.join(model_save_folder, f"checkpoint_{int(cur_running_time)}.pth"))
      np.save(os.path.join(model_save_folder, f"sample_{round(cur_running_time, 2)}.npy"), samples)
      last_save_time = cur_running_time
    
    # elif comparison and np.mean(loss_hist[e-4:e+1]) < 100 and cur_running_time - last_save_time > 2: # if loss is low, save every 2 seconds
    #   print("Saving model checkpoint because of low loss...", flush=True)
    #   torch.save(model_score.state_dict(), os.path.join(model_save_folder, f"checkpoint_{int(cur_running_time)}.pth"))
    #   last_save_time = cur_running_time

    #####################################
    last_epoch_time = time.time()

  if comparison:
    print("Saving final sample...", flush=True)
    samples, iters = ode_sampler(model_score, marginal_prob_std, diffusion_coeff, batch_size=init_x.shape[0], device=device, H = init_x.shape[-2], W=init_x.shape[-1], N=N, temb_method=temb_method)
    #torch.save(model_score.state_dict(), os.path.join(model_save_folder, f"checkpoint_{int(cur_running_time)}.pth"))
    np.save(os.path.join(model_save_folder, f"sample_{round(cur_running_time, 2)}.npy"), samples)

  print(f'\nloss at all channels: {loss_hist[-1]}')
  file = os.path.join(model_save_folder, f'model_final.pth')
  torch.save(model_score.state_dict(), file)
  print(f"model has been saved\n")

  return cur_running_time

def train(model, dataset, N=10, H=28, W=28, channels=3, epochs=1000, sigma=2, comparison=False, model_save_folder = None, tol=1e-3, dx_num=128, temb_method="linear", time_between_samples=30):
  print(f"Training using device: {device}")
  total_train_time = 0
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
  res = [1] * n_data
  e = 0
  convergence_iteration = [1000] * n_data
  while max(res) > tol and e <= 100:
    for idx, data in tqdm(enumerate(dataset)):
      if res[idx] <= tol:
        convergence_iteration[idx] = e if convergence_iteration[idx] == 1000 else convergence_iteration[idx]
        continue
      # diffuse all three channels
      for ch in range(channels):
        diffuse(data, m[idx], dm[idx], ch, time_, diffusion_coeff_fn, scores[idx], dt, H, W, N, kdes[idx], dx_num)

      scores[idx] = dm[idx].copy()

      if e == 1000:
        print(f'No convergence')
        exit(1)

      res[idx] = np.linalg.norm(m[idx] - m_prev[idx])/np.linalg.norm(m_prev[idx])
      print(f'residual at iteration {e} for data {idx}: {res[idx]}', flush=True)
      m_prev[idx] = m[idx].copy()
    e += 1
  
  diffusion_time = time.time() - start_time

  scores_label = scores.copy().reshape((n_data, -1, channels, H, W))
  
  print("Saving scores")
  np.save(os.path.join(os.path.dirname(model_save_folder), "scores.npy"), scores_label.swapaxes(0, 1))

  losses = np.zeros((n_data, epochs))
  total_loss = np.zeros(epochs)
  init_x = torch.zeros((n_data, N, channels, H, W))

  for idx, data in enumerate(dataset):
    init_x[idx] = data

  running_time = diffuse_train(model, init_x, epochs, diffusion_coeff_fn, marginal_prob_std_fn, scores_label, dt, N, total_loss, losses, comparison, model_save_folder, temb_method, time_between_samples)

  data={"running_time": [running_time] * epochs, "diffusion_time": [diffusion_time] * epochs, "loss": total_loss}
  for i in range(n_data):
    data[f"loss_{i}"] = losses[i]
  log_df = pd.DataFrame(data = data)
  log_df.to_csv(os.path.join(os.path.dirname(model_save_folder), "loss.log"), index_label="epoch")

  return diffusion_time, running_time
