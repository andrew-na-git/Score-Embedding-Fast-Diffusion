import numpy as np
from scipy import sparse
import torch
import scipy.stats as stats
from tqdm import tqdm
import functools
import time
import csv
import os

def construct_A(H,W,dh,dh2,f,df,g,s):
  a = (f - 0.5 * g**2 *s) * dh + 0.5 * g**2 * dh2
  b = 2 * g**2 * dh2
  c = (f - 0.5 * g**2 *s) * dh - 0.5 * g**2 * dh2

  Ddiag  = b * sparse.eye(H*W, format="csr")
  Dupper = sparse.diags(a[1:] , 1, format="csr")
  Dlower = sparse.diags(c[:-1], -1, format="csr")
  D_block = Ddiag + Dupper + Dlower
  B_upper_block = sparse.diags(a[H:], H, format="csr")
  C_lower_block = sparse.diags(c[:-H], -H, format="csr")
  A = sparse.eye(H*W, format="csr") + D_block + B_upper_block + C_lower_block + df * sparse.eye(H*W, format="csr") * dh
  return A

def construct_B(H, W, m_prev):
  B = m_prev
  return B

def score_samples(dataset, seed=None):
  if seed is not None:
    np.random.seed(seed)
  init_m_batch = []
  
  for data in dataset:
    init_m = []
    for ch in range(dataset.channels):
      y_train = []
      x_train = []
      
      for j in range(dataset.image_res):
        y_train.append(data[ch, j, :])
        x_train.append(data[ch, :, j])
      
      y_train = np.concatenate(y_train)
      x_train = np.concatenate(x_train)
      
      xy_train = np.vstack([x_train, y_train])
      kde_kernel = stats.gaussian_kde(xy_train)
      init_m.append(kde_kernel.logpdf(xy_train))
    init_m = np.vstack(init_m)
    init_m_batch.append(init_m[None])
  return np.vstack(init_m_batch)

def compute_scores(config, dataset, save_folder=None):
  seed = config.get("data_loader", {}).get("seed", None)
  if seed is not None:
    np.random.seed(seed)

  n_data = dataset.n_data
  channels = dataset.channels
  H = W = dataset.image_res

  N = config["diffusion"]["num_timesteps"]
  sigma = config["diffusion"]["sigma"]
  eps = float(config["misc"]["eps"])
  dt = 1/N
  tol = float(config["diffusion"]["solve_tolerance"])
  dh = config["diffusion"]["dh"]
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

  m = np.zeros((n_data, N, channels, H*W), dtype=np.float32)
  m_prev = np.ones((n_data, N, channels, H*W), dtype=np.float32)
  scores = np.ones((n_data, N, channels, H*W), dtype=np.float32) # initial scores guess
  dm = np.zeros_like(scores, dtype=np.float32)

  t_kde_start = time.time()
  initial_m = score_samples(dataset, seed=seed)
  t_kde = time.time() - t_kde_start

  # we want to sample from random time steps to construct training samples
  time_ = np.linspace(eps, 1, N).astype(np.float32)
  g = diffusion_coeff_fn
  def diffuse(idx, channel):
    m[idx, 0, channel] = initial_m[idx, channel]
    for i in range(1, N):
      A_block = construct_A(H, W, dt/(2*dh), dt/(dh**2), 0, 0, g(time_[i]), scores[idx, i, channel])
      B_block = construct_B(H, W, m[idx, i-1, channel])
      m[idx, i, channel] = sparse.linalg.spsolve(A_block, B_block).reshape((-1, H*W))
    img_log_prob = m[idx, :, channel]
    dm[idx, :, channel, 1:-1] = (img_log_prob[:, 2:] - img_log_prob[:, :-2])/(2*dh)
    dm[idx, :, channel, 0] = (img_log_prob[:, 1] - 0)/(2*dh)
    dm[idx, :, channel, -1] = (0 - img_log_prob[:, -2])/(2*dh)

  # Convergence log
  convergence_log = []  # list of (iteration, data_idx, residual, wall_time)

  res = [1] * n_data
  e = 0
  t_fp_start = time.time()
  while max(res) > tol:
    e += 1
    for idx in tqdm(range(n_data)):
      if res[idx] <= tol:
        continue
      # diffuse all three channels
      for ch in range(channels):
        diffuse(idx, ch)
      scores[idx] = dm[idx].copy()

      res[idx] = np.linalg.norm(m[idx] - m_prev[idx])/np.linalg.norm(m_prev[idx])
      wall = time.time() - t_fp_start
      convergence_log.append((e, idx, res[idx], wall))
      tqdm.write(f'residual at iteration {e} for data {idx}: {res[idx]}')
      m_prev[idx] = m[idx].copy()

  t_fp = time.time() - t_fp_start

  # Save convergence log and timing summary
  if save_folder is not None:
    conv_path = os.path.join(save_folder, "convergence_log.csv")
    with open(conv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["iteration", "data_idx", "residual", "wall_time_s"])
      writer.writerows(convergence_log)

    timing_path = os.path.join(save_folder, "timing.csv")
    with open(timing_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["stage", "time_s"])
      writer.writerow(["kde_init", f"{t_kde:.4f}"])
      writer.writerow(["fp_solve", f"{t_fp:.4f}"])
      writer.writerow(["fp_iterations", e])

  return scores.reshape((n_data, -1, channels, H, W))

def marginal_prob_std(t, sigma):
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  return sigma**t