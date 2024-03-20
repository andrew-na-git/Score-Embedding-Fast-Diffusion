import time
import os

import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from data.Dataset import get_dataset
from data.metrics import mse_metric, ssim_metric

from .kfp import compute_scores
from network.network import Net
from .sample import unconditional_sample, conditional_sample
from .loss import slice_wasserstein_loss
from .dataloader import create_batch
import shutil

from pathlib import Path

# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"


def diffuse_train(model, dataset, scores, config, save_folder, profile=False):
  model_score = model.to(device)
  epochs = config["training"]["epochs"]
  optimizer = Adam(model_score.parameters(), lr=float(config["training"]["lr"]))
  model_score.train();
  
  last_epoch_time = time.time()
  cur_running_time = 0
  last_save_time = 0
  scores_label = torch.tensor(scores)
  
  N = config["diffusion"]["num_timesteps"]
  grad_clip = config["training"]["grad_clip"]
  sample_method = config["sample"]["type"]
  
  init_x = torch.zeros((dataset.n_data, N, dataset.channels, dataset.image_res, dataset.image_res))
  for idx, data in enumerate(dataset):
    init_x[idx] = data
  
  loss_hist_per_image = np.zeros((dataset.n_data, epochs))
  mses = []
  ssims = []
  profile_times = []
  profile_epochs = []
  for e in tqdm(range(epochs)):
    for i, data in enumerate(dataset):
      batch, t, diff_std2, std, z = create_batch(init_x[i], scores_label, config, i)
      loss = slice_wasserstein_loss(model_score, batch.to(device), t.to(device), diff_std2, std, z)
      optimizer.zero_grad()
      loss.backward()

      if grad_clip:
        torch.nn.utils.clip_grad_norm_(model_score.parameters(), grad_clip)
      
      optimizer.step()

      loss_hist_per_image[i, e] = loss.item()

    tqdm.write(f'Epoch {e + 1}/{epochs}, Avg Loss: {np.mean(loss_hist_per_image[:, e]):.4f}')
    cur_running_time += time.time() - last_epoch_time
    
    ## time between here is not counted ##
    if profile and cur_running_time - last_save_time > config["misc"]["profile_freq"]: # sample every n seconds
      with torch.no_grad():
        print("Saving sample checkpoint...", flush=True)
        if sample_method == "unconditional":
          samples, _ = unconditional_sample(model_score, config)
        else:
          conditional_weight = config["sample"]["conditional_weight"]
          samples, _ = conditional_sample(model_score, config, dataset.data, conditional_weight)
        np.save(os.path.join(save_folder, "samples", f"sample_{round(cur_running_time, 2)}.npy"), samples)
        mses.append(mse_metric([x.numpy() for x in dataset.data], samples[-1]))
        ssims.append(ssim_metric([x.numpy() for x in dataset.data], samples[-1]))
        profile_times.append(cur_running_time)
        profile_epochs.append(e)
        model.train()
        last_save_time = cur_running_time
    #####################################
    last_epoch_time = time.time()

  if profile:
    # save final sample
    with torch.no_grad():
      print("Saving final sample...", flush=True)

      if sample_method == "unconditional":
        samples, _ = unconditional_sample(model_score, config)
      else:
        conditional_weight = config["sample"]["conditional_weight"]
        samples, _ = conditional_sample(model_score, config, dataset.data, conditional_weight)

      mses.append(mse_metric([x.numpy() for x in dataset.data], samples[-1]))
      ssims.append(ssim_metric([x.numpy() for x in dataset.data], samples[-1]))
      profile_times.append(cur_running_time)
      profile_epochs.append(epochs)
      np.save(os.path.join(save_folder, "samples", f"sample_{round(cur_running_time, 2)}.npy"), samples)
  
  metrics = dict(losses = loss_hist_per_image, mse = mses, ssim = ssims, times = profile_times, epochs = profile_epochs)
  return cur_running_time, metrics


def train(config, save_folder, profile=False):
  print(f"Training using device: {device}")
  dataset = get_dataset(config)
  np.save(os.path.join(save_folder, "dataset.npy"), dataset)

  ## create model
  model = Net(config)
  pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Num Parameters: {pytorch_total_params}")
  
  ## compute the scores
  start_time = time.time()
  scores_label = compute_scores(config, dataset)
  diffusion_time = time.time() - start_time

  # if profiling, create sample folder:
  if profile:
    sample_folder = os.path.join(save_folder, "samples")
    if os.path.exists(sample_folder):
      shutil.rmtree(sample_folder)
    Path(sample_folder).mkdir(parents=True, exist_ok=True)
  
  print("Saving scores")
  np.save(os.path.join(save_folder, "scores.npy"), scores_label.swapaxes(0, 1))

  ## train
  train_time, metrics = diffuse_train(model, dataset, scores_label, config, save_folder, profile)

  state = dict(model = model.state_dict(), config = config, diffusion_time = diffusion_time, train_time=train_time, metrics = metrics)
  file = os.path.join(save_folder, f'model.pth')
  torch.save(state, file)
  print(f"model has been saved\n")

