import time
import sys
sys.path.insert(0, "../..")

from network.network import Net
from .ema import ExponentialMovingAverage
import torch.optim as optim
from data.Dataset import get_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from .sample import sample
from data.metrics import mse_metric, ssim_metric
import shutil

from pathlib import Path

device = "cuda" if torch.cuda.is_available else "cpu"

def loss_fn(model, batch, N, sac, s1ac):
    timesteps = torch.randint(0, N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = sac.to(batch.device)
    sqrt_1m_alphas_cumprod = s1ac.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[timesteps, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[timesteps, None, None, None] * noise
    score = model(perturbed_data, timesteps)
    losses = torch.square(score - noise)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

def train(config, save_folder, profile=False):
    N = config["diffusion"]["num_timesteps"] ## num timesteps
    beta_min = config["diffusion"]["beta_min"]
    beta_max = config["diffusion"]["beta_max"]
    epochs = config["training"]["epochs"]
    batch_size = config["data_loader"]["batch_size"]
    ema = config["training"]["ema"]
    H = W = config["data_loader"]["image_size"]
    lr = float(config["training"]["lr"])
    warmup = config["training"]["warmup"]
    grad_clip = config["training"]["grad_clip"]
    sample_method = config["sample_method"]

    dataset = get_dataset(config, repeat=batch_size)
    np.save(os.path.join(save_folder, "dataset.npy"), dataset.data)

    # if profiling, create sample folder:
    if profile:
        sample_folder = os.path.join(save_folder, "samples")
        if os.path.exists(sample_folder):
            shutil.rmtree(sample_folder)
        Path(sample_folder).mkdir(parents=True, exist_ok=True)

    # create model
    model = Net(config)
    model.to(device)
    model.train()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num Parameters: {pytorch_total_params}")

    if ema:
        ema_helper = ExponentialMovingAverage(model.parameters(), decay=ema)

    discrete_betas = torch.linspace(beta_min, beta_max, N)
    alphas = 1. - discrete_betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scaler = lambda x: x * 2 - 1

    loss_hist = []
    last_epoch_time = time.time()
    cur_running_time = 0
    last_save_time = 0
    mses = []
    ssims = []
    profile_times = []
    profile_epochs = []
    for e in tqdm(range(epochs)):
        x = next(iter(dataloader))
        x = scaler(x).to(device).float().reshape(batch_size, dataset.channels, H, W)
        optimizer.zero_grad()
        loss = loss_fn(model, x, N, sqrt_alphas_cumprod, sqrt_1m_alphas_cumprod)
        loss.backward()
        
        # adaptive lr is here, can comment out
        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(e / warmup, 1.0)

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if ema:
            ema_helper.update(model.parameters())
        
        tqdm.write(f"Epoch {e} Loss: {loss.item()}")
        loss_hist.append(loss.item())

        cur_running_time += time.time() - last_epoch_time
        

        ### time between here is not counted ###
        if profile and cur_running_time - last_save_time > config["misc"]["profile_freq"]: # sample every n seconds
            print("Saving sample checkpoint...", flush=True)

            with torch.no_grad():
                samples = sample(model, config, ema_helper=ema_helper, sample_method=sample_method, ground_truths=dataset.data)
                
                np.save(os.path.join(save_folder, "samples", f"sample_{round(cur_running_time, 2)}.npy"), samples)
                mses.append(mse_metric([x.numpy() for x in dataset.data], samples[-1]))
                ssims.append(ssim_metric([x.numpy() for x in dataset.data], samples[-1]))
                print(str(ssims[-1]) + '\n')
                profile_times.append(cur_running_time)
                profile_epochs.append(e)
                model.train()
                last_save_time = cur_running_time

        ########################################

        last_epoch_time = time.time()

    if profile:
        # save final sample
        with torch.no_grad():
            print("Saving final sample...", flush=True)
            samples = sample(model, config, ema_helper=ema_helper, sample_method=sample_method, ground_truths=dataset.data)
            
            np.save(os.path.join(save_folder, "samples", f"sample_{round(cur_running_time, 2)}.npy"), samples)           
            mses.append(mse_metric([x.numpy() for x in dataset.data], samples[-1]))
            ssims.append(ssim_metric([x.numpy() for x in dataset.data], samples[-1]))
            profile_times.append(cur_running_time)
            profile_epochs.append(e)
            last_save_time = cur_running_time

    metrics = dict(losses = loss_hist, mse = mses, ssim = ssims, times = profile_times, epochs = profile_epochs)


    print("Saving model")
    state = dict(model=model.state_dict(), ema=ema_helper.state_dict(), config=config, metrics=metrics, train_time=cur_running_time)
    torch.save(state, os.path.join(save_folder, "model.pth"))