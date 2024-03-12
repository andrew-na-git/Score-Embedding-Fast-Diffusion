

import sys
sys.path.insert(0, "../..")

from network.ddpm.ddpm import DDPM
from ema import ExponentialMovingAverage
import torch.optim as optim
from data.Dataset import CIFARDataset, CelebDataset, ImageNetDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np

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

def train():
    N = 1000 # num timesteps
    beta_min = 4
    beta_max = 4
    epochs = 1500
    batch_size = 20
    # height and width of image (change these in the sample function too!)
    H = W = 32
    #pick dataset
    #dataset = CelebDataset(H=H, W=W, n=1, repeat=batch_size)
    #dataset = ImageNetDataset(H=H, W=W, n=1, repeat=batch_size)
    dataset = CIFARDataset(H=H, W=W, n=1, seed=9, repeat=batch_size)

    device = "cuda" if torch.cuda.is_available else "cpu"

    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    alphas = 1. - discrete_betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    model = DDPM(H=H, time_embedding_method="fourier").to(device)
    model.train()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scaler = lambda x: x * 2 - 1


    warmup = 5000
    for e in tqdm(range(epochs)):
        total_loss = 0
        n = 0
        for x in dataloader:
            x = scaler(x).to(device).float().reshape(batch_size, 3, H, W)
            optimizer.zero_grad()

            loss = loss_fn(model, x, N, sqrt_alphas_cumprod, sqrt_1m_alphas_cumprod)

            loss.backward()
            
            # adaptive lr is here, can comment out
            for g in optimizer.param_groups:
                g['lr'] = 2e-4 * np.minimum(e / warmup, 1.0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            ema.update(model.parameters())
            total_loss += loss.item()
            n += 1
        tqdm.write(f"Epoch {e} Loss: {total_loss / n}")

    print("Saving model")
    state = dict(model=model.state_dict(), ema=ema.state_dict())
    torch.save(state, "model.pth")


def compute_alpha(beta, t):
    betas = torch.cat([torch.zeros(1), beta], dim=0)
    a = (1 - betas.to(t.device)).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def sample(model_path):
    print("Sampling...")
    state = torch.load(model_path)
    
    # match these with what you used during training
    N = 1000
    H = W = 32
    beta_min = 4
    beta_max = 4

    # number of samples to generate
    num_samples = 1

    num_channels = 3
    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    device = "cuda" if torch.cuda.is_available else "cpu"

    # restore saved model
    model = DDPM(H=H, time_embedding_method="fourier").to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    ema.load_state_dict(state["ema"])
    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    inverse_scaler = lambda x: (x + 1.) / 2

    with torch.no_grad():
        # inital sample
        x = torch.randn((num_samples, num_channels, H, W), device=device)
        
        samples = [x]
        x0_preds = []
        seq = list(range(0, N))
        seq_next = [-1] + list(seq[:-1])
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq)):
            t = (torch.ones(num_samples) * i).to(x.device)
            next_t = (torch.ones(num_samples) * j).to(x.device)
            at = compute_alpha(discrete_betas, t.long())
            at_next = compute_alpha(discrete_betas, next_t.long())
            xt = samples[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c2 = ((1 - at_next)).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et
            samples.append(xt_next.to('cpu'))

    # just keep 50 out of 1000 images
    idx = np.linspace(0, len(samples) - 1, num=50).astype(int)
    return torch.stack([inverse_scaler(samples[i].cpu()) for i in idx])

train()

samples = sample("model.pth")

np.save("samples.npy", samples.cpu())