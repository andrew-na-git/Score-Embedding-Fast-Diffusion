import torch
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"

def ddim_sampler(config, model):
    # match these with what you used during training
    N = config["diffusion"]["num_timesteps"]
    H = W = config["data_loader"]["image_size"]
    beta_min = config["diffusion"]["beta_min"]
    beta_max = config["diffusion"]["beta_max"]

    # number of samples to generate
    num_samples = config["data_loader"]["num_images"]

    num_channels = config["data_loader"]["channels"]
    discrete_betas = torch.linspace(beta_min, beta_max, N)
    def compute_alpha(beta, t):
        betas = torch.cat([torch.zeros(1), beta], dim=0)
        a = (1 - betas.to(t.device)).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    # inital sample
    x = torch.randn((num_samples, num_channels, H, W), device=device)
    
    samples = [x.cpu()]
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
        c2 = ((1 - at_next)).sqrt()
        xt_next = at_next.sqrt() * x0_t + c2 * et
        samples.append(xt_next.cpu())
    
    return [x.numpy() for x in samples]

def ddpm_sampler(config, model):
    N = config["diffusion"]["num_timesteps"]
    H = W = config["data_loader"]["image_size"]
    beta_min = config["diffusion"]["beta_min"]
    beta_max = config["diffusion"]["beta_max"]

    # number of samples to generate
    num_samples = config["data_loader"]["num_images"]

    num_channels = config["data_loader"]["channels"]
    discrete_betas = torch.linspace(beta_min, beta_max, N)

    alphas = 1. - discrete_betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # inital sample
    x = torch.randn((num_samples, num_channels, H, W), device=device)

    timesteps = torch.linspace(1, 1e-3, N, device=device)

    def sample_update(idx, x):
        t = timesteps[idx]

        vec_t = torch.ones(num_samples, device=t.device) * t
        timestep = (vec_t * (N - 1)).long()
        beta = discrete_betas.to(t.device)[timestep]
        labels = vec_t * (N-1)
        score = - model(x, labels) / sqrt_1m_alphas_cumprod.to(x.device)[labels.long()][:, None, None, None]
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean
    
    samples = []
    for i in tqdm(range(N)):
        x, x_mean = sample_update(i, x)
        samples.append(x_mean.cpu().numpy())

    return samples

def sample(model, config, ema_helper=None, sample_method="ddpm"):    
    model.eval()

    print(f"Sampling {sample_method}...")

    # restore saved model
    if ema_helper:
        ema_helper.store(model.parameters())
        ema_helper.copy_to(model.parameters())

    with torch.no_grad():
        if sample_method == "ddim":
            samples = ddim_sampler(config, model)
        elif sample_method == "ddpm":
            samples = ddpm_sampler(config, model)
        else:
            raise NotImplementedError()

    # just keep 50 out of 1000 steps
    idx = np.linspace(0, len(samples) - 1, num=50).astype(int)

    if ema_helper:
        ema_helper.restore(model.parameters())

        
    inverse_scaler = lambda x: (x + 1.) / 2
    return np.stack([inverse_scaler(samples[i]) for i in idx])
