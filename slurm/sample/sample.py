import sys

sys.path.insert(0, ".")

import torch
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from network.network import ScoreNet
from utils.kfp import diffusion_coeff

device = "gpu" if torch.cuda.is_available() else "cpu"

## The error tolerance for the black-box ODE solver
def ode_sampler(score_model,
                diffusion_coeff,
                batch_size=64,
                H=28,
                W=28,
                channels=3,
                sigma=25,
                z=None,
                eps=1e-6):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, channels, H, W) * torch.sqrt(2 * diffusion_coeff(torch.ones(batch_size, channels, H, W), sigma)**2)
  else:
    init_x = z

  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, dtype=torch.float32).reshape(shape)

    time_steps = torch.tensor(time_steps, dtype=torch.float32).reshape((sample.shape[0],))
    with torch.no_grad():
      score_ = score_model(sample.to(device), time_steps.to(device))
    return score_.cpu().numpy().reshape((-1))

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    # time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t), sigma)
    return  -0.5 * (g**2) * score_eval_wrapper(x, t)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1, eps), init_x.reshape(-1))
  #print(f"Number of function evaluations: {res.nfev}")
  sol = res.y #[:, -1].reshape(shape)
  return sol, res.nfev

def sample(model, n = 5, H=28, W=28, channels=3, sigma=25):
  model_score = ScoreNet(H=H, W=W, in_channels=channels).to(device)

  sample_batch_size = 1
  sampler = ode_sampler

  ckpt = torch.load(model, map_location=device)
  model_score.load_state_dict(ckpt)
  model_score.eval();

  init_x = torch.rand(sample_batch_size, channels, H, W)

  ## Generate samples using the specified sampler.
  samples, n_eval = sampler(model_score,
                    diffusion_coeff,
                    sample_batch_size,
                    H,
                    W,
                    channels,
                    sigma)
  fig, ax = plt.subplots(1, n, figsize=(32, 30/n))

  s = np.round(np.linspace(0, samples.shape[-1] - 1, num=n)).astype(int)

  for idx, col in zip(s, ax):
      col.imshow(np.moveaxis(np.clip((samples[:, idx] + abs(np.min(samples[:, idx]))) / (np.max(samples[:, idx]) + abs(np.min(samples[:, idx]))), 0, 1).reshape((channels, H, W)), 0, 2), aspect="auto")
  
  return fig, n_eval