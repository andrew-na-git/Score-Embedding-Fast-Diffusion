import sys

sys.path.insert(0, ".")

import torch
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import threading
import functools
from utils.kfp import diffusion_coeff, marginal_prob_std
from network.network import ScoreNet

error_tolerance = 1e-5

device = "cuda" if torch.cuda.is_available() else "cpu"

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cpu',
                input_channels=3,
                z=None,
                N= 20,
                H=28,
                W=28,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.tensor(np.linspace(1., eps, N), device=device)
  # Create the latent code
  if z is None:
    initial_x = torch.randn(batch_size, input_channels, H, W, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    initial_x = z + torch.randn(batch_size, input_channels, H, W, device=device) \
      * marginal_prob_std(t)[:, None, None, None]

  shape = initial_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    with torch.no_grad():
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), initial_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"\nNumber of function evaluations: {res.nfev}")
  x = []
  sample_points = np.floor(np.linspace(0, res.y.shape[1] - 1, num=N)).astype(int)
  for idx, i in enumerate(sample_points):
    x.append(res.y[:, i].reshape(shape)[idx][None])
  x = np.concatenate(x, axis = 0)
  return x, res.nfev

# function for sampling on a thread
def diffuse_sample(diffusion_coeff, marginal_prob_std, N, H, W):

  model_score = ScoreNet(marginal_prob_std=marginal_prob_std).to(device)
  file = f'model_cifar_thread_all.pth'
  ckpt = torch.load(file)
  model_score.load_state_dict(ckpt)
  model_score.eval();

  sample_batch_size = N
  sampler = ode_sampler

  # Generate samples using the specified sampler.
  output = sampler(model_score,
                  marginal_prob_std,
                  diffusion_coeff,
                  sample_batch_size,
                  input_channels=3,
                  N=N,
                  H=H,
                  W=W,
                  device=device)

  return output



def sample(n = 5, H=28, W=28, N=20, channels=3, sigma=2):
  #@title Sample each channel on a thread
  samples = []
  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

  
  samples, n_iter = diffuse_sample(diffusion_coeff_fn, marginal_prob_std_fn,N, H, W)

  def get_frame(i):
    return ((samples[i] - samples[i].min())/(samples[i].max() - samples[i].min())).transpose(1, 2, 0)

  
  fig, ax = plt.subplots(1, n, figsize=(32, 30/n))
  s = np.round(np.linspace(0, samples.shape[0] - 1, num=n)).astype(int)

  for idx, col in zip(s, ax):
      col.imshow(get_frame(idx), aspect="auto")
  
  return fig, n_iter