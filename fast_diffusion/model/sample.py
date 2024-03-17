import torch
from scipy import integrate
import numpy as np
from .kfp import diffusion_coeff

def __ode_sampler(score_model,
                sigma,
                batch_size=1,
                input_channels=3,
                z=None,
                cond_weight = 0.1,
                H=32,
                W=32,
                temb_method="linear",
                timestep_multiplier = 1000):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    batch_size: The number of samplers to generate by calling this function once.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
  """
  # Create the latent code
  
  atol = rtol = 1e-3
  device = "cuda" if torch.cuda.is_available() else "cpu"
  eps = 1e-3

  if z is None:
    initial_x = torch.randn(batch_size, input_channels, H, W, device=device)
  else:
    initial_x = torch.tensor(z, device=device) * cond_weight + (1 - cond_weight) * torch.tensor(np.random.normal(scale = sigma, size = (batch_size, input_channels, H, W)), device=device)

  shape = initial_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    if temb_method == "linear":
      time_steps += torch.tensor(list(range(sample.shape[0])), device=device) * 2 - batch_size + 0.5
    else:
     time_steps *= timestep_multiplier
    with torch.no_grad():
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t), sigma).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), initial_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"\nNumber of function evaluations: {res.nfev}")
  x = []
  for i in range(res.y.shape[1]):
    x.append(res.y[:, i].reshape(shape))

  return np.array(x), res.nfev

def unconditional_sample(model, config):
  sigma = config["diffusion"]["sigma"]
  batch_size = config["data_loader"]["num_images"]
  channels = config["data_loader"]["channels"]
  H = W = config["data_loader"]["image_size"]
  time_embedding_method = config["model"]["embedding_method"]
  timestep_multiplier = config["training"]["timestep_multiplier"]
  samples, n_eval = __ode_sampler(model, sigma, batch_size, channels, None, None, H, W, time_embedding_method, timestep_multiplier=timestep_multiplier)

  # only save 50 steps
  idx = np.linspace(0, len(samples) - 1, num=50).astype(int)
  return samples[idx], n_eval

def conditional_sample(model, config, dataset, cond_weight = 0.1):
  sigma = config["diffusion"]["sigma"]
  timestep_multiplier = config["training"]["timestep_multiplier"]

  channels = config["data_loader"]["channels"]
  H = W = config["data_loader"]["image_size"]
  time_embedding_method = config["model"]["embedding_method"]
  samples, n_eval = __ode_sampler(model, sigma, len(dataset), channels, dataset, cond_weight, H, W, time_embedding_method, timestep_multiplier=timestep_multiplier)

  # only save 50 steps
  idx = np.linspace(0, len(samples) - 1, num=50).astype(int)
  return samples[idx], n_eval
