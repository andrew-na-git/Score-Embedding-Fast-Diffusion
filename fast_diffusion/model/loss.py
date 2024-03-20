import torch

def slice_wasserstein_loss(model, batch, t, diff_std2, std, z):
  score = model(batch, t).cpu()
  
  factor = 1/(2 * diff_std2)
  loss = torch.mean(factor * torch.sum((score * std[:, None, None, None]+ z )**2, dim=(1, 2, 3)))

  return loss