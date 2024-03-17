import torch

def slice_wasserstein_loss(model, batch, t, diff_std2, std, z):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    batch: A mini-batch of training data.
  """
  score = model(batch, t).cpu()
  
  factor = 1/diff_std2
  loss = torch.mean(factor * torch.sum((score * std[:, None, None, None]+ z )**2, dim=(1, 2, 3)))

  return loss