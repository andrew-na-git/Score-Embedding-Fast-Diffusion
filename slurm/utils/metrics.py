import numpy as np
from skimage.metrics import structural_similarity

def l2_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        norm_diffuse = (diffuse - diffuse.min()) / (diffuse.max() - diffuse.min())
        losses.append(np.sqrt(np.sum((ground - norm_diffuse) ** 2)))

    return losses
  
def mse_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        norm_diffuse = (diffuse - diffuse.min()) / (diffuse.max() - diffuse.min())
        losses.append(np.mean((ground - norm_diffuse) ** 2))

    return losses

def ssim_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        norm_diffuse = (diffuse - diffuse.min()) / (diffuse.max() - diffuse.min())
        losses.append(structural_similarity(ground, norm_diffuse, channel_axis=0, data_range=1))

    return losses