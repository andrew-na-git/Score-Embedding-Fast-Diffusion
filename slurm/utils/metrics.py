import numpy as np
from skimage.measure import compare_ssim

def l2_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        losses.append(np.sqrt(np.sum((ground - diffused) ** 2)))

    return losses

def ssim_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        losses.append(compare_ssim(ground, diffuse))

    return losses