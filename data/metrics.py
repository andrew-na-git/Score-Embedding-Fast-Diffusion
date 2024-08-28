from skimage.metrics import structural_similarity
from torchmetrics.image.fid import FrechetInceptionDistance

def mse_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        norm_diffuse = (diffuse - diffuse.min()) / (diffuse.max() - diffuse.min())
        losses.append(((ground - norm_diffuse) ** 2).mean())

    return losses

def ssim_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        norm_diffuse = (diffuse - diffuse.min()) / (diffuse.max() - diffuse.min())
        losses.append(structural_similarity(ground, norm_diffuse, channel_axis=0, data_range=1))

    return losses

def fid_metric(ground_truths, diffused):
    losses = []
    for ground, diffuse in zip(ground_truths, diffused):
        fid = FrechetInceptionDistance(feature=64)
        fid.update(ground, real=True)
        fid.update(diffuse, real=False)
        losses.append(fid.compute().detach().numpy()[0])

    return losses