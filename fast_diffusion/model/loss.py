"""Training losses.

Naming
------
The loss below was called `slice_wasserstein_loss`. It is not a sliced Wasserstein
distance and has no connection to one: there is no projection onto random
directions and no 1-D optimal-transport cost anywhere in it. It is the standard
weighted denoising score-matching objective of Vincent (2011) / Song & Ermon (2019),

    E_t E_{x0} E_{z~N(0,I)} [ lambda(t) || s_theta(x0 + std(t) z, t) + z / std(t) ||^2 ]

written with the weight folded in as `1 / (2 * diff_std2)` and the residual scaled
by `std` so the bracket reads `(s*std + z)^2`. Shipping it under the old name in a
paper would misdescribe the method, so it is renamed here.
`slice_wasserstein_loss` remains as a deprecated alias so existing checkpointing
scripts keep working; it warns.
"""

import warnings

import torch


def denoising_score_matching_loss(model, batch, t, diff_std2, std, z, img_idx=None):
    """Weighted denoising score-matching loss for images.

    Parameters
    ----------
    batch : (B, C, H, W) perturbed samples x0 + std * z.
    t : (B,) diffusion times, already scaled by `training.timestep_multiplier`.
    diff_std2 : (B,) weight denominator; the loss weight is 1 / (2 * diff_std2).
    std : (B,) marginal standard deviation at each t.
    z : (B, C, H, W) the noise that was actually added to produce `batch`.
    """
    score = model(batch, t, img_idx=img_idx)

    factor = 1 / (2 * diff_std2)
    loss = torch.mean(
        factor * torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
    )
    return loss


def video_score_matching_loss(model, batch, t, diff_std2, std, z,
                              clip_idx=None, frame_idx=None, frame_weights=None):
    """The same objective for clips, summed over (T, C, H, W).

    Parameters
    ----------
    batch : (B, T, C, H, W) perturbed clips.
    frame_weights : optional (T,) or (B, T) per-frame weights. This is the hook for
        weighting frames by their sequential-importance weight, so frames the
        estimator flagged as poorly covered by the current score field contribute
        more. Left as None the loss is uniform over frames, which is the baseline
        this must be ablated against -- a per-frame weighting that is never compared
        against uniform is an unsupported claim.

    Note the reduction: the squared residual is summed over all of (T, C, H, W) and
    then averaged over the batch, matching the image loss's sum-over-pixels
    convention. Averaging over T instead would silently rescale the loss by 1/T
    relative to the image path and make learning rates non-transferable.
    """
    score = model(batch, t, clip_idx=clip_idx, frame_idx=frame_idx,
                  n_frames=batch.shape[1])

    resid = (score * std[:, None, None, None, None] + z) ** 2

    if frame_weights is not None:
        w = torch.as_tensor(frame_weights, dtype=resid.dtype, device=resid.device)
        if w.dim() == 1:
            w = w[None, :]
        resid = resid * w[:, :, None, None, None]

    factor = 1 / (2 * diff_std2)
    return torch.mean(factor * torch.sum(resid, dim=(1, 2, 3, 4)))


def slice_wasserstein_loss(model, batch, t, diff_std2, std, z, img_idx=None):
    """Deprecated alias for `denoising_score_matching_loss`.

    The original name described the objective incorrectly; see the module docstring.
    """
    warnings.warn(
        "slice_wasserstein_loss is a misnomer for the denoising score-matching "
        "objective it computes; use denoising_score_matching_loss instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return denoising_score_matching_loss(model, batch, t, diff_std2, std, z, img_idx)
