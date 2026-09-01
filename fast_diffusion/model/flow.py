"""Optical flow and warping for the sequential density estimator.

The sequential scheme in `density.SequentialDensityEstimator` treats frame k-1's
converged log-density as a proposal for frame k. That proposal is only useful if
it is advected along the scene motion first, which is what this module provides.

Flow backends
-------------
raft     : torchvision's RAFT (`raft_small` / `raft_large`) with pretrained
           weights. Accurate, needs a GPU for anything but small clips.
blockmatch : dependency-free integer-shift block matching. Coarse, but it makes
           the no-external-weights path runnable and serves as the mid-strength
           entry in the warp-source ablation (PLAN.md 4.1).
identity : zero flow. The control condition -- it isolates how much of the
           sequential scheme's benefit comes from motion compensation versus
           plain reweighting. Report it.

Occlusion handling
------------------
`forward_backward_mask` flags pixels where forward and backward flow disagree,
which is where content is disoccluded or leaves frame. Those pixels have no valid
proposal and must not contribute importance weight; feed the mask to
`warp_samples` so they are dropped rather than warped from nonsense.
"""

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Flow estimation
# --------------------------------------------------------------------------

_RAFT_CACHE = {}


def _load_raft(variant="small", device="cpu"):
    key = (variant, str(device))
    if key in _RAFT_CACHE:
        return _RAFT_CACHE[key]

    from torchvision.models.optical_flow import (
        Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
    )

    if variant == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
    elif variant == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights)
    else:
        raise ValueError(f"unknown RAFT variant {variant!r}")

    model = model.eval().to(device)
    _RAFT_CACHE[key] = model
    return model


def raft_flow(frame_a, frame_b, variant="small", device=None):
    """Dense flow from `frame_a` to `frame_b` via RAFT.

    Frames are (C, H, W) tensors in [0, 1]. RAFT expects [-1, 1] and spatial
    dimensions divisible by 8, so both are handled here.

    Returns
    -------
    (2, H, W) tensor of (dx, dy) displacements in pixels.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_raft(variant, device)

    a = frame_a[None].to(device) * 2 - 1
    b = frame_b[None].to(device) * 2 - 1

    _, _, H, W = a.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        a = F.pad(a, (0, pad_w, 0, pad_h), mode="replicate")
        b = F.pad(b, (0, pad_w, 0, pad_h), mode="replicate")

    with torch.no_grad():
        flow = model(a, b)[-1]

    return flow[0, :, :H, :W].cpu()


def blockmatch_flow(frame_a, frame_b, block=8, search=8):
    """Integer-shift block-matching flow. No external weights required.

    Exhaustive search over a (2*search+1)^2 window per block using SAD. Cost is
    O(H*W*search^2), which is fine at the resolutions in this project and keeps
    the pipeline runnable without pretrained models.
    """
    a = np.asarray(frame_a, dtype=np.float32)
    b = np.asarray(frame_b, dtype=np.float32)
    C, H, W = a.shape

    flow = np.zeros((2, H, W), dtype=np.float32)
    offsets = range(-search, search + 1)

    for by in range(0, H, block):
        for bx in range(0, W, block):
            y1, x1 = min(by + block, H), min(bx + block, W)
            patch = a[:, by:y1, bx:x1]

            best, best_dx, best_dy = np.inf, 0, 0
            for dy in offsets:
                sy, sy1 = by + dy, y1 + dy
                if sy < 0 or sy1 > H:
                    continue
                for dx in offsets:
                    sx, sx1 = bx + dx, x1 + dx
                    if sx < 0 or sx1 > W:
                        continue
                    cost = np.abs(b[:, sy:sy1, sx:sx1] - patch).sum()
                    if cost < best:
                        best, best_dx, best_dy = cost, dx, dy

            flow[0, by:y1, bx:x1] = best_dx
            flow[1, by:y1, bx:x1] = best_dy

    return torch.from_numpy(flow)


def identity_flow(frame_a, frame_b):
    """Zero flow. Ablation control."""
    _, H, W = np.asarray(frame_a).shape
    return torch.zeros(2, H, W)


def estimate_flow(frame_a, frame_b, method="blockmatch", **kwargs):
    """Dispatch to a flow backend: 'raft', 'blockmatch' or 'identity'."""
    if method == "raft":
        return raft_flow(frame_a, frame_b, **kwargs)
    if method == "blockmatch":
        return blockmatch_flow(frame_a, frame_b, **kwargs)
    if method == "identity":
        return identity_flow(frame_a, frame_b)
    raise ValueError(
        f"unknown flow method {method!r}; expected 'raft', 'blockmatch' or 'identity'"
    )


# --------------------------------------------------------------------------
# Warping
# --------------------------------------------------------------------------

def warp_frame(frame, flow, mode="bilinear", padding_mode="border"):
    """Backward-warp `frame` (C, H, W) by `flow` (2, H, W) using grid_sample.

    Sampling is at `p + flow(p)`, i.e. `flow` maps target coordinates into source
    coordinates -- the convention that makes `warp_frame(prev, flow_cur_to_prev)`
    an estimate of the current frame.
    """
    frame = torch.as_tensor(frame, dtype=torch.float32)
    flow = torch.as_tensor(flow, dtype=torch.float32).to(frame.device)
    C, H, W = frame.shape

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=frame.device),
        torch.arange(W, dtype=torch.float32, device=frame.device),
        indexing="ij",
    )
    src_x = xx + flow[0]
    src_y = yy + flow[1]

    # grid_sample wants normalised [-1, 1] coordinates.
    grid = torch.stack(
        [2 * src_x / max(W - 1, 1) - 1, 2 * src_y / max(H - 1, 1) - 1], dim=-1
    )[None]

    warped = F.grid_sample(
        frame[None], grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )
    return warped[0]


def forward_backward_mask(flow_fwd, flow_bwd, tol=1.5):
    """Boolean validity mask from forward-backward flow consistency.

    A pixel is valid when composing forward then backward flow returns it to
    within `tol` pixels. Invalid pixels are disoccluded or out of frame, have no
    usable proposal, and should be excluded from the importance weights.

    Returns
    -------
    (H, W) boolean tensor; True means valid.
    """
    flow_fwd = torch.as_tensor(flow_fwd, dtype=torch.float32)
    flow_bwd = torch.as_tensor(flow_bwd, dtype=torch.float32)

    warped_bwd = warp_frame(flow_bwd, flow_fwd)
    residual = (flow_fwd + warped_bwd).pow(2).sum(dim=0).sqrt()
    return residual <= tol


def warp_samples(frame_prev, flow, mask=None):
    """Warp a previous frame and return per-channel sample clouds for the
    density estimator.

    This is the bridge into `density.SequentialDensityEstimator.estimate`: it
    produces the same (2, N) pixel-value pair layout as
    `density.image_channel_to_samples`, but built from the motion-compensated
    previous frame rather than the raw current frame.

    Parameters
    ----------
    frame_prev : (C, H, W) previous frame.
    flow : (2, H, W) displacements mapping current coordinates into the previous.
    mask : optional (H, W) boolean validity mask; invalid pixels are dropped.

    Returns
    -------
    list of (2, N_valid) arrays, one per channel.
    """
    from .density import image_channel_to_samples

    warped = warp_frame(frame_prev, flow).numpy()
    C = warped.shape[0]

    clouds = []
    for ch in range(C):
        xy = image_channel_to_samples(warped[ch])
        if mask is not None:
            keep = image_channel_to_samples(
                np.asarray(mask, dtype=np.float32)
            )[0] > 0.5
            xy = xy[:, keep]
        clouds.append(xy)
    return clouds


def clip_flows(clip, method="blockmatch", with_mask=True, **kwargs):
    """Per-frame flows for a whole clip.

    Parameters
    ----------
    clip : (T, C, H, W) tensor.
    method : flow backend.

    Returns
    -------
    flows : list of length T-1; entry k is the flow taking frame k+1 into frame k
        (backward flow), which is the direction `warp_samples` needs.
    masks : list of length T-1 of validity masks, or None when `with_mask` is
        False.
    """
    clip = torch.as_tensor(clip, dtype=torch.float32)
    T = clip.shape[0]

    flows, masks = [], []
    for k in range(1, T):
        bwd = estimate_flow(clip[k], clip[k - 1], method=method, **kwargs)
        flows.append(bwd)
        if with_mask:
            fwd = estimate_flow(clip[k - 1], clip[k], method=method, **kwargs)
            masks.append(forward_backward_mask(bwd, fwd))

    return flows, (masks if with_mask else None)
