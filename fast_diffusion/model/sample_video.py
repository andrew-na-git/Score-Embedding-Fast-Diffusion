"""Autoregressive conditional sampling for clips.

The image sampler (`fast_diffusion/model/sample.py`) draws a whole batch from one
probability-flow ODE solve, and conditions with a single scalar
`sample.conditional_weight` blending noise against the reference:

    initial_x = (1 - w) * noise + w * z

Two things must change for video.

1. Conditioning has to be *sequential*. Frame k is generated with frames
   k-1 .. k-L already fixed, so the reference for frame k is its own predecessors
   rather than a ground-truth image. That is what makes the conditional score
   from `density.SequentialDensityEstimator` usable at generation time, and it is
   the mechanism through which temporal consistency enters the sampler rather
   than a consistency penalty in the loss.

2. The scalar weight has to become a schedule. A fixed w applies the same
   conditioning strength to the frame right after a keyframe and to a frame far
   into a drift-prone run. `weight_schedule` provides decaying, constant and
   ramped forms so this is an ablation axis instead of a magic number.

`scipy.integrate.solve_ivp` is called on a flattened state, as in the image
sampler. Clip-at-a-time solves flatten (T, C, H, W), which grows the ODE state
by a factor of T; `window` bounds that by solving a sliding window of frames and
keeping only the newly generated ones.
"""

import numpy as np
import torch
from scipy import integrate

from .kfp import diffusion_coeff


def weight_schedule(n_frames, base=0.6, kind="decay", floor=0.1, rate=0.85):
    """Per-frame conditioning weights.

    kind='constant'
        Reproduces the image pipeline's fixed scalar. Baseline.
    kind='decay'
        Geometric decay from `base` toward `floor`. Conditions strongly right
        after a keyframe and loosens as the run proceeds.
    kind='ramp'
        Linear increase from `floor` to `base`, tightening as drift accumulates.
    """
    if kind == "constant":
        return np.full(n_frames, base, dtype=np.float64)
    if kind == "decay":
        return np.maximum(base * rate ** np.arange(n_frames), floor)
    if kind == "ramp":
        return np.linspace(floor, base, n_frames)
    raise ValueError(f"unknown schedule {kind!r}; expected 'constant', 'decay' or 'ramp'")


def _ode_sample_frames(
    model, sigma, shape, initial_x, timestep_multiplier, clip_idx, frame_idx,
    n_frames, device, atol=1e-3, rtol=1e-3, eps=1e-3,
):
    """Probability-flow ODE solve over a block of frames.

    `shape` is (B, T, C, H, W). Returns the final state and the solver's function
    evaluation count.
    """
    def score_eval(state, t_scalar):
        x = torch.tensor(state, device=device, dtype=torch.float32).reshape(shape)
        labels = torch.full((shape[0],), t_scalar * timestep_multiplier,
                            device=device, dtype=torch.float32)
        with torch.no_grad():
            score = model(
                x, labels,
                clip_idx=clip_idx.to(device) if clip_idx is not None else None,
                frame_idx=frame_idx.to(device) if frame_idx is not None else None,
                n_frames=n_frames,
            )
        return score.cpu().numpy().reshape(-1).astype(np.float64)

    def ode_func(t, state):
        g = float(diffusion_coeff(torch.tensor(t), sigma))
        return -0.5 * (g ** 2) * score_eval(state, float(t))

    res = integrate.solve_ivp(
        ode_func, (1.0, eps), initial_x.reshape(-1).cpu().numpy(),
        rtol=rtol, atol=atol, method="RK45",
    )
    final = res.y[:, -1].reshape(shape)
    return torch.from_numpy(final).float(), res.nfev


def autoregressive_sample(
    model,
    config,
    reference=None,
    n_frames=None,
    window=4,
    schedule="decay",
    clip_idx=0,
    device=None,
    seed=None,
):
    """Generate a clip frame block by frame block, conditioned on what came before.

    Parameters
    ----------
    model : a `network.network3d.VideoNet`.
    config : run config. Reads `diffusion.sigma`, `data_loader.{image_size,
        channels, clip_len}`, `training.timestep_multiplier`,
        `sample.{conditional_weight, schedule_kind, schedule_floor}`.
    reference : optional (T, C, H, W) clip. When given, the first `window` frames
        are conditioned on it, which is the restoration / fitting setting. When
        None the first block is unconditional.
    n_frames : total frames to generate. Defaults to `data_loader.clip_len`.
    window : frames solved jointly per block. Bounds the ODE state size and sets
        how much temporal context each solve sees.
    schedule : 'constant', 'decay' or 'ramp' -- see `weight_schedule`.

    Returns
    -------
    clip : (T, C, H, W) generated clip.
    info : dict with `nfev` per block and the weights used.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = model.to(device).eval()

    dcfg, scfg = config["data_loader"], config.get("sample", {})
    sigma = config["diffusion"]["sigma"]
    H = W = dcfg["image_size"]
    C = dcfg.get("channels", 3)
    n_frames = n_frames or dcfg.get("clip_len", 16)
    timestep_multiplier = config["training"]["timestep_multiplier"]

    weights = weight_schedule(
        n_frames,
        base=scfg.get("conditional_weight", 0.6),
        kind=scfg.get("schedule_kind", schedule),
        floor=scfg.get("schedule_floor", 0.1),
    )

    clip = torch.zeros(n_frames, C, H, W)
    nfevs = []
    produced = 0

    while produced < n_frames:
        block = min(window, n_frames - produced)
        shape = (1, block, C, H, W)

        noise = torch.randn(shape)

        # Condition on already-generated frames, falling back to the reference for
        # the first block. This is the sequential replacement for the image
        # pipeline's fixed blend against a ground-truth image.
        if produced > 0:
            prev = clip[produced - 1][None, None].expand(1, block, C, H, W)
        elif reference is not None:
            ref = reference[:block]
            prev = ref[None].to(torch.float32)
        else:
            prev = None

        if prev is None:
            initial_x = noise * sigma
        else:
            w = torch.tensor(
                weights[produced:produced + block], dtype=torch.float32
            )[None, :, None, None, None]
            initial_x = (1 - w) * noise * sigma + w * prev

        frame_idx = torch.arange(produced, produced + block, dtype=torch.long)
        out, nfev = _ode_sample_frames(
            model, sigma, shape, initial_x.to(device), timestep_multiplier,
            torch.full((1,), clip_idx, dtype=torch.long), frame_idx, block, device,
            eps=float(config["misc"]["eps"]),
        )

        clip[produced:produced + block] = out[0]
        nfevs.append(nfev)
        produced += block

    return clip, {"nfev": nfevs, "weights": weights.tolist(), "window": window}


def clip_sample(model, config, clip_idx=0, device=None, seed=None):
    """Single-solve baseline: the whole clip in one probability-flow ODE.

    No autoregressive conditioning, so this isolates how much the sequential
    conditioning in `autoregressive_sample` actually contributes. Report both.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    model = model.to(device).eval()
    dcfg = config["data_loader"]
    sigma = config["diffusion"]["sigma"]
    H = W = dcfg["image_size"]
    C = dcfg.get("channels", 3)
    T = dcfg.get("clip_len", 16)

    shape = (1, T, C, H, W)
    initial_x = torch.randn(shape, device=device) * sigma

    out, nfev = _ode_sample_frames(
        model, sigma, shape, initial_x, config["training"]["timestep_multiplier"],
        torch.zeros(1, dtype=torch.long), torch.arange(T, dtype=torch.long), T,
        device, eps=float(config["misc"]["eps"]),
    )
    return out[0], {"nfev": nfev}
