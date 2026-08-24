"""Per-instance video inpainting: masks and masked conditional sampling.

This is the paper's target task. It is chosen because it plays to what the method
actually is -- a *per-instance* score precomputation, fitted to one clip -- rather
than against it. Unconditional video synthesis would require the network to model a
distribution over clips, which this pipeline does not do; inpainting asks it to
reconstruct held-out regions of a clip whose score field it has already solved for,
which is exactly the per-instance fitting regime. It also has an unambiguous
reference: the removed pixels are known, so error is measurable rather than a matter
of perceptual opinion.

Why a separate integrator
-------------------------
Diffusion inpainting works by *projecting* the state at every integration step:
outside the mask the sample is overwritten with the reference, diffused to the
current noise level, so the score network only ever fills in the hole given correct
surroundings. `scipy.integrate.solve_ivp`, used by `sample.py` and
`sample_video.py`, owns its internal state and adaptive step sequence, and offers no
hook to modify the state between steps -- projecting inside the right-hand side is
not the same operation and leaves the known region free to drift. So the masked path
uses an explicit fixed-step integrator, which makes the projection exact and the cost
directly comparable across runs (`n_steps` function evaluations, no adaptivity).

This is a deliberate divergence from the unmasked sampler, not an oversight; the two
are compared on cost in `benchmark_solver.py`-style reporting, and `n_steps` should
be set so the unmasked baseline is not unfairly advantaged.
"""

import numpy as np
import torch

from .kfp import diffusion_coeff, marginal_prob_std

# --------------------------------------------------------------------------
# Masks
#
# Convention throughout: mask is 1 where pixels are UNKNOWN (to be generated) and
# 0 where they are known/observed. Shape (T, 1, H, W) so it broadcasts over
# channels, and float32 so it can be used directly in blends.
# --------------------------------------------------------------------------


def _empty(T, H, W):
    return np.zeros((T, 1, H, W), dtype=np.float32)


def static_box_mask(T, H, W, size=0.25, centre=(0.5, 0.5)):
    """A box in the same place on every frame.

    The easy case, and the one an image method handles without any temporal
    reasoning: a purely spatial inpainting problem repeated T times. Include it as
    a baseline so the harder masks below are shown to be harder.
    """
    mask = _empty(T, H, W)
    bh, bw = max(1, int(size * H)), max(1, int(size * W))
    cy, cx = int(centre[0] * H), int(centre[1] * W)
    y0, x0 = np.clip(cy - bh // 2, 0, H - bh), np.clip(cx - bw // 2, 0, W - bw)
    mask[:, :, y0:y0 + bh, x0:x0 + bw] = 1.0
    return mask


def moving_box_mask(T, H, W, size=0.25, start=(0.5, 0.25), velocity=(0.0, 0.5)):
    """A box that translates across the clip. The object-removal case.

    `velocity` is in fractions of the frame traversed over the whole clip, so
    (0, 0.5) sweeps half the width left to right. This is the mask that actually
    tests temporal reasoning: a given pixel is unknown on some frames and known on
    others, so the correct fill is recoverable from other frames -- but only by a
    model that propagates information along time.
    """
    mask = _empty(T, H, W)
    bh, bw = max(1, int(size * H)), max(1, int(size * W))
    for t in range(T):
        frac = t / max(T - 1, 1)
        cy = (start[0] + velocity[0] * frac) * H
        cx = (start[1] + velocity[1] * frac) * W
        y0 = int(np.clip(cy - bh / 2, 0, H - bh))
        x0 = int(np.clip(cx - bw / 2, 0, W - bw))
        mask[t, :, y0:y0 + bh, x0:x0 + bw] = 1.0
    return mask


def stroke_mask(T, H, W, n_strokes=2, width=0.06, seed=0, drift=0.15,
                n_points=4, step=0.12):
    """Free-form strokes that drift over time. Stands in for hand-drawn removal.

    Strokes are random walks, so the mask is neither convex nor axis-aligned and
    cannot be exploited by a model that has memorised rectangles.

    `step` bounds each walk segment to a fraction of the frame. Drawing control
    points uniformly over the frame instead produces strokes that traverse it and a
    coverage near 30%, which is not comparable with the ~6% of the box masks -- and
    inpainting error is meaningless across different coverages. The defaults here are
    tuned to sit in the same range as the boxes; check with `mask_coverage`.
    """
    rng = np.random.default_rng(seed)
    mask = _empty(T, H, W)
    half = max(1, int(width * min(H, W) / 2))

    for _ in range(n_strokes):
        # Random walk from a random start, so segments stay short and local.
        y0, x0 = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        offs = rng.uniform(-step, step, (n_points - 1, 2))
        pts = np.cumsum(np.vstack([[y0, x0], offs]), axis=0)
        ys = np.clip(pts[:, 0], 0.05, 0.95) * H
        xs = np.clip(pts[:, 1], 0.05, 0.95) * W
        dy = rng.uniform(-drift, drift) * H
        dx = rng.uniform(-drift, drift) * W

        for t in range(T):
            frac = t / max(T - 1, 1)
            pts_y = np.clip(ys + dy * frac, 0, H - 1)
            pts_x = np.clip(xs + dx * frac, 0, W - 1)
            # Straight segments between successive control points.
            for k in range(len(pts_y) - 1):
                steps = int(max(abs(pts_y[k + 1] - pts_y[k]),
                                abs(pts_x[k + 1] - pts_x[k]))) + 1
                yy = np.linspace(pts_y[k], pts_y[k + 1], steps).astype(int)
                xx = np.linspace(pts_x[k], pts_x[k + 1], steps).astype(int)
                for y, x in zip(yy, xx):
                    mask[t, :, max(0, y - half):y + half + 1,
                         max(0, x - half):x + half + 1] = 1.0
    return mask


def dilate_mask(mask, radius=1):
    """Grow the unknown region by `radius` pixels, spatially only.

    Real removal masks are drawn loosely, and a mask that clips the object leaves
    a rim of its pixels in the "known" region, which the sampler will then faithfully
    preserve. Dilating spatially but not temporally keeps the mask's motion honest.
    """
    if radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
        out = np.maximum.reduce([
            padded[:, :, 2:, 1:-1], padded[:, :, :-2, 1:-1],
            padded[:, :, 1:-1, 2:], padded[:, :, 1:-1, :-2],
            out,
        ])
    return out


_MASKS = {
    "static_box": static_box_mask,
    "moving_box": moving_box_mask,
    "stroke": stroke_mask,
}


def make_mask(kind, T, H, W, dilate=0, **kwargs):
    """Build a mask by name. Returns (T, 1, H, W) float32, 1 = unknown."""
    if kind not in _MASKS:
        raise ValueError(
            f"unknown mask {kind!r}; expected one of {sorted(_MASKS)}"
        )
    return dilate_mask(_MASKS[kind](T, H, W, **kwargs), dilate)


def mask_coverage(mask):
    """Fraction of unknown pixels overall and the per-frame min/max.

    Report this alongside any inpainting metric. A number quoted without it is
    uninterpretable, since error inside a 5% hole and a 40% hole are not comparable.
    """
    per_frame = mask.reshape(mask.shape[0], -1).mean(axis=1)
    return {
        "overall": float(mask.mean()),
        "per_frame_min": float(per_frame.min()),
        "per_frame_max": float(per_frame.max()),
    }


# --------------------------------------------------------------------------
# Masked sampling
# --------------------------------------------------------------------------

def diffuse_known(reference, t, sigma, noise=None):
    """Diffuse the reference to noise level `t` of the VE forward process.

    The known region must be supplied to the network at the *same* noise level as
    the region being generated; handing it a clean reference at t=1 would put the
    input off the manifold the network was trained on.
    """
    std = float(marginal_prob_std(torch.tensor(float(t)), sigma))
    if noise is None:
        return reference, std
    return reference + std * noise, std


def pf_ode_inpaint(
    model,
    reference,
    mask,
    sigma,
    timestep_multiplier,
    clip_idx=None,
    frame_idx=None,
    n_steps=200,
    eps=1e-3,
    device="cpu",
    known_noise="fixed",
    clamp_output=True,
    seed=None,
):
    """Probability-flow ODE from t=1 to t=eps with the known region projected in.

    Parameters
    ----------
    reference : (B, T, C, H, W) observed clip. Values inside the mask are never read.
    mask : (T, 1, H, W) or broadcastable; 1 = unknown.
    n_steps : fixed integration steps. Doubles as the function-evaluation count, so
        it is the cost figure to compare against the adaptive sampler's `nfev`.
    known_noise : how the known region is diffused at each step.
        'fixed'  -- one noise draw, rescaled by std(t) each step. Keeps the whole
                    trajectory deterministic given the seed, consistent with the
                    probability-flow ODE being a deterministic map.
        'fresh'  -- redraw every step (as in RePaint). Injects stochasticity, which
                    can help the fill agree with its surroundings but makes the
                    sampler no longer a deterministic ODE.
        'none'   -- project the clean reference. Off-manifold; included only to show
                    why it is wrong.
    clamp_output : clamp the filled region to the reference's own value range. The
        output is an image, and an unclamped sample can leave that range by a wide
        margin -- an underfitted network produced fills of order 10 against data in
        [0, 1] here. Clamping is standard practice for diffusion samplers and is not
        a way of improving the numbers, but it does change them, so it is exposed as
        a flag and must be applied identically to every method being compared. The
        range is taken from the reference rather than assumed to be [0, 1], because
        `VideoDataset` normalises per clip.

    Returns
    -------
    clip : (B, T, C, H, W) with known pixels equal to `reference` exactly.
    info : dict with `n_steps`, `nfev`, `known_noise`, `clamped`.
    """
    if seed is not None:
        torch.manual_seed(seed)

    reference = reference.to(device)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
    while mask_t.dim() < reference.dim():
        mask_t = mask_t.unsqueeze(0)

    known = 1.0 - mask_t
    fixed_noise = torch.randn_like(reference) if known_noise == "fixed" else None

    # The valid output range, measured on the pixels that are actually observed.
    if clamp_output:
        obs = reference[known.expand_as(reference) > 0]
        lo = float(obs.min()) if obs.numel() else 0.0
        hi = float(obs.max()) if obs.numel() else 1.0

    # Start from the forward process at t=1: reference where known, pure noise where
    # unknown. Starting the unknown region from the reference would leak it.
    ts = np.linspace(1.0, eps, n_steps + 1)
    std1 = float(marginal_prob_std(torch.tensor(1.0), sigma))
    x = known * (reference + std1 * torch.randn_like(reference)) \
        + mask_t * torch.randn_like(reference) * std1

    nfev = 0
    for i in range(n_steps):
        t, t_next = float(ts[i]), float(ts[i + 1])
        dt = t_next - t  # negative: integrating backwards in time

        # Project before evaluating the score, so the network conditions on a
        # correct neighbourhood rather than on its own earlier guesses.
        if known_noise == "none":
            x = known * reference + mask_t * x
        else:
            noise = fixed_noise if known_noise == "fixed" else torch.randn_like(x)
            noised, _ = diffuse_known(reference, t, sigma, noise)
            x = known * noised + mask_t * x

        labels = torch.full((x.shape[0],), t * timestep_multiplier,
                            device=device, dtype=torch.float32)
        with torch.no_grad():
            score = model(
                x, labels,
                clip_idx=clip_idx.to(device) if clip_idx is not None else None,
                frame_idx=frame_idx.to(device) if frame_idx is not None else None,
                n_frames=x.shape[1],
            )
        nfev += 1

        g = float(diffusion_coeff(torch.tensor(t), sigma))
        x = x + (-0.5 * g ** 2 * score) * dt

    if clamp_output:
        x = x.clamp(lo, hi)

    # Final projection with the clean reference: the known pixels are observed data
    # and there is no reason to return a noisy version of them. This happens after
    # the clamp so observed pixels are never altered by it.
    x = known * reference + mask_t * x
    return x.cpu(), {"n_steps": n_steps, "nfev": nfev, "known_noise": known_noise,
                     "clamped": bool(clamp_output)}


def autoregressive_inpaint(
    model,
    config,
    reference,
    mask,
    window=4,
    overlap=1,
    clip_idx=0,
    n_steps=200,
    device=None,
    known_noise="fixed",
    clamp_output=True,
    seed=None,
):
    """Inpaint a clip in overlapping blocks of frames.

    Blocking bounds memory the same way `sample_video.autoregressive_sample` does.
    `overlap` frames are re-solved at each block boundary and then *treated as known*
    for the next block: without that, each block is filled independently and the
    seams show up as a temporal discontinuity exactly at the block period, which is
    the obvious failure mode of naive blocked video inpainting.

    Returns
    -------
    clip : (T, C, H, W) with observed pixels preserved.
    info : dict with per-block `nfev`, the block ranges, and mask coverage.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    reference = torch.as_tensor(np.asarray(reference), dtype=torch.float32)
    T = reference.shape[0]
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape[0] != T:
        raise ValueError(
            f"mask has {mask.shape[0]} frames but the reference has {T}"
        )

    sigma = config["diffusion"]["sigma"]
    timestep_multiplier = config["training"]["timestep_multiplier"]
    eps = float(config["misc"]["eps"])

    out = reference.clone()
    # Frames already solved are promoted to known for subsequent blocks.
    working_mask = mask.copy()

    nfevs, blocks = [], []
    start = 0
    step = max(1, window - overlap)

    while start < T:
        stop = min(start + window, T)
        block_ref = out[start:stop][None].to(device)
        block_mask = working_mask[start:stop]

        if block_mask.sum() == 0:
            start += step
            continue

        frame_idx = torch.arange(start, stop, dtype=torch.long)
        filled, info = pf_ode_inpaint(
            model, block_ref, block_mask, sigma, timestep_multiplier,
            clip_idx=torch.full((1,), clip_idx, dtype=torch.long),
            frame_idx=frame_idx, n_steps=n_steps, eps=eps, device=device,
            known_noise=known_noise, clamp_output=clamp_output,
            seed=None if seed is None else seed + start,
        )

        out[start:stop] = filled[0]
        # Everything solved outside the last `overlap` frames is now known, so the
        # next block conditions on it instead of re-generating it.
        settled = max(start, stop - overlap) if stop < T else stop
        working_mask[start:settled] = 0.0

        nfevs.append(info["nfev"])
        blocks.append((start, stop))
        if stop >= T:
            break
        start += step

    return out, {
        "nfev": nfevs,
        "blocks": blocks,
        "window": window,
        "overlap": overlap,
        "n_steps": n_steps,
        "clamped": bool(clamp_output),
        "coverage": mask_coverage(mask),
    }


def tests_masked_sampling(verbose=True):
    """Sanity checks that do not need a trained network.

    Two properties must hold for any correct masked sampler, and both are cheap to
    check with an arbitrary network:

    1. An all-zero mask (nothing unknown) must return the reference *exactly*. If it
       does not, the projection is wrong and every reported inpainting error is
       contaminated by damage to pixels that were never missing.
    2. Observed pixels must be preserved bit-exactly for any mask, since they are
       copied, not generated.
    """
    torch.manual_seed(0)
    T, C, H, W = 4, 3, 8, 8

    class _Dummy(torch.nn.Module):
        """Returns a fixed nonzero score, so any leakage shows up as drift."""

        def forward(self, x, labels, clip_idx=None, frame_idx=None, n_frames=None):
            return torch.ones_like(x) * 0.5

    model = _Dummy()
    ref = torch.randn(1, T, C, H, W)
    results = {}

    empty = np.zeros((T, 1, H, W), dtype=np.float32)
    out, _ = pf_ode_inpaint(model, ref, empty, 5.0, 1.0, n_steps=10, seed=0)
    results["empty_mask_exact"] = float((out - ref).abs().max())

    mask = make_mask("moving_box", T, H, W, size=0.5)
    out, _ = pf_ode_inpaint(model, ref, mask, 5.0, 1.0, n_steps=10, seed=0)
    known = torch.as_tensor(1.0 - mask)[None]
    results["known_preserved"] = float(((out - ref) * known).abs().max())
    results["unknown_changed"] = float(
        ((out - ref) * torch.as_tensor(mask)[None]).abs().max()
    )

    cov = mask_coverage(mask)
    results["coverage"] = cov["overall"]

    if verbose:
        for k, v in results.items():
            print(f"  {k:<20} {v:.3e}")

    if results["empty_mask_exact"] != 0.0:
        raise AssertionError(
            f"an all-zero mask altered the reference by {results['empty_mask_exact']:.3e}; "
            "the projection is not exact"
        )
    if results["known_preserved"] != 0.0:
        raise AssertionError(
            f"observed pixels were altered by {results['known_preserved']:.3e}"
        )
    if results["unknown_changed"] == 0.0:
        raise AssertionError(
            "the masked region was left untouched -- the sampler did nothing"
        )
    return results
