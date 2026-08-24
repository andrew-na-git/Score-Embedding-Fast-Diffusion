"""Video evaluation harness. Replaces the deleted `evaluate_fid.py`.

Why the old script was removed
------------------------------
It computed FID over as few as one or three generated images. FID is severely
biased at small sample counts -- the estimator's bias scales roughly as 1/n and
the covariance term is not even well conditioned below a few thousand samples --
so the reported values (0.65, 1.53, 4.30) carried no information about sample
quality. Reporting them in a submission is an unforced error.

What replaces it
----------------
Distribution metrics
    `kid` : Kernel Inception Distance. Preferred over FID here because it is
    unbiased and usable at the sample counts this pipeline can realistically
    produce. Per-frame over a clip set.
    `fvd` : Frechet Video Distance on the canonical Kinetics-400 I3D features, so
    values are comparable with published FVD. The weights are not redistributable
    with the repo; fetch them with `python download_assets.py --only i3d`. If they
    are absent, `fvd` raises rather than falling back to another backbone -- FVD is
    a Frechet distance in one specific feature space, and computing it in a
    different one produces a number on a different scale that looks equally
    plausible. There is no substitute-backbone option at all, by design.

Temporal metrics -- the ones this project's contribution is actually about
    `warping_error` : flow-warped L2 between consecutive frames. The direct
    measure of whether the sequential FP scheme suppresses flicker.
    `warped_lpips` : the same comparison in LPIPS space, which correlates better
    with perceived stability than raw L2.
    `temporal_variation` : mean absolute frame difference. Report alongside the
    warping error, since a model that simply blurs motion scores well on warping
    error alone and this catches it.

Reference metrics for the fitting regime
    `psnr_per_frame`, `ssim_per_frame` : the fitting regime reconstructs known
    clips, so per-frame fidelity against the target is meaningful -- unlike FID at
    n=3.

Inpainting metrics -- the target task
    `masked_psnr`, `masked_mse`, `masked_lpips`, `seam_error`,
    `masked_warping_error`, `evaluate_inpainting`. All restricted to the hole,
    because the sampler copies observed pixels through exactly and a whole-frame
    number therefore mostly measures pixels that were never missing. See the section
    comment above them.

Sample-count guards
-------------------
`check_sample_count` refuses to return a distribution metric below a defensible
number of samples rather than returning a misleading float. That guard is the
whole point of this module; do not remove it to make a table look complete.
"""

import os

import numpy as np
import torch

MIN_KID_SAMPLES = 100
MIN_FVD_SAMPLES = 64


class InsufficientSamples(RuntimeError):
    """Raised when a distribution metric is requested below a usable sample count."""


def check_sample_count(n, minimum, metric):
    if n < minimum:
        raise InsufficientSamples(
            f"{metric} requested with n={n}, below the usable minimum of {minimum}. "
            f"Report a reference metric (PSNR/SSIM) or a temporal metric instead; "
            f"do not report {metric} at this sample count."
        )


# --------------------------------------------------------------------------
# Temporal consistency
# --------------------------------------------------------------------------

def warping_error(clip, flows, masks=None, reduction="mean"):
    """Flow-warped L2 between consecutive frames.

    For each k, warp frame k-1 into frame k's geometry and compare. Low values
    mean the sequence is temporally stable *given the motion*, which is what
    distinguishes a coherent video from a flickering one.

    Parameters
    ----------
    clip : (T, C, H, W) tensor or array in [0, 1].
    flows : list of T-1 flows, entry k mapping frame k+1's coordinates into frame
        k's -- the output layout of `flow.clip_flows`.
    masks : optional list of T-1 validity masks. Invalid (disoccluded) pixels are
        excluded, which matters: without masking, disocclusions dominate the score
        and the metric mostly measures scene motion rather than model stability.

    Returns
    -------
    float, or a (T-1,) array when `reduction` is 'none'.
    """
    from .flow import warp_frame

    clip = torch.as_tensor(clip, dtype=torch.float32)
    errs = []

    for k in range(1, clip.shape[0]):
        warped = warp_frame(clip[k - 1], flows[k - 1])
        diff = (clip[k] - warped) ** 2

        if masks is not None:
            m = torch.as_tensor(masks[k - 1], dtype=torch.float32)[None]
            denom = m.sum() * clip.shape[1]
            errs.append(float((diff * m).sum() / torch.clamp(denom, min=1.0)))
        else:
            errs.append(float(diff.mean()))

    errs = np.array(errs)
    return errs if reduction == "none" else float(errs.mean())


def warped_lpips(clip, flows, masks=None, net="alex", device=None):
    """Flow-warped LPIPS between consecutive frames.

    Requires the `lpips` package. Perceptual distance tracks visible flicker far
    better than L2, so this is the temporal number to lead with.
    """
    import lpips as lpips_lib

    from .flow import warp_frame

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips_lib.LPIPS(net=net).to(device).eval()

    clip = torch.as_tensor(clip, dtype=torch.float32)
    vals = []

    for k in range(1, clip.shape[0]):
        warped = warp_frame(clip[k - 1], flows[k - 1])
        a = (clip[k][None] * 2 - 1).to(device)
        b = (warped[None] * 2 - 1).to(device)

        if masks is not None:
            m = torch.as_tensor(masks[k - 1], dtype=torch.float32)[None, None].to(device)
            a, b = a * m, b * m

        with torch.no_grad():
            vals.append(float(loss_fn(a, b).item()))

    return float(np.mean(vals))


def temporal_variation(clip):
    """Mean absolute difference between consecutive frames.

    Companion to `warping_error`: a model that blurs away motion scores well on
    warping error while collapsing this. Always report the pair.
    """
    clip = torch.as_tensor(clip, dtype=torch.float32)
    return float((clip[1:] - clip[:-1]).abs().mean())


# --------------------------------------------------------------------------
# Reference metrics
# --------------------------------------------------------------------------

def psnr_per_frame(generated, target, data_range=1.0):
    """Per-frame PSNR in dB. Both clips are (T, C, H, W)."""
    generated = np.asarray(generated, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mse = ((generated - target) ** 2).reshape(generated.shape[0], -1).mean(axis=1)
    mse = np.maximum(mse, 1e-20)
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim_per_frame(generated, target, data_range=1.0):
    """Per-frame SSIM. Requires scikit-image."""
    from skimage.metrics import structural_similarity

    generated = np.asarray(generated, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    return np.array([
        structural_similarity(t, g, channel_axis=0, data_range=data_range)
        for g, t in zip(generated, target)
    ])


# --------------------------------------------------------------------------
# Distribution metrics
# --------------------------------------------------------------------------

def _to_uint8(clips):
    """Stack clips to (N, C, H, W) uint8, as torchmetrics image metrics expect."""
    arr = torch.as_tensor(np.asarray(clips), dtype=torch.float32)
    if arr.dim() == 5:
        arr = arr.reshape(-1, *arr.shape[2:])
    arr = arr.clamp(0, 1) * 255
    return arr.to(torch.uint8)


def kid(real_clips, fake_clips, subset_size=50, device=None):
    """Per-frame Kernel Inception Distance over clip sets.

    Unbiased, so unlike FID it remains meaningful at moderate sample counts.
    `subset_size` must not exceed the number of frames available.

    Returns
    -------
    dict with `kid_mean`, `kid_std`, `n_real`, `n_fake`, `subset_size`.
    """
    from torchmetrics.image.kid import KernelInceptionDistance

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    real, fake = _to_uint8(real_clips), _to_uint8(fake_clips)

    check_sample_count(min(len(real), len(fake)), MIN_KID_SAMPLES, "KID")
    subset_size = min(subset_size, len(real), len(fake))

    metric = KernelInceptionDistance(subset_size=subset_size).to(device)
    metric.update(real.to(device), real=True)
    metric.update(fake.to(device), real=False)
    mean, std = metric.compute()

    return {
        "kid_mean": float(mean), "kid_std": float(std),
        "n_real": len(real), "n_fake": len(fake), "subset_size": subset_size,
    }


DEFAULT_I3D_PATH = os.path.join("assets", "i3d_torchscript.pt")

# The I3D export refuses fewer than 9 frames: its temporal pooling stack reduces the
# time axis below 1. A T=8 clip fails inside the TorchScript interpreter with an
# unhelpful message, so it is checked up front.
MIN_I3D_FRAMES = 9

_I3D_CACHE = {}


def load_i3d(weights=None, device=None):
    """Load the Kinetics-400 I3D TorchScript module used to define FVD.

    Refuses to proceed if the weights are absent rather than substituting another
    network. FVD is a Frechet distance in a *specific* feature space; computing it
    with a different extractor yields a number that is not on the same scale as any
    published FVD, and silently doing so is how incomparable figures end up in
    tables.

    Fetch the weights with `python download_assets.py --only i3d`.
    """
    path = weights or DEFAULT_I3D_PATH
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (os.path.abspath(path), str(device))

    if key in _I3D_CACHE:
        return _I3D_CACHE[key]

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"I3D weights not found at {path!r}. FVD is only comparable with "
            "published values when computed with the Kinetics-400 I3D features, so "
            "no substitute backbone is used automatically. Run\n"
            "    python download_assets.py --only i3d\n"
            "or pass weights=<path>. To compute a self-consistent, explicitly "
            "non-comparable value instead, call fvd(..., backbone='r3d_18')."
        )

    model = torch.jit.load(path).eval().to(device)
    _I3D_CACHE[key] = model
    return model


def i3d_features(clips, weights=None, device=None, batch_size=8):
    """400-d Kinetics logits for clips of shape (N, T, C, H, W) in [0, 1].

    The module's own `rescale` and `resize` flags are used rather than
    reimplementing its preprocessing: `rescale=True` expects [0, 255] and applies
    `x/255*2-1`, and `resize=True` bilinearly resizes to 224x224. Matching the
    reference implementation's preprocessing exactly matters as much as matching the
    weights, since FVD is sensitive to both.

    Features are taken pre-softmax (`return_features=True`), which is the layer FVD
    is defined on.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_i3d(weights, device)

    x = torch.as_tensor(np.asarray(clips), dtype=torch.float32)
    if x.dim() != 5:
        raise ValueError(f"expected (N, T, C, H, W), got shape {tuple(x.shape)}")
    if x.shape[1] < MIN_I3D_FRAMES:
        raise ValueError(
            f"I3D needs at least {MIN_I3D_FRAMES} frames per clip; got {x.shape[1]}. "
            "Its temporal pooling stack collapses below that and the TorchScript "
            "module raises an opaque error."
        )

    # (N, T, C, H, W) -> (N, C, T, H, W), and [0, 1] -> [0, 255] for rescale=True.
    # `.contiguous()` is required, not defensive: the module's resize path calls
    # `.view()` on the permuted tensor, which fails outright on a non-contiguous
    # input with a message about spanning two contiguous subspaces.
    x = (x.permute(0, 2, 1, 3, 4) * 255.0).contiguous()

    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            feats = model(x[i:i + batch_size].to(device), rescale=True, resize=True,
                          return_features=True)
            out.append(feats.cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fvd(real_clips, fake_clips, device=None, weights=None):
    """Frechet Video Distance over clip sets of shape (N, T, C, H, W) in [0, 1].

    Computed on the canonical Kinetics-400 I3D features, so values are comparable
    with published FVD. There is deliberately no alternative backbone: FVD is a
    Frechet distance in one specific feature space, and the same clips scored through
    a different extractor give a number on a different scale that looks just as
    plausible. An earlier revision offered torchvision's `r3d_18` as a substitute;
    it was removed, because a non-comparable FVD is not a weaker result but a
    different quantity wearing the same name.

    If the weights are missing this raises. Fetch them with
    `python download_assets.py --only i3d`.

    Returns a dict, not a float: the backbone, feature dimension and sample count
    have to travel with the value or it cannot be interpreted.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    real = torch.as_tensor(np.asarray(real_clips), dtype=torch.float32)
    fake = torch.as_tensor(np.asarray(fake_clips), dtype=torch.float32)

    check_sample_count(min(len(real), len(fake)), MIN_FVD_SAMPLES, "FVD")

    fr = i3d_features(real, weights, device)
    ff = i3d_features(fake, weights, device)

    return {
        "fvd": _frechet_distance(fr, ff),
        "backbone": "i3d",
        "feature_dim": int(fr.shape[1]),
        "comparable_with_published": True,
        "n_real": len(real),
        "n_fake": len(fake),
        "frames_per_clip": int(real.shape[1]),
    }


def _frechet_distance(x, y, eps=1e-6):
    """Frechet distance between two Gaussian fits.

    Two numerical details matter and are handled explicitly rather than hoped away:

    * `linalg.sqrtm` dropped its `disp` keyword in recent SciPy, so it is called
      positionally-compatibly and the result unpacked defensively.
    * The product of two sample covariances is frequently near-singular at these
      sample counts, and `sqrtm` then returns a matrix with a small imaginary part or
      fails outright. A conditioning offset is applied to the diagonals before
      retrying, which is what the reference FID implementations do; discarding a
      large imaginary component silently would corrupt the value instead.
    """
    from scipy import linalg

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    def _sqrtm(a):
        out = linalg.sqrtm(a)
        # Older SciPy returned (sqrt, errest) when disp=False; newer returns the
        # array. Accept either shape of return value.
        return out[0] if isinstance(out, tuple) else out

    covmean = _sqrtm(sigma_x @ sigma_y)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_x.shape[0]) * eps
        covmean = _sqrtm((sigma_x + offset) @ (sigma_y + offset))

    if np.iscomplexobj(covmean):
        imag_scale = np.abs(covmean.imag).max()
        real_scale = max(np.abs(covmean.real).max(), 1e-30)
        if imag_scale / real_scale > 1e-3:
            raise RuntimeError(
                f"matrix square root has a large imaginary component "
                f"({imag_scale:.3e} vs real {real_scale:.3e}); the covariance "
                "estimate is too ill-conditioned for a trustworthy Frechet distance. "
                "Increase the sample count."
            )
        covmean = covmean.real

    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(sigma_x) + np.trace(sigma_y)
                 - 2 * np.trace(covmean))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def evaluate_clip(generated, target=None, flow_method="blockmatch", with_lpips=False):
    """Full single-clip report: temporal metrics, plus reference metrics if given.

    Deliberately excludes distribution metrics -- a single clip cannot support
    them. Use `kid` / `fvd` over a clip set.
    """
    from .flow import clip_flows

    result = {"temporal_variation": temporal_variation(generated)}

    flows, masks = clip_flows(generated, method=flow_method, with_mask=True)
    result["warping_error"] = warping_error(generated, flows, masks)
    if with_lpips:
        result["warped_lpips"] = warped_lpips(generated, flows, masks)

    if target is not None:
        psnr = psnr_per_frame(generated, target)
        result["psnr_mean"] = float(psnr.mean())
        result["psnr_per_frame"] = psnr.tolist()
        try:
            ssim = ssim_per_frame(generated, target)
            result["ssim_mean"] = float(ssim.mean())
            result["ssim_per_frame"] = ssim.tolist()
        except ImportError:
            result["ssim_mean"] = None

    return result


# --------------------------------------------------------------------------
# Inpainting metrics
#
# Everything here restricts the comparison to the hole. That is not a refinement,
# it is the difference between a meaningful number and a meaningless one: the
# sampler copies observed pixels through exactly, so a whole-frame PSNR is dominated
# by pixels that were never missing and rises without bound as coverage falls. A
# whole-frame PSNR of 40 dB on a 5% mask says nothing about the fill.
#
# `hole` is (T, 1, H, W) with 1 = unknown, matching `inpaint.make_mask`. Do not pass
# the flow validity masks used by `warping_error` here; they mean the opposite thing.
# --------------------------------------------------------------------------

def _as_hole(hole, clip):
    h = torch.as_tensor(np.asarray(hole), dtype=torch.float32)
    if h.dim() == 3:
        h = h.unsqueeze(1)
    if h.shape[0] != clip.shape[0]:
        raise ValueError(
            f"hole mask has {h.shape[0]} frames but the clip has {clip.shape[0]}"
        )
    return h


def masked_mse(generated, target, hole, per_frame=False):
    """Mean squared error over unknown pixels only."""
    generated = torch.as_tensor(generated, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    h = _as_hole(hole, generated)

    sq = (generated - target) ** 2 * h
    if per_frame:
        num = sq.flatten(1).sum(1)
        den = (h.expand_as(sq)).flatten(1).sum(1)
        return (num / torch.clamp(den, min=1.0)).numpy()
    den = h.expand_as(sq).sum()
    if float(den) == 0:
        raise ValueError("the hole mask is empty; masked metrics are undefined")
    return float(sq.sum() / den)


def masked_psnr(generated, target, hole, data_range=1.0, per_frame=False):
    """PSNR over unknown pixels only. The headline inpainting fidelity number.

    Returns inf for an exact fill, and inf for frames with no unknown pixels when
    `per_frame=True`. Both are correct but neither can be averaged, so drop the
    non-finite entries before reducing -- or use `per_frame=False`, which pools over
    the clip and sidesteps the issue.
    """
    mse = masked_mse(generated, target, hole, per_frame=per_frame)
    if per_frame:
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(data_range ** 2 / mse)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def masked_lpips(generated, target, hole, net="alex", device=None, dilate=4):
    """LPIPS on a crop-free composite that isolates the fill.

    LPIPS has spatial support, so it cannot be restricted to a hole pixel-exactly.
    What it can do -- and what is reported here -- is compare the *generated* clip
    against a composite that is identical everywhere except inside a dilated hole.
    The dilation is deliberate: seams sit on the boundary, and a metric evaluated on
    the exact hole is blind to them.
    """
    import lpips as lpips_lib

    from .inpaint import dilate_mask

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips_lib.LPIPS(net=net).to(device).eval()

    generated = torch.as_tensor(generated, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    h = _as_hole(hole, generated)
    region = torch.as_tensor(
        dilate_mask(h.numpy(), dilate), dtype=torch.float32
    )

    vals = []
    for k in range(generated.shape[0]):
        if float(region[k].sum()) == 0:
            continue
        # Both inputs share their observed pixels, so the distance is driven by the
        # filled region and its immediate surround.
        a = (generated[k][None] * 2 - 1).to(device)
        b = (target[k][None] * 2 - 1).to(device)
        m = region[k][None].to(device)
        with torch.no_grad():
            vals.append(float(loss_fn(a * m, b * m)))
    return float(np.mean(vals)) if vals else float("nan")


def seam_error(generated, target, hole, band=2):
    """Discontinuity across the hole boundary.

    A fill can be individually plausible on every frame and still betray itself at
    the edge, where it fails to join the observed pixels. This compares a thin band
    just inside the hole against the same band in the target, which is where that
    failure shows up; it is not captured by a hole-wide MSE that averages the
    boundary away with the interior.
    """
    from .inpaint import dilate_mask

    generated = torch.as_tensor(generated, dtype=torch.float32)
    h = _as_hole(hole, generated)
    hn = h.numpy()

    # Band inside the hole: hole minus its erosion. Erosion via dilating the
    # complement, which avoids adding a second morphological primitive.
    eroded = 1.0 - dilate_mask(1.0 - hn, band)
    inner = np.clip(hn - eroded, 0.0, 1.0)
    if inner.sum() == 0:
        return float("nan")
    return masked_mse(generated, target, inner)


def masked_warping_error(clip, flows, hole, flow_masks=None):
    """Warping error restricted to pixels that were filled.

    The temporal question for inpainting is whether the *fill* is coherent frame to
    frame, not whether the clip as a whole is -- the observed majority of the clip is
    copied from a real video and is trivially coherent, which would swamp the signal.
    A pixel is counted at frame k when it is unknown in either frame k or k-1, since
    a fill that disagrees with adjoining observed pixels is exactly the artifact of
    interest.
    """
    clip = torch.as_tensor(clip, dtype=torch.float32)
    h = _as_hole(hole, clip)

    combined = []
    for k in range(1, clip.shape[0]):
        m = torch.clamp(h[k] + h[k - 1], 0.0, 1.0)[0]
        if flow_masks is not None:
            m = m * torch.as_tensor(flow_masks[k - 1], dtype=torch.float32)
        combined.append(m)

    if all(float(m.sum()) == 0 for m in combined):
        return float("nan")
    return warping_error(clip, flows, masks=combined)


def evaluate_inpainting(generated, target, hole, flow_method="blockmatch",
                        with_lpips=False, data_range=1.0):
    """Full inpainting report for one clip.

    Always reports mask coverage, and always reports the whole-frame PSNR next to
    the masked one so the gap between them is visible rather than hidden -- the
    whole-frame value is the number that looks good and means little.
    """
    from .flow import clip_flows
    from .inpaint import mask_coverage

    generated = torch.as_tensor(np.asarray(generated), dtype=torch.float32)
    target = torch.as_tensor(np.asarray(target), dtype=torch.float32)
    h = _as_hole(hole, generated)

    result = {"coverage": mask_coverage(h.numpy())}

    # Observed pixels must be preserved exactly; if they are not, the sampler is
    # broken and every other number below is untrustworthy.
    leak = float(((generated - target) * (1.0 - h)).abs().max())
    result["observed_max_deviation"] = leak
    if leak > 1e-5:
        result["warning"] = (
            f"observed pixels deviate by up to {leak:.2e}; the masked metrics assume "
            "they are copied through exactly"
        )

    result["masked_psnr"] = masked_psnr(generated, target, h, data_range)
    result["masked_mse"] = masked_mse(generated, target, h)
    result["seam_error"] = seam_error(generated, target, h)
    result["whole_frame_psnr"] = float(
        psnr_per_frame(generated, target, data_range).mean()
    )
    result["temporal_variation"] = temporal_variation(generated)

    flows, flow_masks = clip_flows(generated, method=flow_method, with_mask=True)
    result["warping_error"] = warping_error(generated, flows, flow_masks)
    result["masked_warping_error"] = masked_warping_error(
        generated, flows, h, flow_masks
    )

    if with_lpips:
        result["masked_lpips"] = masked_lpips(generated, target, h)

    return result
