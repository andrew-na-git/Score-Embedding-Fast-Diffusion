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
    `fvd` : Frechet Video Distance. Reported with its backbone named and its
    sample count stated, because FVD is backbone-sensitive and comparisons across
    papers using different feature extractors are not meaningful.

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


def fvd(real_clips, fake_clips, device=None, backbone="r3d_18"):
    """Frechet Video Distance over clip sets of shape (N, T, C, H, W).

    The canonical implementation uses an I3D network trained on Kinetics-400,
    whose weights are not distributed with torchvision. This uses torchvision's
    `r3d_18` (Kinetics-400) instead, which is a defensible substitute but *not*
    numerically comparable to I3D-based FVD in other papers.

    Always report the backbone and the sample count next to the number. Do not
    compare this value against published FVD figures computed with I3D.
    """
    from torchvision.models.video import R3D_18_Weights, r3d_18

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    real = torch.as_tensor(np.asarray(real_clips), dtype=torch.float32)
    fake = torch.as_tensor(np.asarray(fake_clips), dtype=torch.float32)

    check_sample_count(min(len(real), len(fake)), MIN_FVD_SAMPLES, "FVD")

    if backbone != "r3d_18":
        raise NotImplementedError(
            f"backbone {backbone!r} not wired up; only 'r3d_18' is available "
            "without external I3D weights"
        )

    weights = R3D_18_Weights.DEFAULT
    net = r3d_18(weights=weights).to(device).eval()
    net.fc = torch.nn.Identity()

    mean = torch.tensor(weights.transforms.keywords["mean"]).view(1, 3, 1, 1, 1)
    std = torch.tensor(weights.transforms.keywords["std"]).view(1, 3, 1, 1, 1)

    def features(clips):
        # (N, T, C, H, W) -> (N, C, T, H, W), normalised.
        x = clips.permute(0, 2, 1, 3, 4)
        x = (x - mean) / std
        out = []
        with torch.no_grad():
            for i in range(0, x.shape[0], 8):
                out.append(net(x[i:i + 8].to(device)).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    fr, ff = features(real), features(fake)
    return {
        "fvd": _frechet_distance(fr, ff),
        "backbone": backbone,
        "n_real": len(real),
        "n_fake": len(fake),
    }


def _frechet_distance(x, y):
    """Frechet distance between two Gaussian fits."""
    from scipy import linalg

    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(sigma_x) + np.trace(sigma_y) - 2 * np.trace(covmean))


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
