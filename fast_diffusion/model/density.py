"""Log-density estimation for Fokker-Planck initialisation.

This module owns every density estimator used to build the FP initial condition,
behind a single config-driven dispatch (`estimate_log_density`). Previously the
histogram estimator was hardcoded in `kfp.py` while the `diffusion.kde_method`
config key was read by nothing, which made the three `celeb1_kde_*` ablation
configs inert.

Estimators
----------
histogram : 2D histogram + FFT Gaussian smoothing. Cost is O(N) for binning plus
            O(M log M) for the convolution, M = n_bins^2. Because M is fixed,
            the cost is effectively *constant in image resolution*. Measured
            crossover against scipy is around 24x24; below that this estimator is
            slower (see `benchmark_density.py`).
scipy     : exact Gaussian KDE, O(N^2). Reference for accuracy; intractable
            past roughly 32x32 without subsampling.
sklearn   : tree-accelerated Gaussian KDE, O(N log N) with a large constant.

Sequential estimation
---------------------
`SequentialDensityEstimator` is the video extension. Rather than re-estimating
each frame's density from scratch, it treats the previous frame's converged
log-density as a proposal, applies a smoothed multi-resolution correction from a
cheap coarse estimate of the current frame, and triggers a full re-estimation (a
keyframe) when the proposal has drifted too far.

The trigger is `KL(target || proposal)`, not effective sample size. ESS was the
original design and was measured to be blind: on a synthetic clip with an injected
scene cut it separated the cut from within-shot frames by +/-0.0002, because the
log-ratio is nearly constant across the support even when the densities differ, so
the weight variance ESS measures stays negligible. KL separated the same cut by
14.6x, grid relative L2 by 8.3x. ESS is still computed and reported as a
diagnostic.

Two caveats to state in any write-up:

1. Dimensionality. The importance sampling operates on the two-dimensional
   density-estimation coordinates, NOT on the H*W-dimensional pixel space. IS in
   the full pixel space would degenerate.
2. Spatial blindness. Pixel-value pairs are a global statistic, insensitive to
   purely spatial rearrangement. A cut that changes layout while preserving the
   value distribution is hard to see from this density alone. Where that matters,
   build the density on temporal pairs `(I_k(p), I_{k-1}(warp(p)))` instead, which
   makes motion compensation part of the coordinate system.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates, zoom


# Density-estimation coordinate range. Pixel values are normalised to [0, 1]
# upstream in `data/Dataset.py`, so the support is the unit square.
_RANGE = [[0.0, 1.0], [0.0, 1.0]]
_LOG_FLOOR = 1e-300


def scotts_bandwidth(n, d=2):
    """Scott's rule bandwidth for `n` samples in `d` dimensions."""
    return n ** (-1.0 / (d + 4))


def image_channel_to_samples(channel):
    """Build the (2, N) pixel-value pair cloud for one image channel.

    Reproduces the sample construction that was inline in `kfp.score_samples`:
    rows and columns are stacked so each sample is a (column-value, row-value)
    pair. Accepts a 2D array of shape (H, W).
    """
    channel = np.asarray(channel)
    if channel.ndim != 2:
        raise ValueError(f"expected a 2D (H, W) channel, got shape {channel.shape}")

    H, W = channel.shape
    x_train = np.concatenate([channel[:, j] for j in range(W)])
    y_train = np.concatenate([channel[j, :] for j in range(H)])
    return np.vstack([x_train, y_train])


# --------------------------------------------------------------------------
# Grid-based (histogram) estimator
# --------------------------------------------------------------------------

def histogram_log_grid(xy, n_bins=256, bandwidth_scale=1.0):
    """Smoothed log-density on an `n_bins` x `n_bins` grid over the unit square.

    Returns
    -------
    log_grid : (n_bins, n_bins) array of log-density values.
    """
    x, y = xy[0], xy[1]
    n = xy.shape[1]

    hist, _, _ = np.histogram2d(x, y, bins=n_bins, range=_RANGE, density=True)

    sigma_bins = scotts_bandwidth(n) * n_bins * bandwidth_scale
    smoothed = ndimage.gaussian_filter(hist.astype(np.float64), sigma=sigma_bins)
    return np.log(np.clip(smoothed, _LOG_FLOOR, None))


def interpolate_grid(grid, xy):
    """Bilinearly sample a unit-square grid at coordinates `xy` of shape (2, N)."""
    n_bins = grid.shape[0]
    idx_x = np.clip(xy[0] * n_bins, 0, n_bins - 1)
    idx_y = np.clip(xy[1] * n_bins, 0, n_bins - 1)
    return map_coordinates(grid, [idx_x, idx_y], order=1, mode="nearest")


def histogram_log_density(xy, n_bins=256, bandwidth_scale=1.0):
    """Log-density at each sample via histogram + FFT Gaussian smoothing."""
    grid = histogram_log_grid(xy, n_bins=n_bins, bandwidth_scale=bandwidth_scale)
    return interpolate_grid(grid, xy)


# --------------------------------------------------------------------------
# Reference estimators
# --------------------------------------------------------------------------

def scipy_log_density(xy, max_samples=None):
    """Exact Gaussian KDE via `scipy.stats.gaussian_kde`. O(N^2).

    `max_samples` subsamples the *fitting* set only; the density is still
    evaluated at every input sample. Leave as None for the true O(N^2) cost,
    which is what the scaling benchmark measures.
    """
    from scipy.stats import gaussian_kde

    fit_xy = xy
    if max_samples is not None and xy.shape[1] > max_samples:
        sel = np.random.choice(xy.shape[1], max_samples, replace=False)
        fit_xy = xy[:, sel]

    kde = gaussian_kde(fit_xy)
    return np.log(np.clip(kde(xy), _LOG_FLOOR, None))


def sklearn_log_density(xy, bandwidth=None, rtol=1e-4):
    """Tree-accelerated Gaussian KDE via `sklearn.neighbors.KernelDensity`."""
    from sklearn.neighbors import KernelDensity

    if bandwidth is None:
        bandwidth = scotts_bandwidth(xy.shape[1])

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, rtol=rtol)
    kde.fit(xy.T)
    return kde.score_samples(xy.T)


_ESTIMATORS = {
    "histogram": histogram_log_density,
    "scipy": scipy_log_density,
    "sklearn": sklearn_log_density,
}


def estimate_log_density(xy, method="histogram", **kwargs):
    """Dispatch to a named log-density estimator.

    Parameters
    ----------
    xy : (2, N) array of samples in the unit square.
    method : one of 'histogram', 'scipy', 'sklearn'.
    """
    if method not in _ESTIMATORS:
        raise ValueError(
            f"unknown density method {method!r}; expected one of {sorted(_ESTIMATORS)}"
        )
    return _ESTIMATORS[method](xy, **kwargs)


def density_method_from_config(config):
    """Read `diffusion.kde_method` from a run config, defaulting to histogram."""
    return config.get("diffusion", {}).get("kde_method", "histogram")


# --------------------------------------------------------------------------
# Importance weights and effective sample size
# --------------------------------------------------------------------------

def normalised_log_weights(log_w):
    """Self-normalise log importance weights. Returns weights summing to 1."""
    log_w = np.asarray(log_w, dtype=np.float64)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        # Complete degeneracy: fall back to uniform so callers see ESS == 1 and
        # the ESS trigger is driven by the explicit finite-weight path instead.
        return np.full(log_w.shape, 1.0 / log_w.size)
    return w / total


def effective_sample_size(log_w, normalised=True):
    """Effective sample size of self-normalised importance weights.

    Returns ESS / N in [1/N, 1] when `normalised`, else the raw ESS. A value of 1
    means the proposal matches the target and every particle contributes equally;
    values near 1/N mean a single particle dominates.

    Retained as a diagnostic, NOT as the keyframe trigger. Measured on a synthetic
    clip with an injected scene cut, ESS separated the cut frame from within-shot
    frames by +/-0.0002 -- indistinguishable from noise -- because the log-ratio is
    nearly constant across the support even when the densities differ, so the
    weight *variance* that ESS measures stays negligible. Use
    `kl_divergence_grids`, which separated the same cut by 14.6x. See
    `SequentialDensityEstimator`.
    """
    w = normalised_log_weights(log_w)
    ess = 1.0 / np.sum(w ** 2)
    return ess / w.size if normalised else ess


def kl_divergence_grids(log_p, log_q):
    """KL(p || q) between two log-density grids over the same support.

    This is the keyframe statistic. Unlike ESS it integrates the log-ratio
    *against* the target density, so a change in density shape registers even when
    the pointwise ratio has small variance.

    Measured discrimination on a synthetic clip with a scene cut at frame 4:

        statistic          cut      max within-shot   ratio
        ---------------------------------------------------
        KL                 0.0060   0.0004            14.6x
        grid relative L2   0.3097   0.0375             8.3x
        ESS                0.9999   0.9997            (none)
    """
    p = np.exp(np.asarray(log_p, dtype=np.float64))
    q = np.exp(np.asarray(log_q, dtype=np.float64))
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("inf")
    p = p / p_sum
    q = q / q_sum
    return float(
        (p * (np.log(np.clip(p, _LOG_FLOOR, None))
              - np.log(np.clip(q, _LOG_FLOOR, None)))).sum()
    )


# --------------------------------------------------------------------------
# Sequential estimator
# --------------------------------------------------------------------------

class SequentialDensityEstimator:
    """Frame-to-frame log-density estimation by importance reweighting.

    For the first frame, and after every triggered keyframe, a full estimate is
    computed at `n_bins` resolution. For subsequent frames the retained fine grid
    acts as a proposal `q`; a cheap coarse histogram of the new frame gives an
    approximate target `p`, and the smoothed log-ratio is added back to `q` as a
    multi-resolution correction.

    This is a control-variate / multi-resolution estimator rather than textbook
    SMC: the target is approximated by the coarse histogram rather than evaluated
    exactly. That is a deliberate accuracy-for-cost trade and should be described
    as such.

    Keyframe trigger
    ----------------
    The trigger is `KL(p || q)` between the target and proposal grids, not
    effective sample size. ESS was implemented first and measured: on a synthetic
    clip with an injected scene cut it separated the cut from within-shot frames by
    +/-0.0002, i.e. not at all, because the log-ratio is nearly constant across the
    support even when the densities differ and so the weight variance ESS measures
    stays negligible. KL separated the same cut by 14.6x. ESS is still computed and
    reported as a diagnostic.

    `threshold_mode='adaptive'` (default) triggers when KL exceeds `kl_factor`
    times the running median of recent non-keyframe KLs, which self-calibrates to
    the sequence. `'absolute'` compares against `kl_threshold` directly; that needs
    per-dataset calibration and is provided mainly for ablations.

    Caveat that must be stated in any write-up: the density coordinates are
    *pixel-value pairs*, a global statistic. They are insensitive to purely spatial
    rearrangement, so a cut that preserves the value distribution while changing
    layout is hard to detect from this density alone. Where that matters, build the
    density on temporal pairs `(I_k(p), I_{k-1}(warp(p)))` instead -- see
    `flow.warp_samples`.

    Parameters
    ----------
    method : estimator used for keyframes ('histogram', 'scipy', 'sklearn').
    n_bins : fine grid resolution retained across frames.
    coarse_bins : resolution of the per-frame target approximation.
    threshold_mode : 'adaptive' or 'absolute'.
    kl_factor : multiplier on the running median KL, for adaptive mode.
    kl_threshold : absolute KL trigger, for absolute mode.
    history_window : how many recent KLs feed the running median.
    warmup : frames processed before the adaptive trigger arms.
    correction_sigma_bins : smoothing applied to the coarse log-ratio before it is
        upsampled, in coarse-grid bin units. Keep small: heavy smoothing flattens
        the ratio and was what made the original ESS trigger blind.
    """

    def __init__(
        self,
        method="histogram",
        n_bins=256,
        coarse_bins=64,
        threshold_mode="adaptive",
        kl_factor=4.0,
        kl_floor=1e-3,
        kl_threshold=2e-3,
        history_window=8,
        warmup=3,
        correction_sigma_bins=1.0,
        estimator_kwargs=None,
    ):
        if coarse_bins > n_bins:
            raise ValueError("coarse_bins must not exceed n_bins")
        if threshold_mode not in ("adaptive", "absolute"):
            raise ValueError("threshold_mode must be 'adaptive' or 'absolute'")

        self.method = method
        self.n_bins = n_bins
        self.coarse_bins = coarse_bins
        self.threshold_mode = threshold_mode
        self.kl_factor = kl_factor
        self.kl_floor = kl_floor
        self.kl_threshold = kl_threshold
        self.history_window = history_window
        self.warmup = warmup
        self.correction_sigma_bins = correction_sigma_bins
        self.estimator_kwargs = estimator_kwargs or {}

        self.reset()

    def reset(self):
        """Drop retained state. Call between independent sequences."""
        self._log_grid = None
        self._kl_history = []
        self.n_frames = 0
        self.n_keyframes = 0
        self.history = []

    # -- internals ---------------------------------------------------------

    def _full_estimate(self, xy):
        if self.method == "histogram":
            return histogram_log_grid(xy, n_bins=self.n_bins, **self.estimator_kwargs)

        # Non-grid estimators produce values at samples, not a grid. Scatter them
        # onto the fine grid so the sequential machinery downstream is uniform.
        values = estimate_log_density(xy, method=self.method, **self.estimator_kwargs)
        return self._scatter_to_grid(xy, values, self.n_bins)

    @staticmethod
    def _scatter_to_grid(xy, values, n_bins):
        """Average sample values into grid cells, filling gaps by dilation."""
        ix = np.clip((xy[0] * n_bins).astype(int), 0, n_bins - 1)
        iy = np.clip((xy[1] * n_bins).astype(int), 0, n_bins - 1)
        flat = ix * n_bins + iy

        total = np.bincount(flat, weights=values, minlength=n_bins * n_bins)
        count = np.bincount(flat, minlength=n_bins * n_bins)

        grid = np.full(n_bins * n_bins, np.nan)
        hit = count > 0
        grid[hit] = total[hit] / count[hit]
        grid = grid.reshape(n_bins, n_bins)

        # Fill empty cells with the nearest occupied value.
        if np.isnan(grid).any():
            idx = ndimage.distance_transform_edt(
                np.isnan(grid), return_distances=False, return_indices=True
            )
            grid = grid[tuple(idx)]
        return grid

    def _coarse_log_grid(self, xy):
        return histogram_log_grid(xy, n_bins=self.coarse_bins)

    def _correction(self, xy):
        """Smoothed coarse log-ratio between the new frame and the proposal.

        Returns the correction resampled to the fine grid, the log weights at `xy`,
        and the coarse target/proposal grids for the KL statistic.
        """
        target_coarse = self._coarse_log_grid(xy)

        # Downsample the retained proposal to the coarse grid for a like-for-like
        # ratio. Block-mean in log space is adequate at these resolutions.
        factor = self.n_bins // self.coarse_bins
        proposal_coarse = self._log_grid.reshape(
            self.coarse_bins, factor, self.coarse_bins, factor
        ).mean(axis=(1, 3))

        ratio = target_coarse - proposal_coarse
        if self.correction_sigma_bins > 0:
            ratio = ndimage.gaussian_filter(ratio, sigma=self.correction_sigma_bins)

        log_w = interpolate_grid(ratio, xy)
        correction_fine = zoom(ratio, factor, order=1)

        # `zoom` can be off by a row/column from integer rounding.
        if correction_fine.shape != self._log_grid.shape:
            correction_fine = np.resize(correction_fine, self._log_grid.shape)

        return correction_fine, log_w, target_coarse, proposal_coarse

    def _should_keyframe(self, kl):
        """Apply the KL trigger.

        Adaptive mode uses `max(kl_factor * running_median, kl_floor)`. The floor
        matters: on smooth motion the running median falls to ~1e-6, and a purely
        relative rule then fires on harmless fluctuations. Measured without a floor,
        a no-cut control clip produced 3 spurious keyframes out of 10 frames; with
        the floor it produces none, while a clip with an injected cut still fires
        exactly once, at the cut.
        """
        if self.threshold_mode == "absolute":
            return kl > self.kl_threshold

        if len(self._kl_history) < self.warmup:
            return False
        baseline = float(np.median(self._kl_history[-self.history_window:]))
        threshold = max(self.kl_factor * baseline, self.kl_floor)
        return kl > threshold

    # -- public API --------------------------------------------------------

    def estimate(self, xy, force_keyframe=False):
        """Estimate log-density at samples `xy` for the next frame.

        Parameters
        ----------
        xy : (2, N) samples for the current frame. To make the estimate
            motion-aware, pass temporal pairs built from the warped previous frame
            -- see `flow.warp_samples`.
        force_keyframe : bypass the trigger and re-estimate from scratch.

        Returns
        -------
        log_density : (N,) log-density at each sample.
        info : dict with `keyframe`, `kl`, `ess`, `frame`. `ess` is a diagnostic
            only; the decision is driven by `kl`.
        """
        self.n_frames += 1

        is_keyframe = force_keyframe or self._log_grid is None
        kl = float("nan")
        ess = float("nan")

        if not is_keyframe:
            correction, log_w, target_coarse, proposal_coarse = self._correction(xy)
            kl = kl_divergence_grids(target_coarse, proposal_coarse)
            ess = effective_sample_size(log_w)

            if self._should_keyframe(kl):
                is_keyframe = True
            else:
                self._log_grid = self._log_grid + correction
                self._kl_history.append(kl)

        if is_keyframe:
            self._log_grid = self._full_estimate(xy)
            self.n_keyframes += 1

        info = {
            "keyframe": bool(is_keyframe),
            "kl": kl,
            "ess": ess,
            "frame": self.n_frames - 1,
        }
        self.history.append(info)
        return interpolate_grid(self._log_grid, xy), info

    @property
    def keyframe_rate(self):
        """Fraction of processed frames that required a full re-estimation."""
        if self.n_frames == 0:
            return float("nan")
        return self.n_keyframes / self.n_frames

    def kl_trace(self):
        """Per-frame KL(target || proposal), NaN on keyframes. Plot this."""
        return np.array([h["kl"] for h in self.history])

    def ess_trace(self):
        """Per-frame normalised ESS, NaN on keyframes. Diagnostic only."""
        return np.array([h["ess"] for h in self.history])
