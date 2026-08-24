"""GPU (torch) backend for the spatio-temporal Fokker-Planck solve.

Why a separate module
---------------------
`fp_video.py` is a tested numpy implementation and stays the reference. This
module mirrors its solver in torch so the work runs on GPU. Rather than abstract
over both array libraries -- which invites silent divergence -- the two are kept
side by side and pinned together by `tests_equivalence`, which asserts the torch
path reproduces the numpy path to floating-point tolerance on identical inputs.

Why this is worth doing
-----------------------
The relaxation kernel is a batched tridiagonal solve. Its sequential loop runs over
the line length (16-256), while the batch dimension is the number of independent
lines: 1024 at 16x64x64 and 2048 at 16x128x128 per direction, times the C colour
channels, which this backend batches together. That is an ideal GPU shape -- a short
serial recursion over a very wide batch.

The measured payoff is strongly size-dependent (`benchmark_solver.py`; RTX 2000 Ada
Laptop, medians of 3). End-to-end per-clip score precompute:

    8x32x32      GPU  12.7 s   CPU   4.6 s   0.36x  -- GPU is a loss
    16x64x64     GPU  31.1 s   CPU  42.2 s   1.36x  -- marginal
    16x128x128   GPU  55.7 s   CPU 298.5 s   5.36x  -- clear win

Below roughly 64x64 the CPU path is faster and should be used. The GPU port is what
makes the 128x128 campaign affordable, which is exactly what it was written for.

What the bottleneck actually is
------------------------------
This solver is kernel-launch and bandwidth bound, not arithmetic bound. Two
measurements establish that, and both shaped the implementation:

* fp64 costs 0.85-1.33x fp32 on device across all sizes. Ada fp64 arithmetic runs
  at 1/64 the fp32 rate, so near-parity means arithmetic is a negligible share of
  the runtime.
* Testing a Thomas pivot on device forces a device-to-host synchronisation. Doing
  it inside the recursion, as the numpy version does harmlessly, cost ~2.5x overall
  (1701 -> 677 ms at 8x32x32). It is off by default here; see `thomas_batch_torch`.

The remaining headroom is in launch count: a sweep issues on the order of a thousand
small kernels. CUDA graph capture of the sweep body is the obvious next step and is
not implemented.

Precision
---------
`dtype` defaults to float32. It is never slower than float64 by more than noise, and
its deviation from the float64 result is ~9e-8 -- an order of magnitude below the
1e-6 inner tolerance the solver is run at, so it does not limit accuracy in practice.
float64 is available at negligible extra cost if a tighter tolerance is ever needed;
`fp_solve_torch` warns if the requested tolerance is unreachable in float32.

The stencil, node-indexing and full-diagonal conventions are identical to
`fp_video`; see that module's docstring for the derivations and for why each of
those choices is load-bearing.
"""

import warnings

import numpy as np
import torch

from .kfp import diffusion_coeff

_FLOAT32_EPS = float(np.finfo(np.float32).eps)


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Tridiagonal kernel
# --------------------------------------------------------------------------

def thomas_batch_torch(lower, diag, upper, rhs, check=False):
    """Solve M independent tridiagonal systems of length n, batched on device.

    All tensors are (M, n). `lower[:, 0]` and `upper[:, -1]` are ignored. The
    recursion is serial in n and fully parallel in M.

    `check` defaults to False here, unlike the numpy version. Testing a pivot
    on device requires a device-to-host synchronisation, and doing that inside
    the recursion serialises the whole pipeline: it measured as the dominant cost
    of the GPU solve (see `fp_torch` notes in PLAN.md). Under `stencil='upwind'`
    the matrix is a diagonally dominant M-matrix, so no pivot can vanish and the
    check is provably redundant. Callers using 'central' or 'legacy' should either
    pass `check=True` and accept the cost, or validate the result's finiteness
    afterwards -- `fp_solve_torch` does the latter, once per solve.
    """
    n = diag.shape[1]
    cp = torch.empty_like(diag)
    dp = torch.empty_like(rhs)

    cp[:, 0] = upper[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0]

    if check:
        tiny = 1e-12 * max(float(diag.abs().max()), 1.0)

    for j in range(1, n):
        denom = diag[:, j] - lower[:, j] * cp[:, j - 1]
        if check:
            bad = denom.abs() < tiny
            if bool(bad.any()):
                raise torch.linalg.LinAlgError(
                    f"tridiagonal breakdown at j={j}: {int(bad.sum())} of "
                    f"{denom.numel()} lines have a vanishing pivot. Switch to "
                    "stencil='upwind', which cannot break down."
                )
        cp[:, j] = upper[:, j] / denom
        dp[:, j] = (rhs[:, j] - lower[:, j] * dp[:, j - 1]) / denom

    x = torch.empty_like(rhs)
    x[:, -1] = dp[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]
    return x


# --------------------------------------------------------------------------
# Coefficients (mirrors fp_video.directional_coefficients)
# --------------------------------------------------------------------------

def directional_coefficients_torch(g, s, dh, dh2, f=0.0, stencil="upwind"):
    """Per-node off-diagonal coefficients and per-direction diagonal excess."""
    v = f - 0.5 * g ** 2 * s
    D_dh2 = 0.5 * g ** 2 * dh2

    if stencil == "upwind":
        a = -D_dh2 - 2.0 * dh * torch.clamp(v, min=0.0)
        c = -D_dh2 + 2.0 * dh * torch.clamp(v, max=0.0)
        diag_extra = 2.0 * D_dh2 + 2.0 * dh * v.abs()
    elif stencil == "central":
        a = -(v * dh + D_dh2)
        c = v * dh - D_dh2
        diag_extra = torch.full_like(v, 2.0 * D_dh2)
    elif stencil == "legacy":
        a = v * dh + D_dh2
        c = v * dh - D_dh2
        diag_extra = torch.full_like(v, 2.0 * D_dh2)
    else:
        raise ValueError(
            f"unknown stencil {stencil!r}; expected 'upwind', 'central' or 'legacy'"
        )

    return a, c, diag_extra


def _boundary_folds_torch(a, c, axis):
    """Neumann fold contributions to the diagonal for one direction."""
    folds = torch.zeros_like(a)
    f_m = folds.movedim(axis, -1)
    a_m = a.movedim(axis, -1)
    c_m = c.movedim(axis, -1)
    f_m[..., -1] += a_m[..., -1]
    f_m[..., 0] += c_m[..., 0]
    return folds


def full_diagonal_torch(a, c, diag_extra, axes=(0, 1, 2)):
    """True diagonal of the unsplit operator A."""
    diag = 1.0 + len(axes) * diag_extra
    for axis in axes:
        diag = diag + _boundary_folds_torch(a, c, axis)
    return diag


def _offdiag_bands_torch(a, c, axis, node_indexed=True):
    """Off-diagonal bands for one direction, flattened to (M, n)."""
    a_m = a.movedim(axis, -1)
    n = a_m.shape[-1]
    a_m = a_m.reshape(-1, n)
    c_m = c.movedim(axis, -1).reshape(-1, n)

    upper = torch.zeros_like(a_m)
    lower = torch.zeros_like(a_m)

    if node_indexed:
        upper[:, :-1] = a_m[:, :-1]
        lower[:, 1:] = c_m[:, 1:]
    else:
        upper[:, :-1] = a_m[:, 1:]
        lower[:, 1:] = c_m[:, :-1]
    return lower, upper


def apply_offdiag_torch(u, a, c, axis, node_indexed=True):
    """Off-diagonal action of one direction on `u`."""
    u_m = u.movedim(axis, -1)
    shape = u_m.shape
    n = shape[-1]

    lower, upper = _offdiag_bands_torch(a, c, axis, node_indexed)
    x = u_m.reshape(-1, n)

    out = torch.zeros_like(x)
    out[:, :-1] += upper[:, :-1] * x[:, 1:]
    out[:, 1:] += lower[:, 1:] * x[:, :-1]

    return out.reshape(shape).movedim(-1, axis)


def apply_A_torch(u, a, c, diag, axes=(0, 1, 2), node_indexed=True):
    """Compute A @ u, with `diag` from `full_diagonal_torch`."""
    out = diag * u
    for axis in axes:
        out = out + apply_offdiag_torch(u, a, c, axis, node_indexed)
    return out


def line_solve_torch(rhs, a, c, diag, axis, node_indexed=True, check=False):
    """Solve (full_diagonal + offdiag_axis) x = rhs along one direction."""
    rhs_m = rhs.movedim(axis, -1)
    shape = rhs_m.shape
    n = shape[-1]

    lower, upper = _offdiag_bands_torch(a, c, axis, node_indexed)
    d = diag.movedim(axis, -1).reshape(-1, n).clone()
    b = rhs_m.reshape(-1, n)

    x = thomas_batch_torch(lower, d, upper, b, check=check)
    return x.reshape(shape).movedim(-1, axis)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

def fp_solve_torch(rhs, g, s, dh, dh2, f=0.0, axes=None, tol=1e-6,
                   max_sweeps=100, stencil="upwind", check=False,
                   residual_every=4, return_info=False, warn_precision=True):
    """Line Gauss-Seidel FP step on a (T, H, W) tensor, or a batch of them.

    `axes` defaults to the trailing three axes, so a (C, T, H, W) input is solved
    as C independent volumes in one call. That is not a convenience: batching the
    colour channels into the tridiagonal batch dimension does the same arithmetic
    with a third of the kernel launches, and this solver is launch-bound.

    Converges to the exact unsplit solution. `scheme='adi'` is not mirrored here:
    it exists in `fp_video` only as a cost/accuracy ablation point and is not worth
    a second implementation to keep in sync.

    `residual_every` controls how often the convergence test runs. Each test needs
    a device-to-host copy, so testing every sweep costs real time; the default
    checks every 4th sweep and may therefore overshoot `tol` by up to 3 sweeps.
    """
    if axes is None:
        axes = tuple(range(rhs.ndim - 3, rhs.ndim))

    if warn_precision and rhs.dtype == torch.float32 and tol < 2 * _FLOAT32_EPS:
        warnings.warn(
            f"inner tolerance {tol:.1e} is at float32 eps ({_FLOAT32_EPS:.1e}); "
            "the relaxation will stall before reaching it. Use tol >= 5e-7 in "
            "float32, or dtype=torch.float64 (measured at 0.85-1.33x the fp32 "
            "cost, since this solver is launch- not arithmetic-bound).",
            RuntimeWarning,
            stacklevel=2,
        )

    a, c, diag_extra = directional_coefficients_torch(g, s, dh, dh2, f, stencil=stencil)
    node_indexed = stencil != "legacy"

    diag = full_diagonal_torch(a, c, diag_extra, axes)
    u = rhs / diag
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    resid = float("inf")
    sweeps = 0

    for sweeps in range(1, max_sweeps + 1):
        for axis in axes:
            others = None
            for other in axes:
                if other == axis:
                    continue
                term = apply_offdiag_torch(u, a, c, other, node_indexed)
                others = term if others is None else others + term
            u = line_solve_torch(rhs - others, a, c, diag, axis, node_indexed, check)

        if rhs_norm == 0:
            resid = 0.0
            break
        if sweeps % residual_every == 0 or sweeps == max_sweeps:
            resid = float(
                torch.linalg.vector_norm(
                    apply_A_torch(u, a, c, diag, axes, node_indexed) - rhs
                )
            ) / rhs_norm
            if resid <= tol:
                break

    if not np.isfinite(resid):
        raise torch.linalg.LinAlgError(
            f"FP solve produced a non-finite residual with stencil={stencil!r}. "
            "'central' and 'legacy' lose diagonal dominance once |s| grows; use "
            "stencil='upwind'."
        )

    if return_info:
        return u, {"sweeps": sweeps, "residual": resid}
    return u


def log_density_gradient_3d_torch(m, dh, axis=-1):
    """Non-wrapping finite-difference derivative along `axis`."""
    grad = torch.zeros_like(m)
    m_m = m.movedim(axis, -1)
    g_m = grad.movedim(axis, -1)

    g_m[..., 1:-1] = (m_m[..., 2:] - m_m[..., :-2]) / (2 * dh)
    g_m[..., 0] = (m_m[..., 1] - m_m[..., 0]) / dh
    g_m[..., -1] = (m_m[..., -1] - m_m[..., -2]) / dh
    return grad


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def compute_scores_clip_torch(
    config,
    initial_m,
    warm_start_scores=None,
    device=None,
    dtype=torch.float32,
    progress=None,
):
    """GPU equivalent of `fp_video.compute_scores_clip`.

    Accepts and returns numpy arrays so it is a drop-in replacement; tensors are
    created on `device` internally.

    Returns
    -------
    scores : (N, C, T, H, W) float32 numpy array.
    info : dict with `iterations`, `residuals`, `converged`, `stencil`, `device`,
        `dtype`, `inner_sweeps` (per-solve sweep counts for the final iteration).
    """
    device = device or default_device()

    N = config["diffusion"]["num_timesteps"]
    sigma = config["diffusion"]["sigma"]
    dh = config["diffusion"]["dh"]
    tol = float(config["diffusion"]["solve_tolerance"])
    max_iter = int(config["diffusion"].get("max_fp_iterations", 100))
    stencil = config["diffusion"].get("stencil", "upwind")
    inner_tol = float(config["diffusion"].get("inner_tolerance", 1e-6))
    max_sweeps = int(config["diffusion"].get("max_inner_sweeps", 100))
    eps = float(config["misc"]["eps"])

    initial_m_t = torch.as_tensor(np.asarray(initial_m), dtype=dtype, device=device)
    C, T, H, W = initial_m_t.shape
    dt = 1.0 / N
    dh_s, dh2_s = dt / (2 * dh), dt / (dh ** 2)

    times = np.linspace(eps, 1, N)

    m = torch.zeros((N, C, T, H, W), dtype=dtype, device=device)
    m_prev = torch.ones_like(m)
    if warm_start_scores is None:
        scores = torch.ones_like(m)
    else:
        scores = torch.as_tensor(
            np.asarray(warm_start_scores), dtype=dtype, device=device
        ).clone()

    residuals = []
    converged = False
    inner_sweeps = []

    for it in range(1, max_iter + 1):
        inner_sweeps = []
        m[0] = initial_m_t
        for i in range(1, N):
            g = float(diffusion_coeff(times[i], sigma))
            # All C channels solved in one call: they are independent, and batching
            # them cuts the kernel-launch count -- which is what this solver is
            # bound by -- by a factor of C at identical arithmetic cost.
            m[i], sinfo = fp_solve_torch(
                m[i - 1], g, scores[i], dh_s, dh2_s,
                axes=(1, 2, 3), tol=inner_tol, max_sweeps=max_sweeps,
                stencil=stencil, return_info=True,
                warn_precision=(it == 1 and i == 1),
            )
            inner_sweeps.append(sinfo["sweeps"])
        scores = log_density_gradient_3d_torch(m, dh, axis=-1)

        res = float(
            torch.linalg.vector_norm(m - m_prev) / torch.linalg.vector_norm(m_prev)
        )
        residuals.append(res)
        if progress is not None:
            progress(it, res)
        m_prev = m.clone()

        if res <= tol:
            converged = True
            break

    return scores.to(torch.float32).cpu().numpy(), {
        "iterations": len(residuals),
        "residuals": residuals,
        "converged": converged,
        "stencil": stencil,
        "device": str(device),
        "dtype": str(dtype),
        "inner_sweeps": inner_sweeps,
    }


# --------------------------------------------------------------------------
# Equivalence check against the numpy reference
# --------------------------------------------------------------------------

def tests_equivalence(seed=0, shape=(6, 16, 16), sigma=5.0, N=20, tol=1e-8,
                      device=None, verbose=True):
    """Assert the torch solver matches `fp_video` on identical inputs.

    Run in float64 on CPU for a like-for-like comparison; the point is to catch
    divergence between the two implementations, not to measure GPU speed.

    Each stencil is tested at a score magnitude where it is actually stable.
    'central' loses diagonal dominance for |s| >~ 2 and then diverges -- and it
    diverges *identically* in both backends, whose difference is then exactly zero
    over an overflowed norm. Comparing the two at large |s| therefore reports a
    perfect match while checking nothing, so 'central' is tested in its stable
    range and every result is asserted finite.

    Returns
    -------
    dict of relative differences per checked quantity.
    """
    from . import fp_video

    device = device or torch.device("cpu")

    dt = 1.0 / N
    dh_s, dh2_s = dt / 2, dt
    g = float(sigma)

    # 'upwind' is unconditionally stable, so it is exercised at a demanding score
    # magnitude; 'central' is held inside its dominance limit.
    score_scale = {"upwind": 5.0, "central": 0.5}

    out = {}

    for stencil in ("upwind", "central"):
        rng = np.random.default_rng(seed)
        rhs = rng.random(shape)
        s = rng.standard_normal(shape) * score_scale[stencil]

        rhs_t = torch.as_tensor(rhs, dtype=torch.float64, device=device)
        s_t = torch.as_tensor(s, dtype=torch.float64, device=device)

        a_n, c_n, d_n = fp_video.directional_coefficients(
            g, s, dh_s, dh2_s, stencil=stencil
        )
        a_t, c_t, d_t = directional_coefficients_torch(
            g, s_t, dh_s, dh2_s, stencil=stencil
        )
        out[f"coeff_a_{stencil}"] = float(
            np.abs(a_t.cpu().numpy() - a_n).max() / max(np.abs(a_n).max(), 1e-30)
        )
        out[f"coeff_diag_{stencil}"] = float(
            np.abs(d_t.cpu().numpy() - d_n).max() / max(np.abs(d_n).max(), 1e-30)
        )

        diag_n = fp_video.full_diagonal(a_n, c_n, d_n)
        diag_t = full_diagonal_torch(a_t, c_t, d_t)
        out[f"full_diag_{stencil}"] = float(
            np.abs(diag_t.cpu().numpy() - diag_n).max() / np.abs(diag_n).max()
        )

        Au_n = fp_video.apply_A(rhs, a_n, c_n, diag_n)
        Au_t = apply_A_torch(rhs_t, a_t, c_t, diag_t)
        out[f"apply_A_{stencil}"] = float(
            np.abs(Au_t.cpu().numpy() - Au_n).max() / np.abs(Au_n).max()
        )

        u_n = fp_video.fp_solve(rhs, g, s, dh_s, dh2_s, scheme="line",
                                stencil=stencil, tol=tol)
        u_t = fp_solve_torch(rhs_t, g, s_t, dh_s, dh2_s, stencil=stencil, tol=tol,
                             residual_every=1, warn_precision=False)
        u_t_np = u_t.cpu().numpy()

        if not (np.isfinite(u_n).all() and np.isfinite(u_t_np).all()):
            raise AssertionError(
                f"stencil={stencil!r} diverged during the equivalence check; the "
                "comparison would be vacuous. Lower score_scale for this stencil."
            )
        out[f"fp_solve_{stencil}"] = float(
            np.linalg.norm(u_t_np - u_n) / np.linalg.norm(u_n)
        )

    rng = np.random.default_rng(seed)
    rhs = rng.random(shape)
    rhs_t = torch.as_tensor(rhs, dtype=torch.float64, device=device)
    grad_n = fp_video.log_density_gradient_3d(rhs, 1.0, axis=-1)
    grad_t = log_density_gradient_3d_torch(rhs_t, 1.0, axis=-1)
    out["gradient"] = float(
        np.abs(grad_t.cpu().numpy() - grad_n).max() / np.abs(grad_n).max()
    )

    if verbose:
        width = max(len(k) for k in out)
        for k, v in out.items():
            status = "OK" if v < 1e-10 else "MISMATCH"
            print(f"  {k:<{width}}  rel diff {v:.3e}  {status}")

    worst = max(out.values())
    if worst >= 1e-10:
        raise AssertionError(
            f"torch backend diverges from the numpy reference (worst {worst:.3e})"
        )
    return out
