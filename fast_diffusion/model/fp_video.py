"""Fokker-Planck solve on spatio-temporal (T, H, W) domains.

Scope: 2D dynamic video
-----------------------
`T x H x W` is time plus two further axes, and those two are pixel-*value* axes of the
density grid, not spatial ones. Nothing here is volumetric, and "the 3D operator"
below always means "the three-axis linear system". The project scope is 2D imagery in
motion; dynamic 3D / 4D representations are explicitly out of scope.

Why this module exists
----------------------
`kfp.compute_scores` assembles an (H*W) x (H*W) sparse system and calls `spsolve`,
once per diffusion timestep, per channel, per image, inside a fixed-point loop.
Adding a temporal axis makes that system (T*H*W)^2. A 16x64x64 clip is 65536
unknowns; multiplied by N timesteps, 3 channels and several fixed-point
iterations, a direct sparse factorisation is not viable.

The operator is decomposed by direction instead. Each solve is a set of
independent tridiagonal systems handled by the Thomas algorithm, O(n) per line
and trivially batched: the Python loop runs over the (small) line length while
every line is processed in parallel by numpy.

Discretisation
--------------
Writing `v = f - 0.5 g^2 s` for the drift and `D = 0.5 g^2` for the diffusivity,
with the callers' scaling `dh = dt/(2h)` and `dh2 = dt/h^2`, the unsplit operator
is `A = full_diagonal + sum_d offdiag_d`. Three stencils are available; see
`directional_coefficients`. **`upwind` is the default** because it makes `A` an
M-matrix with a diagonal-dominance margin of exactly 1 for every sigma, dt, h and
score magnitude, which removes the time-step restriction that the other stencils
impose.

Two assembly conventions matter, and getting either wrong is fatal:

Node indexing
    Row p must take all its coefficients from node p (`A[p, p+1] = a[p]`). The
    inherited code used `A[p, p+1] = a[p+1]`, mixing three nodes' drift values
    into one row; under that convention no row-wise dominance statement is
    possible and upwinding is meaningless, because the sign of `v` selecting the
    upwind direction for row p is not the sign used to build row p's
    off-diagonals. Only `stencil='legacy'` keeps the old convention.

Diagonal ownership
    The relaxation must retain the **full** operator diagonal
    (`1 + n_directions * diag_extra + boundary folds`) in every tridiagonal solve,
    with the other directions' off-diagonals moved to the right-hand side. An
    earlier version gave each factor only `1 + diag_extra`; the remainder then
    carried `2 * diag_extra`, so the iteration matrix norm approached 2 and the
    scheme diverged for anything but tiny dt. With the full diagonal the remainder
    ratio is `2 diag_extra / (1 + 2 diag_extra) < 1` unconditionally.

Two schemes
-----------
scheme='line' (default)
    Line Gauss-Seidel over directions. The fixed point is the *exact* unsplit
    solution, so there is no splitting error to defend in review.

scheme='adi'
    Single-pass approximate factorisation `(I + L_x)(I + L_y)(I + L_t)`. Cheapest,
    but carries a fixed splitting error. Keep as a cost/accuracy ablation point.

A 'strang' option was implemented, measured and removed: composing two implicit
half-solves, `(I + L/2)^-2`, does not agree with `(I + L)^-1` at second order, and
it scored strictly worse than plain ADI.

`unsplit_solve` is retained purely as the validation reference.
"""

import numpy as np
from scipy import sparse

from .kfp import diffusion_coeff


# --------------------------------------------------------------------------
# Tridiagonal kernel
# --------------------------------------------------------------------------

def thomas_batch(lower, diag, upper, rhs, check=True):
    """Solve M independent tridiagonal systems of length n.

    All arrays have shape (M, n). `lower[:, 0]` and `upper[:, -1]` are ignored.
    The loops run over n (the line length) while M (the number of lines) is
    vectorised.

    The Thomas recursion does no pivoting, so it needs the factors to stay
    non-singular. With `check=True` a breakdown raises `numpy.linalg.LinAlgError`
    with diagnostics rather than silently producing NaNs. Under `stencil='upwind'`
    the matrix is an M-matrix and breakdown cannot occur.
    """
    n = diag.shape[1]
    cp = np.empty_like(diag)
    dp = np.empty_like(rhs)

    tiny = 1e-12 * max(float(np.abs(diag).max()), 1.0)

    if check and np.any(np.abs(diag[:, 0]) < tiny):
        raise np.linalg.LinAlgError(
            "tridiagonal breakdown at j=0: leading diagonal is numerically zero"
        )

    cp[:, 0] = upper[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0]

    for j in range(1, n):
        denom = diag[:, j] - lower[:, j] * cp[:, j - 1]
        if check:
            bad = np.abs(denom) < tiny
            if np.any(bad):
                raise np.linalg.LinAlgError(
                    f"tridiagonal breakdown at j={j}: {int(bad.sum())} of "
                    f"{denom.size} lines have a vanishing pivot (min |denom| = "
                    f"{float(np.abs(denom).min()):.3e}). The matrix has lost "
                    "diagonal dominance; check `stability_margin`, or switch to "
                    "stencil='upwind', which cannot break down."
                )
        cp[:, j] = upper[:, j] / denom
        dp[:, j] = (rhs[:, j] - lower[:, j] * dp[:, j - 1]) / denom

    x = np.empty_like(rhs)
    x[:, -1] = dp[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]
    return x


# --------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------

def directional_coefficients(g, s, dh, dh2, f=0.0, stencil="upwind"):
    """Per-node off-diagonal coefficients and per-direction diagonal excess.

    Returns `(a, c, diag_extra)`: `a` is the +direction coefficient, `c` the
    -direction coefficient, and `diag_extra` this direction's diagonal
    contribution excluding the shared identity.

    stencil='upwind' (default)
        One-sided drift selected by the sign of `v`:

            a          = -D dh2 - 2 dh max(v, 0)
            c          = -D dh2 + 2 dh min(v, 0)
            diag_extra = 2 D dh2 + 2 dh |v|

        Only one of max/min is non-zero, so `|a| + |c| == diag_extra` exactly and
        the shared identity supplies a dominance margin of exactly 1 for every
        parameter choice. Both off-diagonals are non-positive with a positive
        diagonal, making the operator an M-matrix, which also gives a discrete
        maximum principle: no spurious oscillation.

        Cost: first-order accurate in space against central differencing's second
        order, introducing numerical diffusion of order |v| h / 2. Ablate against
        'central' wherever central is stable.

    stencil='central'
        Second-order central drift, sign-consistent with implicit Euler:
        `a = -(v dh + D dh2)`, `c = v dh - D dh2`, `diag_extra = 2 D dh2`.

    stencil='legacy'
        The inherited coefficients, which are not sign-consistent with implicit
        Euler (the diffusion term multiplies the antisymmetric combination and the
        drift the symmetric one -- the roles are swapped; only `a` differs). Kept
        so published numbers remain reproducible. Implies neighbour indexing.
    """
    v = np.asarray(f - 0.5 * g ** 2 * s, dtype=np.float64)
    D_dh2 = 0.5 * g ** 2 * dh2

    if stencil == "upwind":
        a = -D_dh2 - 2.0 * dh * np.maximum(v, 0.0)
        c = -D_dh2 + 2.0 * dh * np.minimum(v, 0.0)
        diag_extra = 2.0 * D_dh2 + 2.0 * dh * np.abs(v)
    elif stencil == "central":
        a = -(v * dh + D_dh2)
        c = v * dh - D_dh2
        diag_extra = np.full_like(v, 2.0 * D_dh2)
    elif stencil == "legacy":
        a = v * dh + D_dh2
        c = v * dh - D_dh2
        diag_extra = np.full_like(v, 2.0 * D_dh2)
    else:
        raise ValueError(
            f"unknown stencil {stencil!r}; expected 'upwind', 'central' or 'legacy'"
        )

    return a, c, diag_extra


def is_node_indexed(stencil):
    """Whether row p draws its coefficients from node p. False only for 'legacy'."""
    return stencil != "legacy"


# --------------------------------------------------------------------------
# Assembly helpers
# --------------------------------------------------------------------------

def _boundary_folds(a, c, axis):
    """Neumann fold contributions to the diagonal for one direction.

    A ghost node outside the domain takes the value of the node itself, so its
    coefficient moves onto the diagonal. At the + end of each line that is `a`, at
    the - end `c`.
    """
    folds = np.zeros_like(a)
    f_m = np.moveaxis(folds, axis, -1)
    a_m = np.moveaxis(a, axis, -1)
    c_m = np.moveaxis(c, axis, -1)
    f_m[..., -1] += a_m[..., -1]
    f_m[..., 0] += c_m[..., 0]
    return folds


def full_diagonal(a, c, diag_extra, axes=(0, 1, 2)):
    """The true diagonal of the unsplit operator A.

    `1 + len(axes) * diag_extra`, plus the Neumann folds from every direction.
    Every tridiagonal solve must use this, not a per-factor share -- see the
    module docstring on diagonal ownership.
    """
    diag = 1.0 + len(axes) * diag_extra
    for axis in axes:
        diag = diag + _boundary_folds(a, c, axis)
    return diag


def _offdiag_bands(a, c, axis, node_indexed=True):
    """Off-diagonal bands for one direction, flattened to (M, n).

    Boundary entries are zero because those couplings were folded into the
    diagonal by `full_diagonal`.
    """
    a_m = np.moveaxis(a, axis, -1)
    c_m = np.moveaxis(c, axis, -1)
    n = a_m.shape[-1]
    a_m = a_m.reshape(-1, n)
    c_m = c_m.reshape(-1, n)

    upper = np.zeros_like(a_m)
    lower = np.zeros_like(a_m)

    if node_indexed:
        upper[:, :-1] = a_m[:, :-1]   # A[p, p+1] = a[p]
        lower[:, 1:] = c_m[:, 1:]     # A[p, p-1] = c[p]
    else:
        upper[:, :-1] = a_m[:, 1:]    # inherited: A[p, p+1] = a[p+1]
        lower[:, 1:] = c_m[:, :-1]
    return lower, upper


def apply_offdiag(u, a, c, axis, node_indexed=True):
    """Off-diagonal action of one direction on `u`. No diagonal contribution."""
    u_m = np.moveaxis(u, axis, -1)
    shape = u_m.shape
    n = shape[-1]

    lower, upper = _offdiag_bands(a, c, axis, node_indexed)
    x = np.ascontiguousarray(u_m.reshape(-1, n))

    out = np.zeros_like(x)
    out[:, :-1] += upper[:, :-1] * x[:, 1:]
    out[:, 1:] += lower[:, 1:] * x[:, :-1]

    return np.moveaxis(out.reshape(shape), -1, axis)


def apply_A(u, a, c, diag, axes=(0, 1, 2), node_indexed=True):
    """Compute A @ u, where `diag` is the output of `full_diagonal`."""
    out = diag * u
    for axis in axes:
        out += apply_offdiag(u, a, c, axis, node_indexed)
    return out


def line_solve(rhs, a, c, diag, axis, node_indexed=True):
    """Solve the tridiagonal system formed by the full diagonal plus one direction.

    This is the relaxation kernel: `(diag + offdiag_axis) x = rhs`.
    """
    rhs_m = np.moveaxis(rhs, axis, -1)
    shape = rhs_m.shape
    n = shape[-1]

    lower, upper = _offdiag_bands(a, c, axis, node_indexed)
    d = np.ascontiguousarray(np.moveaxis(diag, axis, -1).reshape(-1, n)).copy()
    b = np.ascontiguousarray(rhs_m.reshape(-1, n))

    x = thomas_batch(lower, d, upper, b)
    return np.moveaxis(x.reshape(shape), -1, axis)


def factor_solve(rhs, a, c, diag_extra, axis, node_indexed=True):
    """Solve the approximate-factorisation factor `(I + L_axis) x = rhs`.

    Used only by `scheme='adi'`. Unlike `line_solve` this takes a per-factor share
    of the diagonal, which is what makes the product an approximation to A.
    """
    rhs_m = np.moveaxis(rhs, axis, -1)
    shape = rhs_m.shape
    n = shape[-1]

    lower, upper = _offdiag_bands(a, c, axis, node_indexed)
    folds = _boundary_folds(a, c, axis)
    d = np.ascontiguousarray(
        np.moveaxis(1.0 + diag_extra + folds, axis, -1).reshape(-1, n)
    ).copy()
    b = np.ascontiguousarray(rhs_m.reshape(-1, n))

    x = thomas_batch(lower, d, upper, b)
    return np.moveaxis(x.reshape(shape), -1, axis)


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def fp_solve(rhs, g, s, dh, dh2, f=0.0, scheme="line", axes=(0, 1, 2),
             tol=1e-8, max_sweeps=100, stencil="upwind", return_info=False):
    """One implicit step of the log-density FP operator on a (T, H, W) grid.

    scheme='line'
        Line Gauss-Seidel: sweep the directions, each time solving the full
        diagonal plus that direction's off-diagonals against a right-hand side
        that uses the newest iterate elsewhere. Converges to the exact unsplit
        solution.

    scheme='adi'
        Single-pass approximate factorisation. Cheaper, fixed splitting error.
    """
    a, c, diag_extra = directional_coefficients(g, s, dh, dh2, f, stencil=stencil)
    node_indexed = is_node_indexed(stencil)

    if scheme == "adi":
        u = rhs
        for axis in axes:
            u = factor_solve(u, a, c, diag_extra, axis, node_indexed)
        return (u, {"sweeps": 1, "residual": None}) if return_info else u

    if scheme != "line":
        raise ValueError(f"unknown scheme {scheme!r}; expected 'line' or 'adi'")

    diag = full_diagonal(a, c, diag_extra, axes)
    u = rhs / diag
    rhs_norm = np.linalg.norm(rhs)
    resid = np.inf
    sweeps = 0

    for sweeps in range(1, max_sweeps + 1):
        for axis in axes:
            others = sum(
                apply_offdiag(u, a, c, other, node_indexed)
                for other in axes if other != axis
            )
            u = line_solve(rhs - others, a, c, diag, axis, node_indexed)

        resid = np.linalg.norm(apply_A(u, a, c, diag, axes, node_indexed) - rhs)
        if rhs_norm == 0 or resid / max(rhs_norm, 1e-300) <= tol:
            break

    if return_info:
        return u, {"sweeps": sweeps, "residual": resid / max(rhs_norm, 1e-300)}
    return u


# Backwards-compatible alias; `adi_solve` was the original name.
def adi_solve(rhs, g, s, dh, dh2, f=0.0, scheme="line", **kwargs):
    """Deprecated alias for `fp_solve`."""
    return fp_solve(rhs, g, s, dh, dh2, f, scheme=scheme, **kwargs)


# --------------------------------------------------------------------------
# Unsplit reference
# --------------------------------------------------------------------------

def construct_A_3d(T, H, W, g, s, dh, dh2, f=0.0, stencil="upwind"):
    """Assemble the exact unsplit (T*H*W)^2 operator with Neumann boundaries.

    Reference only; intended for validation at small sizes.
    """
    n = T * H * W
    s = np.reshape(np.broadcast_to(np.asarray(s, dtype=np.float64), (T, H, W)), (T, H, W))
    a, c, diag_extra = directional_coefficients(g, s, dh, dh2, f, stencil=stencil)
    node_indexed = is_node_indexed(stencil)

    diag = full_diagonal(a, c, diag_extra, axes=(0, 1, 2)).reshape(-1).copy()
    a_f = a.reshape(-1)
    c_f = c.reshape(-1)

    p = np.arange(n)
    col = p % W
    row = (p // W) % H
    frame = p // (H * W)

    rows, cols, vals = [], [], []

    def couple(valid, offset, coeff):
        interior = p[valid]
        rows.append(interior)
        cols.append(interior + offset)
        vals.append(coeff[interior] if node_indexed else coeff[interior + offset])

    couple(col < W - 1, 1, a_f)
    couple(col > 0, -1, c_f)
    couple(row < H - 1, W, a_f)
    couple(row > 0, -W, c_f)
    couple(frame < T - 1, H * W, a_f)
    couple(frame > 0, -(H * W), c_f)

    rows.append(p)
    cols.append(p)
    vals.append(diag)

    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def unsplit_solve(rhs, g, s, dh, dh2, f=0.0, stencil="upwind"):
    """Exact sparse solve of the unsplit three-axis operator. Validation reference.

    "Three-axis" not "3D": the axes are (time, value, value) for a 2D video, so this
    is not a volumetric solve. Scope is 2D dynamic video throughout.
    """
    T, H, W = rhs.shape
    A = construct_A_3d(T, H, W, g, s, dh, dh2, f, stencil=stencil)
    return sparse.linalg.spsolve(A, rhs.reshape(-1)).reshape(T, H, W)


def splitting_error(rhs, g, s, dh, dh2, f=0.0, scheme="line", stencil="upwind"):
    """Relative L2 difference between the split and unsplit solves.

    For `scheme='line'` this should be at solver tolerance; it is the number that
    justifies using the split solver at all.
    """
    ref = unsplit_solve(rhs, g, s, dh, dh2, f, stencil=stencil)
    approx = fp_solve(rhs, g, s, dh, dh2, f, scheme=scheme, stencil=stencil)
    return float(np.linalg.norm(approx - ref) / np.linalg.norm(ref))


# --------------------------------------------------------------------------
# Stability diagnostics
# --------------------------------------------------------------------------

def stability_margin(g, s, dh, dh2, f=0.0, stencil="upwind", axes=(0, 1, 2)):
    """Minimum row-wise diagonal-dominance margin of the unsplit operator.

    Under `stencil='upwind'` this is identically 1 for every configuration:
    `|a| + |c| == diag_extra` by construction, so the shared identity supplies the
    margin. Verify rather than assume.

    Under 'central' or 'legacy' the margin is 1 only while the scheme is
    diffusion-dominated; in the drift-dominated regime it degrades and, for
    `f = 0`, `h = 1`, fails once `|s| > 2 + 2/(g^2 dt)`.
    """
    a, c, diag_extra = directional_coefficients(g, s, dh, dh2, f, stencil=stencil)
    diag = full_diagonal(a, c, diag_extra, axes)

    off = np.zeros_like(diag)
    node_indexed = is_node_indexed(stencil)
    for axis in axes:
        lower, upper = _offdiag_bands(a, c, axis, node_indexed)
        shape = np.moveaxis(diag, axis, -1).shape
        mass = (np.abs(lower) + np.abs(upper)).reshape(shape)
        off = off + np.moveaxis(mass, -1, axis)

    return float(np.min(diag - off))


def max_stable_score(g, dt, h=1.0):
    """Largest |s| keeping the *central* stencil diagonally dominant.

    Not a constraint under `stencil='upwind'`, which is dominant unconditionally.
    """
    return 2.0 * h + 2.0 * h ** 2 / (g ** 2 * dt)


# Empirical limit on sigma^2 * dt for the *central* stencil in three dimensions,
# measured on 3-channel 4x16x16 clips at tol=1e-4: everything at or below 0.5
# converged in 4 iterations, everything at or above 0.9 diverged or broke down.
# Upwinding removes this restriction.
STABILITY_LIMIT = 0.5


def required_timesteps(sigma, limit=STABILITY_LIMIT):
    """Minimum `num_timesteps` for the central stencil. Not needed for upwind."""
    return int(np.ceil(sigma ** 2 / limit))


def check_config_stability(config, raise_on_fail=True):
    """Pre-flight check for configs using the central or legacy stencil."""
    sigma = config["diffusion"]["sigma"]
    N = config["diffusion"]["num_timesteps"]
    sigma2_dt = sigma ** 2 / N
    needed = required_timesteps(sigma)
    stable = sigma2_dt <= STABILITY_LIMIT

    report = {
        "sigma": sigma, "N": N, "sigma2_dt": sigma2_dt,
        "stable": stable, "required_N": needed,
    }

    if not stable and raise_on_fail:
        raise ValueError(
            f"unstable configuration: sigma={sigma}, num_timesteps={N} gives "
            f"sigma^2*dt = {sigma2_dt:.3f}, above the measured limit of "
            f"{STABILITY_LIMIT} for this stencil. Use num_timesteps >= {needed}, "
            "reduce sigma, or switch to stencil='upwind', which has no such limit."
        )
    return report


# --------------------------------------------------------------------------
# Gradient on the spatio-temporal grid
# --------------------------------------------------------------------------

def log_density_gradient_3d(m, dh, axis=-1):
    """Non-wrapping finite-difference derivative of a (..., T, H, W) log-density.

    Centred in the interior, one-sided at the boundaries of the chosen axis. The
    score field keeps the shape of the clip, matching `kfp.log_density_gradient`.
    """
    grad = np.zeros_like(m)
    m_m = np.moveaxis(m, axis, -1)
    g_m = np.moveaxis(grad, axis, -1)

    g_m[..., 1:-1] = (m_m[..., 2:] - m_m[..., :-2]) / (2 * dh)
    g_m[..., 0] = (m_m[..., 1] - m_m[..., 0]) / dh
    g_m[..., -1] = (m_m[..., -1] - m_m[..., -2]) / dh
    return grad


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def compute_scores_clip(
    config,
    initial_m,
    scheme="line",
    progress=None,
):
    """Fixed-point FP solve for one clip, returning the score field.

    Parameters
    ----------
    config : run config. Reads `diffusion.{num_timesteps, sigma, dh,
        solve_tolerance, max_fp_iterations, stencil}` and `misc.eps`.
    initial_m : (C, T, H, W) initial log-density, e.g. from
        `density.SequentialDensityEstimator` reshaped per channel.
    scheme : 'line' (exact) or 'adi' (single-pass approximation).
    progress : optional callable(iteration, residual).

    Returns
    -------
    scores : (N, C, T, H, W) float32 score field.
    info : dict with `iterations`, `residuals`, `converged`, `stencil`.
    """
    N = config["diffusion"]["num_timesteps"]
    sigma = config["diffusion"]["sigma"]
    dh = config["diffusion"]["dh"]
    tol = float(config["diffusion"]["solve_tolerance"])
    max_iter = int(config["diffusion"].get("max_fp_iterations", 100))
    stencil = config["diffusion"].get("stencil", "upwind")
    # The inner line-relaxation tolerance need not match the outer fixed-point
    # tolerance. Convergence is linear at roughly 5 sweeps per decade, so the inner
    # tolerance is the dominant cost knob. Measured at 16x64x64, upwind, |s| ~ 5:
    #
    #   inner tol   sweeps   ms/solve   rel err vs exact
    #   ------------------------------------------------
    #     1e-2        10        283       8.2e-3
    #     1e-4        20        507       7.6e-5
    #     1e-6        30        875       7.2e-7
    #     1e-8        40       1041       6.9e-9
    #
    # 1e-6 keeps the inner solve two decades tighter than a typical outer tolerance
    # of 2e-8..1e-4 without paying for the last decade.
    inner_tol = float(config["diffusion"].get("inner_tolerance", 1e-6))
    max_sweeps = int(config["diffusion"].get("max_inner_sweeps", 100))
    eps = float(config["misc"]["eps"])

    # The sigma^2 dt <= 0.5 restriction comes from central differencing losing
    # diagonal dominance in the drift-dominated regime. Upwinding removes it.
    if stencil != "upwind":
        check_config_stability(config)

    initial_m = np.asarray(initial_m, dtype=np.float64)
    C, T, H, W = initial_m.shape
    dt = 1.0 / N
    dh_s, dh2_s = dt / (2 * dh), dt / (dh ** 2)

    times = np.linspace(eps, 1, N)

    m = np.zeros((N, C, T, H, W))
    m_prev = np.ones_like(m)
    scores = np.ones_like(m)

    residuals = []
    converged = False

    for it in range(1, max_iter + 1):
        for ch in range(C):
            m[0, ch] = initial_m[ch]
            for i in range(1, N):
                g = float(diffusion_coeff(times[i], sigma))
                m[i, ch] = fp_solve(
                    m[i - 1, ch], g, scores[i, ch], dh_s, dh2_s,
                    scheme=scheme, stencil=stencil,
                    tol=inner_tol, max_sweeps=max_sweeps,
                )
            # axis=-1 is the column (W) axis, matching the 2D convention.
            scores[:, ch] = log_density_gradient_3d(m[:, ch], dh, axis=-1)

        res = float(np.linalg.norm(m - m_prev) / np.linalg.norm(m_prev))
        residuals.append(res)
        if progress is not None:
            progress(it, res)
        m_prev = m.copy()

        if res <= tol:
            converged = True
            break

    return scores.astype(np.float32), {
        "iterations": len(residuals),
        "residuals": residuals,
        "converged": converged,
        "stencil": stencil,
    }
