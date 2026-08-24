import numpy as np
from scipy import sparse
import torch
from tqdm import tqdm
import functools
import time
import csv
import os

from .density import (
    density_method_from_config,
    estimate_log_density,
    image_channel_to_samples,
)


def construct_A(H, W, dh, dh2, f, df, g, s, bc="neumann", stencil="upwind"):
    """Assemble the sparse log-density Fokker-Planck operator on an H x W grid.

    Stencil
    -------
    stencil='upwind' (default)
        One-sided drift, giving an M-matrix whose diagonal dominance margin is
        exactly 1 for every sigma, dt and score magnitude. See
        `fp_video.directional_coefficients` for the derivation; the coefficients
        here are the 2D specialisation of the same formulas.

    stencil='central'
        Second-order central drift, sign-consistent with implicit Euler.

    stencil='legacy'
        The inherited coefficients. Note that these are *not* sign-consistent with
        an implicit Euler step: expanding a row gives

            m_p + 2 D dh2 m_p + D dh2 (m_{p+1} - m_{p-1}) + v dh (m_{p+1} + m_{p-1})

        whereas implicit Euler on dm/dt = v dm/dx + D d2m/dx2 requires

            m_p + 2 D dh2 m_p - D dh2 (m_{p+1} + m_{p-1}) - v dh (m_{p+1} - m_{p-1})

        so the diffusion term multiplies the antisymmetric combination and the
        drift the symmetric one -- the two roles are swapped. Only the `a`
        (+direction) coefficient differs; `c` already matches. Retained solely to
        reproduce published numbers.

    Boundary handling
    -----------------
    bc='neumann' (default)
        Zero normal derivative. Ghost nodes outside the domain take the value of
        the node itself, so their coefficients fold onto the diagonal. Crucially,
        nodes at the end of a row are NOT coupled to the start of the next row.

    bc='legacy'
        Reproduces the original assembly, in which `sparse.diags(a[1:], 1)`
        coupled index p to p+1 unconditionally -- wrapping the end of each row
        into the start of the next -- and the row stride was `H` rather than `W`.
        Retained so published numbers remain reproducible and so the boundary
        treatment can be ablated. Do not use for new work: with a temporal axis
        added, the wrapping leaks density across frame boundaries.

    Parameters
    ----------
    dh, dh2 : the pre-scaled step factors (callers pass dt/(2*h) and dt/h**2).
    f, df   : drift and its divergence. Zero for the variance-exploding SDE.
    g       : diffusion coefficient at the current timestep (scalar).
    s       : (H*W,) score field from the previous fixed-point iterate.
    """
    n = H * W
    v = f - 0.5 * g ** 2 * s
    D_dh2 = 0.5 * g ** 2 * dh2

    if stencil == "upwind":
        a = -D_dh2 - 2.0 * dh * np.maximum(v, 0.0)
        c = -D_dh2 + 2.0 * dh * np.minimum(v, 0.0)
        diag_dir = 2.0 * D_dh2 + 2.0 * dh * np.abs(v)
    elif stencil == "central":
        a = -(v * dh + D_dh2)
        c = v * dh - D_dh2
        diag_dir = np.broadcast_to(np.asarray(2.0 * D_dh2, dtype=np.float64), (n,))
    elif stencil == "legacy":
        a = v * dh + D_dh2
        c = v * dh - D_dh2
        diag_dir = np.broadcast_to(np.asarray(2.0 * D_dh2, dtype=np.float64), (n,))
    else:
        raise ValueError(
            f"unknown stencil {stencil!r}; expected 'upwind', 'central' or 'legacy'"
        )

    a = np.broadcast_to(np.asarray(a, dtype=np.float64), (n,))
    c = np.broadcast_to(np.asarray(c, dtype=np.float64), (n,))
    diag_dir = np.asarray(diag_dir, dtype=np.float64)

    if bc == "legacy":
        # Two directions' worth of diagonal, matching the original scalar `b`.
        Ddiag = sparse.diags(2.0 * diag_dir, 0, format="csr")
        Dupper = sparse.diags(a[1:], 1, format="csr")
        Dlower = sparse.diags(c[:-1], -1, format="csr")
        B_upper_block = sparse.diags(a[H:], H, format="csr")
        C_lower_block = sparse.diags(c[:-H], -H, format="csr")
        return (
            sparse.eye(n, format="csr")
            + Ddiag + Dupper + Dlower + B_upper_block + C_lower_block
            + df * sparse.eye(n, format="csr") * dh
        )

    if bc != "neumann":
        raise ValueError(f"unknown bc {bc!r}; expected 'neumann' or 'legacy'")

    p = np.arange(n)
    col = p % W
    row = p // W

    # Two directions, plus the shared identity and the divergence term.
    diag = 1.0 + 2.0 * diag_dir + df * dh
    node_indexed = stencil != "legacy"
    rows, cols, vals = [], [], []

    def couple(valid, offset, coeff):
        """Link p -> p+offset where valid, folding the rest onto the diagonal.

        Node-indexed for 'upwind' and 'central': row p's coefficient comes from
        node p, so `|a[p]| + |c[p]|` can be compared against row p's own diagonal.
        The inherited convention took it from the neighbour, which mixes three
        nodes' drift values into one row and makes upwinding meaningless.
        """
        interior = p[valid]
        rows.append(interior)
        cols.append(interior + offset)
        vals.append(coeff[interior] if node_indexed else coeff[interior + offset])
        boundary = p[~valid]
        np.add.at(diag, boundary, coeff[boundary])

    couple(col < W - 1, 1, a)    # +x
    couple(col > 0, -1, c)       # -x
    couple(row < H - 1, W, a)    # +y  (row stride is W, not H)
    couple(row > 0, -W, c)       # -y

    rows.append(p)
    cols.append(p)
    vals.append(diag)

    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def construct_B(H, W, m_prev):
    return m_prev


def log_density_gradient(m, H, W, dh, bc="neumann", out=None):
    """Finite-difference derivative of the log-density along the flattened axis.

    The score field is stored with the same shape as the image, so a single
    derivative component is retained per pixel -- the convention inherited from
    the original scheme. What is fixed here is the boundary treatment.

    bc='neumann'
        Centred differences in the interior; one-sided differences at the two
        ends of each row, with no coupling across row boundaries.

    bc='legacy'
        Centred differences straight through the flattened array (wrapping across
        rows) with a hard zero substituted outside the two global endpoints.

    Parameters
    ----------
    m : (..., H*W) array of log-density values.
    out : optional destination array of the same shape.
    """
    m = np.asarray(m)
    if out is None:
        out = np.zeros_like(m, dtype=m.dtype)

    if bc == "legacy":
        out[..., 1:-1] = (m[..., 2:] - m[..., :-2]) / (2 * dh)
        out[..., 0] = (m[..., 1] - 0) / (2 * dh)
        out[..., -1] = (0 - m[..., -2]) / (2 * dh)
        return out

    if bc != "neumann":
        raise ValueError(f"unknown bc {bc!r}; expected 'neumann' or 'legacy'")

    grid = m.reshape(m.shape[:-1] + (H, W))
    grad = out.reshape(out.shape[:-1] + (H, W))

    grad[..., 1:-1] = (grid[..., 2:] - grid[..., :-2]) / (2 * dh)
    # One-sided at each row end: no wrap into the neighbouring row.
    grad[..., 0] = (grid[..., 1] - grid[..., 0]) / dh
    grad[..., -1] = (grid[..., -1] - grid[..., -2]) / dh
    return out


def score_samples(dataset, seed=None, method="histogram", estimator_kwargs=None):
    """Initial log-density estimates for the FP solver, per image and channel.

    `method` selects the estimator from `density.py` ('histogram', 'scipy',
    'sklearn'). This is what makes the `diffusion.kde_method` config key live;
    previously the histogram estimator was hardcoded here and the three
    `celeb1_kde_*` ablation configs had no effect.
    """
    if seed is not None:
        np.random.seed(seed)

    estimator_kwargs = estimator_kwargs or {}

    init_m_batch = []
    for data in dataset:
        init_m = []
        for ch in range(dataset.channels):
            xy = image_channel_to_samples(np.asarray(data[ch]))
            init_m.append(estimate_log_density(xy, method=method, **estimator_kwargs))
        init_m_batch.append(np.vstack(init_m)[None])
    return np.vstack(init_m_batch)


def compute_scores(config, dataset, save_folder=None):
    seed = config.get("data_loader", {}).get("seed", None)
    if seed is not None:
        np.random.seed(seed)

    n_data = dataset.n_data
    channels = dataset.channels
    H = W = dataset.image_res

    N = config["diffusion"]["num_timesteps"]
    sigma = config["diffusion"]["sigma"]
    eps = float(config["misc"]["eps"])
    dt = 1 / N
    tol = float(config["diffusion"]["solve_tolerance"])
    dh = config["diffusion"]["dh"]
    bc = config["diffusion"].get("boundary", "neumann")
    stencil = config["diffusion"].get("stencil", "upwind")
    method = density_method_from_config(config)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    m = np.zeros((n_data, N, channels, H * W), dtype=np.float32)
    m_prev = np.ones((n_data, N, channels, H * W), dtype=np.float32)
    scores = np.ones((n_data, N, channels, H * W), dtype=np.float32)
    dm = np.zeros_like(scores, dtype=np.float32)

    t_kde_start = time.time()
    initial_m = score_samples(dataset, seed=seed, method=method)
    t_kde = time.time() - t_kde_start

    time_ = np.linspace(eps, 1, N).astype(np.float32)
    g = diffusion_coeff_fn

    def diffuse(idx, channel):
        m[idx, 0, channel] = initial_m[idx, channel]
        for i in range(1, N):
            A_block = construct_A(
                H, W, dt / (2 * dh), dt / (dh ** 2), 0, 0,
                g(time_[i]), scores[idx, i, channel], bc=bc, stencil=stencil,
            )
            B_block = construct_B(H, W, m[idx, i - 1, channel])
            m[idx, i, channel] = sparse.linalg.spsolve(A_block, B_block).reshape((-1, H * W))
        log_density_gradient(m[idx, :, channel], H, W, dh, bc=bc, out=dm[idx, :, channel])

    convergence_log = []
    res = [1] * n_data
    e = 0
    t_fp_start = time.time()
    while max(res) > tol:
        e += 1
        for idx in tqdm(range(n_data)):
            if res[idx] <= tol:
                continue
            for ch in range(channels):
                diffuse(idx, ch)
            scores[idx] = dm[idx].copy()

            res[idx] = np.linalg.norm(m[idx] - m_prev[idx]) / np.linalg.norm(m_prev[idx])
            wall = time.time() - t_fp_start
            convergence_log.append((e, idx, res[idx], wall))
            tqdm.write(f'residual at iteration {e} for data {idx}: {res[idx]}')
            m_prev[idx] = m[idx].copy()

    t_fp = time.time() - t_fp_start

    if save_folder is not None:
        conv_path = os.path.join(save_folder, "convergence_log.csv")
        with open(conv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "data_idx", "residual", "wall_time_s"])
            writer.writerows(convergence_log)

        timing_path = os.path.join(save_folder, "timing.csv")
        with open(timing_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "time_s"])
            writer.writerow(["kde_init", f"{t_kde:.4f}"])
            writer.writerow(["fp_solve", f"{t_fp:.4f}"])
            writer.writerow(["fp_iterations", e])
            writer.writerow(["density_method", method])
            writer.writerow(["boundary", bc])

    return scores.reshape((n_data, -1, channels, H, W))


def marginal_prob_std(t, sigma):
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t
