"""Video training loop: precompute clip scores, then fit the network to them.

Relation to `train.py`
----------------------
The image path precomputes a score field per image and fits `network.Net` indexed by
an image identity embedding. This is the same regime -- per-instance fitting, not
distribution learning -- with three differences that matter:

1. Scores are solved on a (T, H, W) grid by `fp_video` / `fp_torch` rather than a
   per-frame (H, W) grid, so temporal structure is in the score field itself rather
   than something the network has to infer.
2. Clips are indexed by a clip embedding and frames by a frame embedding, so one
   network holds several clips exactly as the image path holds several images.
3. The initial log-density is produced by `SequentialDensityEstimator`, which carries
   a running density grid across frames and applies an importance correction rather
   than re-estimating from scratch, re-estimating only when its KL statistic says the
   content has changed. That is the sequential-importance-sampling contribution.

Removed: cross-clip score warm starts
-------------------------------------
An earlier revision started each clip's fixed-point iteration from the previous
clip's converged score field. It is gone, and the reason is worth recording so it is
not reintroduced. Measured against a cold-start control it saved 1.12x in iterations
(4.00 vs 4.50) -- because upwinding already converges in 4-5 iterations from a cold
start, there was almost nothing left to save. The stability work consumed the headroom
the warm start was meant to exploit, and carrying a whole mechanism, its config
surface and its correctness caveats for 12% was not worth it.

The KL keyframe trigger *survives* this removal: it still governs whether the density
estimator corrects its running grid or re-estimates it, which is a separate and
load-bearing mechanism.

Score fields are large: (N, C, T, H, W) float32 at N=20, C=3, T=16, H=W=128 is 2.0 GB
per clip. `score_store` handles that; this module hands it the array and does not
hold more than one clip's field in memory at a time.
"""

import csv
import json
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from .density import SequentialDensityEstimator, density_method_from_config
from .fp_video import compute_scores_clip
from .kfp import diffusion_coeff, image_channel_to_samples, marginal_prob_std
from .loss import video_score_matching_loss


def _select_backend(config, device):
    """Choose the numpy or torch FP backend.

    The GPU backend is a loss below roughly 64x64 -- it is kernel-launch bound, so
    small grids pay launch overhead without earning it back (measured: 0.36x at
    8x32x32, 5.4x at 16x128x128; see `benchmark_solver.py`). Default to whichever
    wins at the configured resolution rather than assuming GPU is always faster.
    """
    requested = config.get("diffusion", {}).get("backend", "auto")
    size = config["data_loader"]["image_size"]

    if requested == "numpy":
        return "numpy"
    if requested == "torch":
        return "torch"
    if requested != "auto":
        raise ValueError(
            f"unknown diffusion.backend {requested!r}; expected 'auto', 'numpy' or 'torch'"
        )
    if device.type != "cuda":
        return "numpy"
    return "torch" if size >= 96 else "numpy"


def score_precompute(config, dataset, save_folder=None, device=None):
    """Solve the FP equation for every clip.

    The initial condition is built exactly as the image path builds it
    (`kfp.image_channel_to_samples` + `estimate_log_density`), so the solver's domain
    is the *pixel-value pair* grid, not the spatial grid -- the (H, W) axes of
    `initial_m` are value axes. The added axis is time, which is spatial in the
    ordinary sense. Mixing those up would be easy and silent, hence this note.

    One `SequentialDensityEstimator` is kept per colour channel. Sharing a single
    estimator across channels would interleave three unrelated density sequences into
    one running grid and one KL history, so its keyframe decisions would be driven by
    channel switches rather than by scene changes.

    Each clip's fixed-point iteration starts cold. Warm-starting it from the previous
    clip was measured at 1.12x and removed; see the module docstring.

    Returns
    -------
    scores : (n_clips, N, C, T, H, W) float32.
    info : per-clip iteration counts, keyframe decisions and KL values.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = _select_backend(config, device)

    if backend == "torch":
        from .fp_torch import compute_scores_clip_torch

    method = density_method_from_config(config)
    est_kwargs = config.get("diffusion", {}).get("estimator", {}) or {}

    n_clips = len(dataset)
    clip0 = np.asarray(dataset[0])
    T, C = clip0.shape[0], clip0.shape[1]
    estimators = [
        SequentialDensityEstimator(method=method, **est_kwargs) for _ in range(C)
    ]

    per_clip = []
    scores = None

    for idx in range(n_clips):
        clip = np.asarray(dataset[idx], dtype=np.float64)  # (T, C, H, W)
        H, W = clip.shape[2], clip.shape[3]

        initial_m = np.empty((C, T, H, W), dtype=np.float64)
        keyframes, kls = [], []

        for t in range(T):
            for c in range(C):
                xy = image_channel_to_samples(clip[t, c])
                log_p, est_info = estimators[c].estimate(xy)
                initial_m[c, t] = log_p.reshape(H, W)
                if c == 0:
                    # Trigger diagnostics are reported from the luminance-ish first
                    # channel; recording all three would triple the log for no gain.
                    keyframes.append(bool(est_info["keyframe"]))
                    kls.append(float(est_info["kl"]))

        t0 = time.perf_counter()
        if backend == "torch":
            sc, solve_info = compute_scores_clip_torch(config, initial_m, device=device)
        else:
            sc, solve_info = compute_scores_clip(config, initial_m)
        elapsed = time.perf_counter() - t0

        record = {
            "clip": idx,
            "iterations": solve_info["iterations"],
            "converged": bool(solve_info["converged"]),
            "seconds": elapsed,
            # Frame 0 of clip 0 is always a keyframe: there is no prior grid to
            # correct. Later frame-0 keyframes mean the clip boundary is a genuine
            # content change; interior ones mean it changed mid-clip.
            "keyframe_frames": [i for i, k in enumerate(keyframes) if k],
            "boundary_keyframe": bool(keyframes[0]) if keyframes else False,
            "interior_keyframes": [i for i, k in enumerate(keyframes) if k and i > 0],
            "kl_max": max((k for k in kls if np.isfinite(k)), default=None),
            "backend": backend,
        }

        per_clip.append(record)

        if scores is None:
            scores = np.empty((n_clips,) + sc.shape, dtype=np.float32)
        scores[idx] = sc

        if not solve_info["converged"]:
            print(f"  WARNING clip {idx}: FP iteration did not converge "
                  f"(residual {solve_info['residuals'][-1]:.2e})")

    info = {
        "per_clip": per_clip,
        "backend": backend,
        "estimator": method,
        "keyframes_per_channel": [e.n_keyframes for e in estimators],
        "frames_seen_per_channel": [e.n_frames for e in estimators],
    }

    if save_folder:
        with open(os.path.join(save_folder, "score_precompute.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

    return scores, info


def _sample_batch(clip, scores, config, device, generator=None):
    """One training batch: perturb a clip at random diffusion times.

    `scores` is the precomputed field (N, C, T, H, W); the entry nearest each drawn
    time supplies the weight denominator, which is how the precomputed solve enters
    training.
    """
    sigma = config["diffusion"]["sigma"]
    N = config["diffusion"]["num_timesteps"]
    eps = float(config["misc"]["eps"])
    batch_size = config["data_loader"]["batch_size"]

    t = torch.rand(batch_size, generator=generator) * (1.0 - eps) + eps
    std = marginal_prob_std(t, sigma)

    z = torch.randn((batch_size,) + tuple(clip.shape), generator=generator)
    batch = clip[None] + z * std[:, None, None, None, None]

    # Nearest precomputed timestep for each drawn t.
    grid = np.linspace(eps, 1, N)
    idx = np.abs(grid[None, :] - t.numpy()[:, None]).argmin(axis=1)
    sc = torch.as_tensor(scores[idx], dtype=torch.float32)
    diff_std2 = (torch.as_tensor(
        [float(diffusion_coeff(ti, sigma)) for ti in t]) ** 2)

    return (batch.to(device), t.to(device), diff_std2.to(device), std.to(device),
            z.to(device), sc)


def diffuse_train_video(model, dataset, scores, config, save_folder, device=None):
    """Fit the network to the precomputed score fields."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=float(config["training"]["lr"]))

    epochs = int(config["training"]["epochs"])
    grad_clip = config["training"].get("grad_clip", False)
    n_clips = len(dataset)
    T = dataset[0].shape[0]

    losses = np.zeros((n_clips, epochs), dtype=np.float64)
    t_start = time.time()

    for e in tqdm(range(epochs), desc="epochs"):
        for i in range(n_clips):
            clip = torch.as_tensor(np.asarray(dataset[i]), dtype=torch.float32)
            batch, t, diff_std2, std, z, _ = _sample_batch(
                clip, scores[i], config, device)

            b = batch.shape[0]
            clip_idx = torch.full((b,), i, dtype=torch.long, device=device)
            frame_idx = torch.arange(T, dtype=torch.long, device=device)

            loss = video_score_matching_loss(
                model, batch, t * config["training"]["timestep_multiplier"],
                diff_std2, std, z, clip_idx=clip_idx, frame_idx=frame_idx,
            )
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            losses[i, e] = loss.item()

        if (e + 1) % max(1, epochs // 10) == 0:
            tqdm.write(f"epoch {e+1}/{epochs}  loss {losses[:, e].mean():.4f}")

    return time.time() - t_start, {"losses": losses}


def _score_cache_key(config, clips):
    """Content hash identifying a score field: clip pixels + everything that
    changes the FP solve.

    Keyed on the actual clip array rather than the config's clip selection, so it
    is correct regardless of how the dataset picks clips (two configs that resolve
    to the same pixels share a cache entry; anything that changes the pixels or the
    solve misses). The solver/estimator source files are folded in so that editing
    the solver invalidates stale scores instead of silently reusing them -- the one
    failure mode that would corrupt a whole campaign.

    The training seed is deliberately NOT in the key: scores do not depend on it,
    which is exactly what lets a multi-seed campaign reuse one precompute.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(clips, dtype=np.float32).tobytes())
    h.update(json.dumps(config.get("diffusion", {}), sort_keys=True,
                        default=str).encode())
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("fp_video.py", "fp_torch.py", "density.py", "kfp.py"):
        try:
            with open(os.path.join(here, name), "rb") as fh:
                h.update(fh.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _load_or_compute_scores(config, dataset, clips, save_folder, device):
    """Return (scores, info, was_cached).

    Cache is on by default; disable with `diffusion.cache_scores: false` or the
    `--no-score-cache` flag in run_video.py. Location is `diffusion.score_cache_dir`
    (default `saves/video/_score_cache`), which is gitignored.
    """
    diff = config.get("diffusion", {})
    if not bool(diff.get("cache_scores", True)):
        scores, info = score_precompute(config, dataset, save_folder=None, device=device)
        return scores, info, False

    cache_dir = diff.get("score_cache_dir") or os.path.join(
        "saves", "video", "_score_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = _score_cache_key(config, clips)
    spath = os.path.join(cache_dir, key + ".npy")
    ipath = os.path.join(cache_dir, key + ".info.json")

    if os.path.exists(spath) and os.path.exists(ipath):
        scores = np.load(spath)
        with open(ipath, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        print(f"  score cache HIT {key} ({scores.nbytes / 1e6:.0f} MB) <- {spath}")
        return scores, info, True

    scores, info = score_precompute(config, dataset, save_folder=None, device=device)
    np.save(spath, scores)
    with open(ipath, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(f"  score cache MISS {key}; saved {scores.nbytes / 1e6:.0f} MB -> {spath}")
    return scores, info, False


def train_video(config, save_folder, device=None):
    """Full video pipeline: dataset -> score precompute -> fit -> save."""
    from data.VideoDataset import get_video_dataset
    from network.network3d import VideoNet

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_folder, exist_ok=True)

    dataset = get_video_dataset(config)
    clips_arr = np.stack([np.asarray(dataset[i]) for i in range(len(dataset))])
    np.save(os.path.join(save_folder, "clips.npy"), clips_arr)

    print(f"precomputing scores for {len(dataset)} clip(s) on {device}")
    t0 = time.time()
    scores, score_info, was_cached = _load_or_compute_scores(
        config, dataset, clips_arr, save_folder, device)
    fp_time = time.time() - t0

    # Written for this run's report whether the scores were solved or loaded; on a
    # cache hit the per-clip solve seconds inside score_info are the originals.
    score_info = dict(score_info, cached=was_cached)
    with open(os.path.join(save_folder, "score_precompute.json"), "w",
              encoding="utf-8") as fh:
        json.dump(score_info, fh, indent=2)

    iters = [r["iterations"] for r in score_info["per_clip"]]
    keys = sum(len(r["keyframe_frames"]) for r in score_info["per_clip"])
    print(f"  {fp_time:.1f}s total, iterations per clip: {iters}")
    print(f"  keyframes: {keys} full re-estimates over "
          f"{score_info['frames_seen_per_channel'][0]} frames")

    model = VideoNet(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"network: {n_params/1e6:.2f} M parameters")

    # The clip embedding is sized from the config, so a mismatch against the actual
    # dataset would silently index out of range (or, worse, alias two clips onto one
    # embedding row and quietly degrade the fit).
    n_emb = model.clip_emb.num_embeddings
    if n_emb < len(dataset):
        raise ValueError(
            f"model.num_clips is {n_emb} but the dataset has {len(dataset)} clips; "
            "set model.num_clips to match data_loader.num_clips"
        )

    train_time, metrics = diffuse_train_video(
        model, dataset, scores, config, save_folder, device)

    torch.save({"model": model.state_dict(), "config": config},
               os.path.join(save_folder, "model.pth"))
    np.save(os.path.join(save_folder, "losses.npy"), metrics["losses"])

    with open(os.path.join(save_folder, "timing.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "seconds"])
        w.writerow(["fp_solve", f"{fp_time:.3f}"])
        w.writerow(["training", f"{train_time:.3f}"])
        w.writerow(["total", f"{fp_time + train_time:.3f}"])

    return {"fp_time": fp_time, "train_time": train_time,
            "n_params": n_params, "score_info": score_info}
