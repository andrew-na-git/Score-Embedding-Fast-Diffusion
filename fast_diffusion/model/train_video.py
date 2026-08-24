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
3. Warm starts. Successive clips are similar, so each clip's fixed-point iteration
   starts from the previous clip's converged score field. `SequentialDensityEstimator`
   decides when that is no longer safe and a keyframe is needed. This is where the
   paper's speedup comes from, so `score_precompute` records the per-clip iteration
   counts with and without the warm start when `measure_warm_start=True` -- a claimed
   speedup with no such measurement is not a result.

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


def score_precompute(config, dataset, save_folder=None, device=None,
                     measure_warm_start=False):
    """Solve the FP equation for every clip, warm-starting from the previous clip.

    The initial condition is built exactly as the image path builds it
    (`kfp.image_channel_to_samples` + `estimate_log_density`), so the solver's domain
    is the *pixel-value pair* grid, not the spatial grid -- the (H, W) axes of
    `initial_m` are value axes. The added axis is time, which is spatial in the
    ordinary sense. Mixing those up would be easy and silent, hence this note.

    One `SequentialDensityEstimator` is kept per colour channel. Sharing a single
    estimator across channels would interleave three unrelated density sequences into
    one running grid and one KL history, so its keyframe decisions would be driven by
    channel switches rather than by scene changes.

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
    warm = None

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

        # Which keyframes matter for the warm-start decision.
        #
        # Frame 0 of clip 0 is *always* a keyframe -- there is no prior grid to
        # correct -- so treating any keyframe as disqualifying would disable warm
        # starts on the very first clip and then, because each clip boundary is itself
        # a density discontinuity when clips are independent scenes, on every clip
        # after it. That is exactly what happened before this was separated out: warm
        # starts never once engaged and the mechanism went unmeasured.
        #
        # A keyframe at frame 0 of a later clip is a genuine signal that the clip
        # boundary is a cut, and it does disqualify the warm start. Keyframes strictly
        # inside the clip mean the content changed mid-clip and also disqualify it.
        bootstrap = (idx == 0)
        boundary_keyframe = keyframes[0] if keyframes else False
        interior_keyframes = [i for i, k in enumerate(keyframes) if k and i > 0]

        use_warm = (
            warm is not None
            and not interior_keyframes
            and not (boundary_keyframe and not bootstrap)
        )

        t0 = time.perf_counter()
        if backend == "torch":
            sc, solve_info = compute_scores_clip_torch(
                config, initial_m, warm_start_scores=warm if use_warm else None,
                device=device,
            )
        else:
            sc, solve_info = compute_scores_clip(
                config, initial_m, warm_start_scores=warm if use_warm else None,
            )
        elapsed = time.perf_counter() - t0

        record = {
            "clip": idx,
            "iterations": solve_info["iterations"],
            "converged": bool(solve_info["converged"]),
            "seconds": elapsed,
            "warm_started": bool(use_warm),
            "keyframe_frames": [i for i, k in enumerate(keyframes) if k],
            "boundary_keyframe": bool(boundary_keyframe),
            "interior_keyframes": interior_keyframes,
            "kl_max": max((k for k in kls if np.isfinite(k)), default=None),
            "backend": backend,
        }

        # The cold-start control. Without it the warm-start speedup is an assertion.
        if measure_warm_start and use_warm:
            t0 = time.perf_counter()
            if backend == "torch":
                _, cold = compute_scores_clip_torch(config, initial_m, device=device)
            else:
                _, cold = compute_scores_clip(config, initial_m)
            record["cold_iterations"] = cold["iterations"]
            record["cold_seconds"] = time.perf_counter() - t0

        per_clip.append(record)
        warm = sc

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


def train_video(config, save_folder, device=None, measure_warm_start=False):
    """Full video pipeline: dataset -> score precompute -> fit -> save."""
    from data.VideoDataset import get_video_dataset
    from network.network3d import VideoNet

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_folder, exist_ok=True)

    dataset = get_video_dataset(config)
    np.save(os.path.join(save_folder, "clips.npy"),
            np.stack([np.asarray(dataset[i]) for i in range(len(dataset))]))

    print(f"precomputing scores for {len(dataset)} clip(s) on {device}")
    t0 = time.time()
    scores, score_info = score_precompute(
        config, dataset, save_folder, device, measure_warm_start)
    fp_time = time.time() - t0

    iters = [r["iterations"] for r in score_info["per_clip"]]
    print(f"  {fp_time:.1f}s total, iterations per clip: {iters}")
    if measure_warm_start:
        warm = [r for r in score_info["per_clip"] if "cold_iterations" in r]
        if warm:
            w = np.mean([r["iterations"] for r in warm])
            c = np.mean([r["cold_iterations"] for r in warm])
            print(f"  warm start: {w:.2f} vs cold {c:.2f} iterations "
                  f"({c/max(w,1e-9):.2f}x fewer)")

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
