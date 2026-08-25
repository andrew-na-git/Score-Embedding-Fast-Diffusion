"""Post-hoc Frechet Video Distance for a finished video-inpainting run.

Why this exists
---------------
`run_video.py` computes only per-clip metrics (masked PSNR/SSIM, warping error)
in its evaluation loop. FVD is a *distributional* metric: a single Frechet
distance between the I3D feature distribution of the real clips and that of the
generated clips, defined only over a set (>= 64 clips; see MIN_FVD_SAMPLES). It
therefore cannot be produced inside a per-clip loop, and `evaluation.compute_fvd`
in the DAVIS config is currently read by nobody.

This script closes that gap without re-running anything: every run already saves
`clips.npy` (the real clips) and `filled_clip{idx}.npy` (the video method's fill),
so FVD can be computed from those artifacts alone, once per run directory.

Scope and a caveat you must keep attached to the number
-------------------------------------------------------
* Only the **video method** is scored, because only its fills are saved. The
  `per_frame` baseline needs the trained model + sampler to reproduce and the
  `copy_prev` baseline needs re-synthesis; neither array is on disk. To get their
  FVD, save their fills in `run_video.py` and point this script at them.
* FVD here is **whole-frame**, not masked -- I3D ingests entire frames and there
  is no masked FVD. At DAVIS's ~12% mean coverage most of each frame is the
  untouched original, so this FVD is dominated by the known region and is
  optimistic about the fill. Report it as whole-frame FVD, alongside the masked
  per-clip metrics, not instead of them.

Usage
-----
    python evaluate_fvd_run.py saves/video/davis_seed9
    python evaluate_fvd_run.py saves/video/davis_seed9 saves/video/davis_seed42
    python evaluate_fvd_run.py saves/video/davis_seed*        # shell-expanded

Writes `<run_dir>/fvd.json` and prints a one-line summary per run. Requires the
I3D weights (`python download_assets.py --only i3d`); `fvd()` raises without them
rather than substituting a non-comparable backbone.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fast_diffusion"))

import numpy as np

from fast_diffusion.model.evaluate_video import (
    MIN_FVD_SAMPLES, MIN_I3D_FRAMES, fvd,
)


def _load_run(run_dir):
    """Return (reals, fakes) as (N, T, C, H, W) float32 arrays, or raise.

    `reals` is `clips.npy`; `fakes` is the per-clip `filled_clip{idx}.npy` stack,
    in clip order so real[i] and fake[i] are the same clip.
    """
    clips_path = os.path.join(run_dir, "clips.npy")
    if not os.path.isfile(clips_path):
        raise SystemExit(
            f"{run_dir}: no clips.npy -- the run has not reached evaluation yet, "
            "or this is not a run directory."
        )
    reals = np.load(clips_path).astype(np.float32)
    if reals.ndim != 5:
        raise SystemExit(
            f"{run_dir}: clips.npy has shape {reals.shape}, expected (N, T, C, H, W)."
        )

    n = reals.shape[0]
    fills, missing = [], []
    for i in range(n):
        p = os.path.join(run_dir, f"filled_clip{i}.npy")
        if os.path.isfile(p):
            fills.append(np.load(p).astype(np.float32))
        else:
            missing.append(i)
    if missing:
        raise SystemExit(
            f"{run_dir}: missing {len(missing)}/{n} filled_clip*.npy "
            f"(first few: {missing[:5]}). The run's inpainting loop is incomplete."
        )

    fakes = np.stack(fills)
    if fakes.shape != reals.shape:
        raise SystemExit(
            f"{run_dir}: fills stack {fakes.shape} != reals {reals.shape}."
        )
    return reals, fakes


def _range_guard(name, arr):
    """FVD preprocessing assumes [0, 1]; warn (do not silently rescale) on drift."""
    lo, hi = float(arr.min()), float(arr.max())
    if lo < -0.01 or hi > 1.01:
        print(f"  WARNING {name} outside [0,1] (min {lo:.3f}, max {hi:.3f}); "
              f"clipping for FVD preprocessing.")
    return np.clip(arr, 0.0, 1.0)


def evaluate_one(run_dir, weights=None, device=None):
    reals, fakes = _load_run(run_dir)
    n, t = reals.shape[0], reals.shape[1]

    if t < MIN_I3D_FRAMES:
        raise SystemExit(
            f"{run_dir}: {t} frames/clip < I3D minimum {MIN_I3D_FRAMES}."
        )
    if n < MIN_FVD_SAMPLES:
        # fvd() would raise via check_sample_count; surface it as a clear message
        # rather than a traceback, and record why no number was produced.
        msg = (f"{run_dir}: only {n} clips < MIN_FVD_SAMPLES ({MIN_FVD_SAMPLES}); "
               "FVD is not meaningful below this and is not computed.")
        print("  " + msg)
        return {"run_dir": run_dir, "fvd": None, "reason": msg, "n_clips": n}

    reals = _range_guard("reals", reals)
    fakes = _range_guard("fakes", fakes)

    result = fvd(reals, fakes, device=device, weights=weights)
    result["run_dir"] = run_dir
    result["method"] = "video"
    result["masked"] = False
    result["note"] = ("whole-frame FVD; dominated by the untouched region at low "
                       "mask coverage. Report alongside masked per-clip metrics.")

    out_path = os.path.join(run_dir, "fvd.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"  FVD {result['fvd']:.2f}  (n={result['n_real']}, "
          f"T={result['frames_per_clip']}, {result['feature_dim']}-d I3D)  "
          f"-> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+",
                    help="one or more run directories (each with clips.npy + "
                         "filled_clip*.npy)")
    ap.add_argument("--weights", default=None,
                    help="path to the I3D TorchScript weights (defaults to the "
                         "canonical assets/ path used by evaluate_video)")
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto by default)")
    args = ap.parse_args()

    summary = []
    for run_dir in args.run_dirs:
        print(f"{run_dir}:")
        summary.append(evaluate_one(run_dir, weights=args.weights, device=args.device))

    scored = [s for s in summary if s.get("fvd") is not None]
    if len(scored) > 1:
        vals = [s["fvd"] for s in scored]
        print(f"\nacross {len(scored)} runs: FVD mean {np.mean(vals):.2f} "
              f"std {np.std(vals):.2f}  min {min(vals):.2f} max {max(vals):.2f}")


if __name__ == "__main__":
    main()
