"""Offline FVD across inpainting methods, from saved filled clips.

FVD is a set-level metric, so it cannot live in the per-clip results.json rows.
This script computes it after a campaign has run, reusing the exact I3D feature
extractor and Frechet distance in `fast_diffusion.model.evaluate_video` (so the
value is comparable with published FVD), and reconstructing the real target
clips deterministically from the same config the campaign used.

Per seed it scores 70 real vs 70 fake clips (>= the 64-sample floor); it then
reports the mean and standard deviation across seeds, which is more honest than
tiling the identical real set to fake a larger sample count.

Method -> saved-file convention (written by run_video.py):
    video (control off/on) : filled_clip{idx}.npy   in the run's out_dir
    per_frame              : per_frame_clip{idx}.npy in the off out_dir
    copy_prev              : copy_prev_clip{idx}.npy in the off out_dir

Usage:
    python compute_fvd.py --config fast_diffusion/configs/video/davis_inpaint_128.yml \
        --base saves/video/davis128 --seeds 9 42 123 --out figures/fvd_128.json
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from data.VideoDataset import get_video_dataset
from fast_diffusion.model import evaluate_video as ev


def _load_set(directory, prefix, n):
    """Load {prefix}{i}.npy for i in [0, n) as (N, T, C, H, W) float32, or None."""
    clips = []
    for i in range(n):
        p = os.path.join(directory, f"{prefix}{i}.npy")
        if not os.path.isfile(p):
            return None
        clips.append(np.load(p).astype(np.float32))
    return np.stack(clips)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True,
                    help="out_dir stem; per-seed dirs are {base}_seed{seed} and "
                         "{base}_seed{seed}_control")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = get_video_dataset(config)
    n = len(dataset)
    real = np.stack([dataset[i].numpy().astype(np.float32) for i in range(n)])
    real_feats = ev.i3d_features(real, device=device)  # cached across methods

    # (method label, per-seed directory suffix, file prefix)
    specs = [
        ("per_frame", "", "per_frame_clip"),
        ("copy_prev", "", "copy_prev_clip"),
        ("video_off", "", "filled_clip"),
        ("video_on", "_control", "filled_clip"),
        ("propainter", "_propainter", "propainter_clip"),
    ]

    per_method = {}
    for label, suffix, prefix in specs:
        per_seed = []
        for seed in args.seeds:
            directory = f"{args.base}_seed{seed}{suffix}"
            fake = _load_set(directory, prefix, n)
            if fake is None:
                print(f"  {label:12s} seed {seed}: MISSING in {directory} -- skipped")
                continue
            ff = ev.i3d_features(fake, device=device)
            val = ev._frechet_distance(real_feats, ff)
            per_seed.append(val)
            print(f"  {label:12s} seed {seed}: FVD = {val:.2f} (n={len(fake)})")
        if per_seed:
            per_method[label] = {
                "fvd_mean": float(np.mean(per_seed)),
                "fvd_std": float(np.std(per_seed)),
                "per_seed": per_seed,
                "n_seeds": len(per_seed),
            }

    print("\n=== FVD summary (mean +/- std across seeds; lower is better) ===")
    for label, d in per_method.items():
        print(f"  {label:12s} {d['fvd_mean']:8.2f} +/- {d['fvd_std']:6.2f}  "
              f"({d['n_seeds']} seeds)")

    result = {
        "config": args.config,
        "n_clips": n,
        "frames_per_clip": int(real.shape[1]),
        "resolution": int(real.shape[-1]),
        "backbone": "i3d",
        "comparable_with_published": True,
        "methods": per_method,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
