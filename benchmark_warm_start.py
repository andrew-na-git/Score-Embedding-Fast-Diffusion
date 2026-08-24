"""Measure the cross-clip warm start against a cold-start control.

Two claims need separate evidence and this script produces both:

1. On contiguous clips (one sequence, consecutive windows) the warm start reduces the
   fixed-point iteration count.
2. On independent clips the KL trigger declines the warm start. That is the *correct*
   behaviour, not a failure -- a warm start across a cut would start from a bad guess.

Reporting only (1) would be reporting half the mechanism.
"""
import json
import time

import numpy as np
import torch
import yaml

from data.VideoDataset import get_video_dataset
from fast_diffusion.model.train_video import score_precompute

CONFIGS = {
    "contiguous": "fast_diffusion/configs/video/synth_contiguous.yml",
    "independent": "fast_diffusion/configs/video/synth_inpaint.yml",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = {}

for label, path in CONFIGS.items():
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    # Keep it affordable: the point is the iteration counts, not the resolution.
    config["data_loader"]["image_size"] = 48
    config["data_loader"]["clip_len"] = 8
    config["data_loader"]["num_clips"] = 4
    config["model"]["num_clips"] = 4
    config["diffusion"]["num_timesteps"] = 10

    ds = get_video_dataset(config)
    print(f"\n=== {label}: {len(ds)} clips of {tuple(ds[0].shape)} ===")

    t0 = time.time()
    _, info = score_precompute(config, ds, device=device, measure_warm_start=True)
    print(f"  {time.time()-t0:.1f}s")

    rows = []
    for r in info["per_clip"]:
        rows.append(r)
        cold = r.get("cold_iterations")
        print(f"  clip {r['clip']}: warm_started={str(r['warm_started']):<5} "
              f"iters={r['iterations']}"
              + (f" cold_iters={cold}" if cold is not None else "")
              + f"  boundary_kf={r['boundary_keyframe']} "
              f"interior_kf={r['interior_keyframes']} kl_max={r['kl_max']}")
    out[label] = rows

    warmed = [r for r in rows if r.get("cold_iterations") is not None]
    if warmed:
        w = np.mean([r["iterations"] for r in warmed])
        c = np.mean([r["cold_iterations"] for r in warmed])
        ws = np.mean([r["seconds"] for r in warmed])
        cs = np.mean([r["cold_seconds"] for r in warmed])
        print(f"  --> warm {w:.2f} iters / {ws:.1f}s   cold {c:.2f} iters / {cs:.1f}s")
        print(f"  --> iteration saving {c/max(w,1e-9):.2f}x, "
              f"wall-clock {cs/max(ws,1e-9):.2f}x  (n={len(warmed)} warm-started clips)")
    else:
        print(f"  --> no clip was warm-started; nothing to compare. "
              f"For '{label}' that is the expected outcome if clips are independent.")

with open("figures/warm_start_measurement.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote figures/warm_start_measurement.json")
