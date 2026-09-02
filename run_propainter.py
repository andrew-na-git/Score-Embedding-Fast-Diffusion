"""Run ProPainter as an external video-inpainting baseline, scored identically.

Fairness is the whole point of this script, so three things are forced:

* ProPainter sees the *same* clips and the *same* DAVIS masks our method sees --
  reconstructed from the same config and dataset, at the same 128 px.
* Its output is re-composited onto the original frame outside our hole, so
  observed pixels are bit-exact (ProPainter dilates masks internally; without
  this the masked metrics' observed-pixel assumption would be violated).
* Scoring uses the identical `evaluate_inpainting` + block-matching warp metric
  as `run_video.py`, so the ProPainter row is comparable with the others.

Per-clip filled clips are saved as propainter_clip{idx}.npy so FVD can be
computed offline alongside the other methods.

Usage:
    python run_propainter.py --config fast_diffusion/configs/video/davis_inpaint_128.yml \
        --seed 9 --out-dir saves/video/davis128_seed9_propainter
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml
from PIL import Image

from data.VideoDataset import get_video_dataset
from fast_diffusion.model import evaluate_video as ev

PROPAINTER_DIR = os.path.join("external", "ProPainter")


def _write_frames(clip, mask, frame_dir, mask_dir):
    """clip (T,C,H,W) in [0,1], mask (T,1,H,W) 1=hole -> PNGs (white=hole)."""
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    T = clip.shape[0]
    for t in range(T):
        rgb = (clip[t].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(frame_dir, f"{t:04d}.png"))
        m = (mask[t, 0].numpy() > 0.5).astype(np.uint8) * 255
        Image.fromarray(m, mode="L").save(os.path.join(mask_dir, f"{t:04d}.png"))


def _read_frames(frames_dir, T, res):
    out = []
    for t in range(T):
        p = os.path.join(frames_dir, f"{t:04d}.png")
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(img).permute(2, 0, 1))
    x = torch.stack(out)  # (T,C,H,W)
    if x.shape[-1] != res:
        x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear",
                                             align_corners=False)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N clips")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if args.seed is not None:
        config["data_loader"]["seed"] = args.seed

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = get_video_dataset(config)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    res = int(config["data_loader"]["image_size"])
    flow_method = config.get("evaluation", {}).get("flow_method", "blockmatch")

    results = []
    work = tempfile.mkdtemp(prefix="propainter_")
    for idx in range(n):
        clip = dataset[idx]                      # (T,C,H,W) in [0,1]
        mask = dataset.real_mask(idx)            # (T,1,H,W) 1=hole
        T = clip.shape[0]

        frame_dir = os.path.join(work, f"clip{idx}", "frames")
        mask_dir = os.path.join(work, f"clip{idx}", "masks")
        out_dir = os.path.join(work, f"clip{idx}", "out")
        _write_frames(clip, mask, frame_dir, mask_dir)

        cmd = [sys.executable, "inference_propainter.py",
               "-i", os.path.abspath(frame_dir),
               "-m", os.path.abspath(mask_dir),
               "-o", os.path.abspath(out_dir),
               "--width", str(res), "--height", str(res),
               "--save_frames", "--mask_dilation", "4"]
        if args.fp16:
            cmd.append("--fp16")
        proc = subprocess.run(cmd, cwd=PROPAINTER_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
            raise SystemExit(f"ProPainter failed on clip {idx}")

        # ProPainter writes <out>/frames/<video_name>/... ; the video name is the
        # frames dir basename ('frames'). Locate the produced frames robustly.
        cand = os.path.join(out_dir, "frames", "frames")
        if not os.path.isdir(cand):
            subdirs = [d for d in os.listdir(out_dir)
                       if os.path.isdir(os.path.join(out_dir, d))]
            cand = None
            for d in [os.path.join(out_dir, s) for s in subdirs]:
                fr = os.path.join(d, "frames")
                if os.path.isdir(fr):
                    cand = fr
                    break
            if cand is None:
                raise SystemExit(f"cannot find ProPainter frames under {out_dir}")

        prop = _read_frames(cand, T, res)        # (T,C,H,W) in [0,1]

        # Force observed pixels bit-exact: keep the original outside our hole.
        filled = clip * (1 - mask) + prop * mask

        rep = ev.evaluate_inpainting(filled, clip, mask, flow_method="identity")
        from fast_diffusion.model.flow import clip_flows
        flows, flow_masks = clip_flows(filled, method=flow_method, with_mask=True)
        rep["warping_error"] = ev.warping_error(filled, flows, flow_masks)
        rep["masked_warping_error"] = ev.masked_warping_error(
            filled, flows, mask, flow_masks)
        rep["flow_method"] = flow_method
        rep["method"] = "propainter"
        rep["clip"] = idx
        rep["mask_source"] = "dataset"
        results.append(rep)
        np.save(os.path.join(args.out_dir, f"propainter_clip{idx}.npy"),
                filled.numpy().astype(np.float32))
        print(f"  clip {idx:3d}: masked PSNR {rep['masked_psnr']:.2f} dB   "
              f"masked warp {rep['masked_warping_error']:.4e}")

    summary = {"config": args.config, "seed": args.seed, "method": "propainter",
               "n_clips": n, "resolution": res, "results": results}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    mp = float(np.mean([r["masked_psnr"] for r in results]))
    mw = float(np.mean([r["masked_warping_error"] for r in results]))
    print(f"\nProPainter mean: masked PSNR {mp:.2f} dB   masked warp {mw:.4e}   "
          f"({n} clips)")


if __name__ == "__main__":
    main()
