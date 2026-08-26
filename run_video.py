"""Video pipeline driver: precompute scores, fit, inpaint, evaluate.

Usage
-----
    python run_video.py --config synth_inpaint.yml
    python run_video.py --config synth_inpaint.yml --epochs 5 --smoke
    python run_video.py --config synth_inpaint.yml --no-train   # evaluate a checkpoint

What it reports, and why
-----------------------
Inpainting quality is reported *inside the mask*, alongside mask coverage and the
whole-frame PSNR. The gap between masked and whole-frame PSNR is roughly
-10*log10(coverage) -- about 11 dB at 8% coverage -- so a whole-frame number quoted
alone would flatter any fill by that much. Both are printed so the difference is
visible rather than available.

Two baselines are run by default, because the method's claim is about temporal
reasoning and neither of these has any:
  * `per_frame`  -- the same masked sampler applied one frame at a time, with no
    temporal context. This is what an image method does. If the video path does not
    beat it, the temporal machinery is not earning its cost.
  * `copy_prev`  -- fill the hole with the previous frame's pixels at the same
    location. Trivial, and surprisingly strong on slow motion, which is exactly why
    it belongs in the table.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fast_diffusion"))

import numpy as np
import torch
import yaml

from data.VideoDataset import get_video_dataset
from fast_diffusion.model import evaluate_video as ev
from fast_diffusion.model import inpaint
from fast_diffusion.model.train_video import train_video
from network.network3d import VideoNet

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fast_diffusion", "configs", "video")


def build_mask(config, clip, dataset=None, clip_idx=None):
    """The hole to fill: the dataset's real mask when it has one, else synthetic.

    A dataset that ships real object masks (DAVIS) must take precedence over a
    generated box or stroke, and the choice has to be recorded in the results,
    because masked metrics are not comparable across mask families. A synthetic box
    covering 6% of the frame and a DAVIS object mask covering 55% are different
    tasks, and a table that mixes them says nothing.

    Returns (mask, source_name).
    """
    icfg = config.get("inpaint", {})
    requested = config.get("evaluation", {}).get("mask_kind",
                                                 icfg.get("mask", "moving_box"))

    if requested == "dataset":
        if dataset is None or not hasattr(dataset, "real_mask"):
            raise ValueError(
                "evaluation.mask_kind='dataset' but "
                f"{type(dataset).__name__} provides no real_mask(); use a DAVIS "
                "config or set a synthetic mask kind"
            )
        mask = torch.as_tensor(dataset.real_mask(clip_idx), dtype=torch.float32)
        name = f"dataset:{getattr(dataset, 'sequence_name', lambda i: i)(clip_idx)}"
        # The dataset resizes masks itself with nearest-neighbour; dilation is a
        # separate, config-level decision applied here for non-DAVIS sources.
        return mask, name

    T, _, H, W = clip.shape
    kwargs = {k: v for k, v in icfg.items() if k not in ("mask", "dilate")}
    mask = inpaint.make_mask(
        requested, T, H, W,
        dilate=int(icfg.get("dilate", 0)), **kwargs,
    )
    return mask, f"synthetic:{requested}"


def get_flows(config, clip, dataset, clip_idx):
    """Flows for the evaluation metrics, plus the name of their source.

    'ground_truth' is only available for synthetic data. Returning the resolved name
    alongside the flows keeps it in the results JSON, because a warping error is not
    comparable across flow estimators and a number without its estimator named is
    not reportable.
    """
    from fast_diffusion.model.flow import clip_flows, forward_backward_mask

    method = config.get("evaluation", {}).get("flow_method", "blockmatch")

    if method == "ground_truth":
        if not hasattr(dataset, "true_flow"):
            raise ValueError(
                f"evaluation.flow_method='ground_truth' but {type(dataset).__name__} "
                "has no true_flow(); use 'raft', 'blockmatch' or 'identity'"
            )
        # `true_flow()` is a single constant translation shared by every frame pair,
        # so it is replicated rather than indexed per frame.
        one = dataset.true_flow()
        flows = [one for _ in range(clip.shape[0] - 1)]
        # With exact flow the forward-backward check is still worth applying: it
        # removes disoccluded pixels, which have no correct correspondence regardless
        # of how good the flow is.
        masks = [forward_backward_mask(f, -f) for f in flows]

        cut = config["data_loader"].get("scene_cut_at")
        if cut is not None and 0 < cut < clip.shape[0]:
            # Across a cut there is no motion correspondence at all, so the
            # "ground-truth" translation is simply wrong there. Zero the mask for
            # that pair instead of letting a bogus correspondence inflate the
            # warping error and be read as model instability.
            masks[cut - 1] = torch.zeros_like(masks[cut - 1])
        return flows, masks, method

    flows, masks = clip_flows(clip, method=method, with_mask=True)
    return flows, masks, method


def baseline_per_frame(model, config, clip, mask, device, n_steps, seed=0,
                       clamp_output=True):
    """The masked sampler run frame by frame, with no temporal context.

    Each frame is solved as a (1-frame) clip, so the temporal convolutions and
    attention see a length-1 axis and can contribute nothing. This isolates what the
    temporal path buys.
    """
    out = clip.clone()
    for t in range(clip.shape[0]):
        if mask[t].sum() == 0:
            continue
        filled, _ = inpaint.pf_ode_inpaint(
            model, clip[t:t + 1][None].to(device), mask[t:t + 1],
            config["diffusion"]["sigma"], config["training"]["timestep_multiplier"],
            clip_idx=torch.zeros(1, dtype=torch.long),
            frame_idx=torch.tensor([t]), n_steps=n_steps,
            eps=float(config["misc"]["eps"]), device=device, seed=seed + t,
            clamp_output=clamp_output,
        )
        out[t] = filled[0, 0]
    return out


def baseline_copy_prev(clip, mask):
    """Fill each hole from the previous frame at the same pixel location.

    No motion compensation, no network. Frame 0 has no predecessor, so its hole is
    filled from frame 1 instead; leaving it as noise would make the baseline look
    artificially bad on the first frame.
    """
    out = clip.clone()
    m = torch.as_tensor(mask, dtype=torch.float32)
    T = clip.shape[0]
    for t in range(T):
        src = t - 1 if t > 0 else min(1, T - 1)
        if src == t:
            continue
        out[t] = (1 - m[t]) * clip[t] + m[t] * out[src]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="config file name in fast_diffusion/configs/video")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to saves/video/<config name>")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="override the masked sampler's fixed step count")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to check the pipeline end to end; results from a "
                         "smoke run are not usable as results")
    ap.add_argument("--no-train", action="store_true",
                    help="load model.pth from the output directory instead of training")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--no-score-cache", action="store_true",
                    help="recompute the FP score field even if a cached one exists")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    path = args.config if os.path.isfile(args.config) else os.path.join(
        CONFIG_DIR, args.config)
    if not os.path.isfile(path):
        raise SystemExit(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.seed is not None:
        config["data_loader"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.n_steps is not None:
        config.setdefault("sample", {})["n_steps"] = args.n_steps
    if args.no_score_cache:
        config.setdefault("diffusion", {})["cache_scores"] = False

    if args.smoke:
        # Shrink everything that costs time. Kept explicit so a smoke run cannot be
        # mistaken for a real one in the saved config.
        config["training"]["epochs"] = min(config["training"]["epochs"], 2)
        config["data_loader"]["clip_len"] = min(config["data_loader"]["clip_len"], 6)
        config["data_loader"]["image_size"] = min(config["data_loader"]["image_size"], 32)
        config["diffusion"]["num_timesteps"] = min(
            config["diffusion"]["num_timesteps"], 5)
        config["diffusion"]["max_fp_iterations"] = 2
        config.setdefault("sample", {})["n_steps"] = 10
        config["diffusion"]["cache_scores"] = False  # tiny scores must not touch the cache
        config["smoke_run"] = True

    out_dir = args.out_dir or os.path.join("saves", "video", config["name"])
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"config: {path}")
    print(f"output: {out_dir}")
    print(f"device: {device}")
    if args.smoke:
        print("SMOKE RUN -- reduced sizes; the numbers below are not results")

    # ---------------------------------------------------------------- train
    if args.no_train:
        ckpt_path = os.path.join(out_dir, "model.pth")
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"--no-train given but no checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt.get("config", config)
        model = VideoNet(config)
        model.load_state_dict(ckpt["model"])
        train_summary = {"loaded_from": ckpt_path}
    else:
        train_summary = train_video(config, out_dir, device=device)
        ckpt = torch.load(os.path.join(out_dir, "model.pth"),
                          map_location=device, weights_only=False)
        model = VideoNet(config)
        model.load_state_dict(ckpt["model"])

    model = model.to(device).eval()

    # ---------------------------------------------------------------- inpaint
    dataset = get_video_dataset(config)
    scfg = config.get("sample", {})
    n_steps = int(scfg.get("n_steps", 200))
    results = []

    for idx in range(len(dataset)):
        clip = torch.as_tensor(np.asarray(dataset[idx]), dtype=torch.float32)
        mask, mask_source = build_mask(config, clip, dataset, idx)
        cov = inpaint.mask_coverage(mask)
        print(f"\nclip {idx}: {tuple(clip.shape)}  mask {mask_source}  coverage "
              f"{cov['overall']*100:.1f}%")

        t0 = time.time()
        filled, info = inpaint.autoregressive_inpaint(
            model, config, clip, mask,
            window=int(scfg.get("window", 4)),
            overlap=int(scfg.get("overlap", 1)),
            clip_idx=idx, n_steps=n_steps, device=device,
            known_noise=scfg.get("known_noise", "fixed"),
            clamp_output=bool(scfg.get("clamp_output", True)),
            seed=config["data_loader"].get("seed", 0),
            constraints=scfg.get("constraints"),
            constraint_weight=float(scfg.get("constraint_weight", 1.0)),
            flow_method=config.get("diffusion", {}).get("flow_method", "blockmatch"),
            cg_tol=float(scfg.get("cg_tol", 1e-4)),
            cg_maxiter=int(scfg.get("cg_maxiter", 50)),
        )
        sample_time = time.time() - t0

        flows, flow_masks, flow_name = get_flows(config, filled, dataset, idx)
        rep = ev.evaluate_inpainting(
            filled, clip, mask, flow_method="identity",
            with_lpips=bool(config.get("evaluation", {}).get("with_lpips", False)),
        )
        # Recompute the flow-dependent entries with the configured flow source; the
        # call above uses 'identity' only so it does not estimate flow twice.
        rep["warping_error"] = ev.warping_error(filled, flows, flow_masks)
        rep["masked_warping_error"] = ev.masked_warping_error(
            filled, flows, mask, flow_masks)
        rep["flow_method"] = flow_name
        rep["sample_seconds"] = sample_time
        rep["nfev_total"] = int(sum(info["nfev"]))
        if info.get("cg_iters_total"):
            rep["cg_iters_total"] = int(sum(info["cg_iters_total"]))
        rep["method"] = "video"
        rep["clip"] = idx
        # Masked metrics are only comparable within one mask family, so the source
        # travels with every row rather than living in the config alone.
        rep["mask_source"] = mask_source
        results.append(rep)

        print(f"  video      masked PSNR {rep['masked_psnr']:.2f} dB   "
              f"whole-frame {rep['whole_frame_psnr']:.2f} dB   "
              f"masked warp {rep['masked_warping_error']:.4e}   {sample_time:.1f}s")

        if not args.skip_baselines:
            # The same clamp as the method, so the comparison is not decided by
            # a post-processing difference.
            pf = baseline_per_frame(
                model, config, clip, mask, device, n_steps,
                seed=config["data_loader"].get("seed", 0),
                clamp_output=bool(scfg.get("clamp_output", True)))
            cp = baseline_copy_prev(clip, mask)
            for name, out in (("per_frame", pf), ("copy_prev", cp)):
                r = ev.evaluate_inpainting(out, clip, mask, flow_method="identity")
                f2, m2, _ = get_flows(config, out, dataset, idx)
                r["warping_error"] = ev.warping_error(out, f2, m2)
                r["masked_warping_error"] = ev.masked_warping_error(out, f2, mask, m2)
                r["flow_method"] = flow_name
                r["mask_source"] = mask_source
                r["method"] = name
                r["clip"] = idx
                results.append(r)
                print(f"  {name:<10} masked PSNR {r['masked_psnr']:.2f} dB   "
                      f"whole-frame {r['whole_frame_psnr']:.2f} dB   "
                      f"masked warp {r['masked_warping_error']:.4e}")

        np.save(os.path.join(out_dir, f"filled_clip{idx}.npy"),
                filled.numpy().astype(np.float32))
        np.save(os.path.join(out_dir, f"mask_clip{idx}.npy"), mask)

    # ---------------------------------------------------------------- report
    summary = {
        "config_path": path,
        "config": config,
        "device": str(device),
        "train": {k: v for k, v in train_summary.items() if k != "score_info"},
        "score_info": train_summary.get("score_info"),
        "results": results,
        "smoke_run": bool(args.smoke),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n" + "=" * 74)
    print("SUMMARY (masked metrics; coverage is reported because masked and")
    print("whole-frame PSNR differ by about -10*log10(coverage))")
    print("=" * 74)
    print(f"{'method':<12}{'clip':>5}{'cover%':>8}{'mPSNR':>9}{'wfPSNR':>9}"
          f"{'mWarp':>11}{'seam':>11}")
    for r in results:
        print(f"{r['method']:<12}{r['clip']:>5}{r['coverage']['overall']*100:>8.1f}"
              f"{r['masked_psnr']:>9.2f}{r['whole_frame_psnr']:>9.2f}"
              f"{r['masked_warping_error']:>11.3e}{r['seam_error']:>11.3e}")

    vid = [r for r in results if r["method"] == "video"]
    base = [r for r in results if r["method"] == "per_frame"]
    if vid and base:
        d = np.mean([v["masked_psnr"] for v in vid]) - \
            np.mean([b["masked_psnr"] for b in base])
        print(f"\nvideo vs per-frame baseline: {d:+.2f} dB masked PSNR")
        if d <= 0:
            print("  The temporal path is NOT beating the per-frame baseline. Do not "
                  "report a temporal-consistency claim on this run.")
    if args.smoke:
        print("\nSMOKE RUN -- these numbers are not results.")
    print(f"\nwrote {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
