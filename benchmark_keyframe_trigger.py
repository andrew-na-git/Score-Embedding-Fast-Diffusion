"""Measure the sequential density estimator's KL keyframe trigger.

This is the surviving half of the sequential-importance-sampling contribution. The
cross-clip score warm start was removed after measuring at 1.12x (see PLAN.md); the
mechanism here is different and independent of it: rather than re-estimating the
log-density from scratch on every frame, the estimator carries a running grid and
applies an importance correction, re-estimating fully only when its KL statistic says
the content has actually changed.

Three things need separate evidence, and reporting fewer than all three would
overstate the result:

1. **Cost.** Incremental correction must be cheaper than full re-estimation per frame.
   That is the efficiency claim.
2. **Discrimination.** The KL statistic must separate a real scene cut from ordinary
   within-shot variation by a wide margin. Measured here as a ratio.
3. **False positives.** On a control sequence with no cut, the trigger must stay quiet.
   A trigger that fires constantly is cheap to build and worthless -- it degenerates to
   full re-estimation while claiming to be adaptive.

An earlier ESS-based trigger passed (1) and (3) trivially and failed (2) completely,
separating a hard cut from within-shot variation by +/-0.0002, which is why (2) is
measured explicitly rather than assumed.

Usage
-----
    python benchmark_keyframe_trigger.py
    python benchmark_keyframe_trigger.py --frames 24 --res 64
"""

import argparse
import json
import os
import time

import numpy as np

from fast_diffusion.model.density import SequentialDensityEstimator
from fast_diffusion.model.kfp import image_channel_to_samples


def make_sequence(n_frames, res, cut_at=None, seed=0, drift=0.01):
    """A drifting textured sequence, optionally with a hard content change.

    The drift is deliberately small relative to the cut so the two are genuinely
    different in magnitude rather than merely different in kind.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float64)

    def pattern(r):
        base = 0.25 * (np.sin(2 * np.pi * xx / max(res / 6.0, 1))
                       * np.cos(2 * np.pi * yy / max(res / 5.0, 1)))
        cy, cx = r.uniform(0.2, 0.8, 2) * res
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (res / 8.0) ** 2)))
        return base + r.uniform(0.4, 1.0) * blob

    frames = []
    cur = pattern(rng)
    for t in range(n_frames):
        if cut_at is not None and t == cut_at:
            cur = pattern(rng)  # hard change: new structure, new intensity histogram
        frames.append(np.clip(cur + rng.normal(0, drift, cur.shape), 0, None))
    return frames


def run_sequence(frames, force_all, **est_kwargs):
    """Estimate the log-density for every frame; return per-frame info and timing."""
    est = SequentialDensityEstimator(**est_kwargs)
    infos, times = [], []
    for f in frames:
        xy = image_channel_to_samples(f)
        t0 = time.perf_counter()
        _, info = est.estimate(xy, force_keyframe=force_all)
        times.append(time.perf_counter() - t0)
        infos.append(info)
    return est, infos, np.array(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    cut_at = args.frames // 2
    seq_cut = make_sequence(args.frames, args.res, cut_at=cut_at, seed=0)
    seq_flat = make_sequence(args.frames, args.res, cut_at=None, seed=0)

    results = {"frames": args.frames, "res": args.res, "cut_at": cut_at}

    # ---------------------------------------------------------------- 1. cost
    print("=" * 76)
    print("1. COST: incremental correction vs full re-estimation every frame")
    print("=" * 76)
    _, inc_infos, inc_t = run_sequence(seq_flat, force_all=False)
    _, _, full_t = run_sequence(seq_flat, force_all=True)

    # Frame 0 is a full estimate in both runs; comparing it would dilute the effect.
    inc_steady, full_steady = inc_t[1:], full_t[1:]
    n_key = sum(1 for i in inc_infos if i["keyframe"])
    print(f"  incremental : {inc_steady.mean()*1000:7.1f} ms/frame  "
          f"({n_key} keyframe(s) of {args.frames} frames)")
    print(f"  full        : {full_steady.mean()*1000:7.1f} ms/frame")
    speedup = full_steady.mean() / max(inc_steady.mean(), 1e-12)
    print(f"  --> {speedup:.2f}x cheaper per frame on a no-cut sequence")
    results["cost"] = {
        "incremental_ms": float(inc_steady.mean() * 1000),
        "full_ms": float(full_steady.mean() * 1000),
        "speedup": float(speedup),
        "keyframes": int(n_key),
    }

    # ------------------------------------------------------- 2. discrimination
    print()
    print("=" * 76)
    print("2. DISCRIMINATION: KL at the cut vs within-shot")
    print("=" * 76)
    # A threshold high enough never to fire, so every frame's KL is observable and
    # the statistic is measured rather than the decision it drives.
    _, infos, _ = run_sequence(seq_cut, force_all=False,
                               threshold_mode="absolute", kl_threshold=1e9)
    kls = np.array([i["kl"] for i in infos], dtype=float)
    esss = np.array([i["ess"] for i in infos], dtype=float)

    finite = np.isfinite(kls)
    at_cut = kls[cut_at] if finite[cut_at] else float("nan")
    within = kls[finite & (np.arange(len(kls)) != cut_at)]
    within = within[1:] if len(within) > 1 else within
    ratio = at_cut / max(within.mean(), 1e-30)

    print(f"  KL at cut (frame {cut_at}) : {at_cut:.4e}")
    print(f"  KL within shot (mean)     : {within.mean():.4e}  "
          f"(max {within.max():.4e})")
    print(f"  --> separation {ratio:8.1f}x   vs cut/max_within "
          f"{at_cut/max(within.max(),1e-30):.1f}x")

    # The rejected alternative, measured on the identical sequence.
    ess_f = np.isfinite(esss)
    ess_cut = esss[cut_at] if ess_f[cut_at] else float("nan")
    ess_within = esss[ess_f & (np.arange(len(esss)) != cut_at)]
    ess_within = ess_within[1:] if len(ess_within) > 1 else ess_within
    ess_ratio = ess_cut / max(ess_within.mean(), 1e-30)
    print(f"  ESS at cut {ess_cut:.4f} vs within {ess_within.mean():.4f}  "
          f"--> {ess_ratio:.3f}x  (why ESS was rejected)")

    results["discrimination"] = {
        "kl_at_cut": float(at_cut),
        "kl_within_mean": float(within.mean()),
        "kl_within_max": float(within.max()),
        "kl_ratio_mean": float(ratio),
        "kl_ratio_max": float(at_cut / max(within.max(), 1e-30)),
        "ess_at_cut": float(ess_cut),
        "ess_within_mean": float(ess_within.mean()),
        "ess_ratio": float(ess_ratio),
    }

    # ------------------------------------------------------- 3. false positives
    print()
    print("=" * 76)
    print("3. FALSE POSITIVES on a no-cut control, and detection on the cut")
    print("=" * 76)
    _, ctrl_infos, _ = run_sequence(seq_flat, force_all=False)
    _, cut_infos, _ = run_sequence(seq_cut, force_all=False)

    # Frame 0 is always a keyframe (no prior grid), so it is excluded from both counts.
    ctrl_fp = [i["frame"] for i in ctrl_infos[1:] if i["keyframe"]]
    cut_fired = [i["frame"] for i in cut_infos[1:] if i["keyframe"]]
    detected = cut_at in cut_fired

    print(f"  no-cut control : {len(ctrl_fp)} false positive(s) in "
          f"{args.frames - 1} frames  {ctrl_fp}")
    print(f"  cut sequence   : fired at {cut_fired}; cut at {cut_at} "
          f"{'DETECTED' if detected else 'MISSED'}")
    results["false_positives"] = {
        "control_false_positives": ctrl_fp,
        "control_frames": args.frames - 1,
        "cut_fired_at": cut_fired,
        "cut_detected": bool(detected),
    }

    print()
    print("=" * 76)
    verdict = []
    if speedup <= 1.0:
        verdict.append("COST: incremental correction is NOT cheaper -- the mechanism "
                       "does not pay for itself")
    if not (ratio > 5):
        verdict.append("DISCRIMINATION: KL does not separate the cut by >5x")
    if not detected:
        verdict.append("DETECTION: the trigger MISSED a hard cut")
    if len(ctrl_fp) > 1:
        verdict.append(f"FALSE POSITIVES: {len(ctrl_fp)} on a no-cut control; "
                       "kl_floor is likely mistuned for this resolution")
    if verdict:
        print("PROBLEMS FOUND -- do not report the trigger as working:")
        for v in verdict:
            print(f"  - {v}")
    else:
        print("All three properties hold at this configuration.")
    print("=" * 76)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "keyframe_trigger_measurement.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
