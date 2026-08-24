"""Benchmark the spatio-temporal FP solver: numpy CPU vs torch GPU.

Produces the solver cost table that the paper's feasibility argument rests on, and
the two measurements that justify the implementation choices:

* channel batching -- the C colour channels are independent, so batching them into
  the tridiagonal batch dimension does identical arithmetic with 1/C the kernel
  launches;
* dtype -- on Ada hardware fp64 arithmetic runs at 1/64 the fp32 rate, so if fp64
  and fp32 take comparable time then arithmetic is not the bottleneck.

Single-shot timings on this class of workload vary by more than 2x between runs
(clock and thermal state), so every measurement is repeated and the median
reported. Run with --repeats 1 for a quick smoke test only.

Usage
-----
    python benchmark_solver.py --out-dir figures
    python benchmark_solver.py --repeats 5 --no-largest
"""

import argparse
import csv
import json
import os
import statistics
import time

import numpy as np

SHAPES = [(8, 32, 32), (16, 64, 64), (16, 128, 128), (32, 256, 256)]
SIGMA = 5.0
N_TIMESTEPS = 20
INNER_TOL = 1e-6
CHANNELS = 3


def _torch():
    import torch
    return torch


def median_time(fn, repeats, sync=None):
    """Median wall time of `fn` over `repeats` runs, after one warm-up run."""
    fn()
    if sync:
        sync()
    times = []
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


def solver_scaling(repeats, shapes, device):
    """numpy CPU vs torch GPU for a single volume, and the GPU/CPU crossover."""
    import torch

    from fast_diffusion.model import fp_torch, fp_video

    dt = 1.0 / N_TIMESTEPS
    dh_s, dh2_s = dt / 2, dt
    rows = []

    for T, H, W in shapes:
        rng = np.random.default_rng(0)
        rhs = rng.random((T, H, W))
        s = rng.standard_normal((T, H, W)) * 5

        u_np, t_np = median_time(
            lambda: fp_video.fp_solve(rhs, SIGMA, s, dh_s, dh2_s, scheme="line",
                                      stencil="upwind", tol=INNER_TOL),
            repeats,
        )

        rhs_t = torch.as_tensor(rhs, dtype=torch.float32, device=device)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        sync = torch.cuda.synchronize if device.type == "cuda" else None
        (u_t, info), t_gpu = median_time(
            lambda: fp_torch.fp_solve_torch(rhs_t, SIGMA, s_t, dh_s, dh2_s,
                                            tol=INNER_TOL, return_info=True,
                                            warn_precision=False),
            repeats, sync,
        )

        rel = float(np.linalg.norm(u_t.cpu().numpy() - u_np) / np.linalg.norm(u_np))
        rows.append({
            "shape": f"{T}x{H}x{W}",
            "nodes": T * H * W,
            "cpu_ms": round(t_np * 1000, 1),
            "gpu_ms": round(t_gpu * 1000, 1),
            "speedup": round(t_np / t_gpu, 2),
            "sweeps": info["sweeps"],
            "gpu_vs_cpu_rel_err": rel,
        })
        del rhs_t, s_t, u_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def channel_batching(repeats, shapes, device):
    """C sequential solves vs one batched solve. Results must be identical."""
    import torch

    from fast_diffusion.model import fp_torch

    dt = 1.0 / N_TIMESTEPS
    dh_s, dh2_s = dt / 2, dt
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    rows = []

    for T, H, W in shapes:
        rng = np.random.default_rng(0)
        rhs = torch.as_tensor(rng.random((CHANNELS, T, H, W)),
                              dtype=torch.float32, device=device)
        s = torch.as_tensor(rng.standard_normal((CHANNELS, T, H, W)) * 5,
                            dtype=torch.float32, device=device)

        def sequential():
            return torch.stack([
                fp_torch.fp_solve_torch(rhs[k], SIGMA, s[k], dh_s, dh2_s,
                                        tol=INNER_TOL, warn_precision=False)
                for k in range(CHANNELS)
            ])

        def batched():
            return fp_torch.fp_solve_torch(rhs, SIGMA, s, dh_s, dh2_s,
                                           axes=(1, 2, 3), tol=INNER_TOL,
                                           warn_precision=False)

        u_seq, t_seq = median_time(sequential, repeats, sync)
        u_bat, t_bat = median_time(batched, repeats, sync)
        rel = float(torch.linalg.vector_norm(u_bat - u_seq)
                    / torch.linalg.vector_norm(u_seq))
        rows.append({
            "shape": f"{T}x{H}x{W}",
            "sequential_ms": round(t_seq * 1000, 1),
            "batched_ms": round(t_bat * 1000, 1),
            "gain": round(t_seq / t_bat, 2),
            "rel_diff": rel,
        })
        del rhs, s, u_seq, u_bat
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def dtype_cost(repeats, shapes, device):
    """fp32 vs fp64 on device. Comparable timings => not arithmetic bound."""
    import torch

    from fast_diffusion.model import fp_torch

    dt = 1.0 / N_TIMESTEPS
    dh_s, dh2_s = dt / 2, dt
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    rows = []

    for T, H, W in shapes:
        rng = np.random.default_rng(0)
        rhs = rng.random((CHANNELS, T, H, W))
        s = rng.standard_normal((CHANNELS, T, H, W)) * 5
        got = {}
        for dtype in (torch.float32, torch.float64):
            rt = torch.as_tensor(rhs, dtype=dtype, device=device)
            st = torch.as_tensor(s, dtype=dtype, device=device)
            u, el = median_time(
                lambda: fp_torch.fp_solve_torch(rt, SIGMA, st, dh_s, dh2_s,
                                                axes=(1, 2, 3), tol=INNER_TOL,
                                                warn_precision=False),
                repeats, sync,
            )
            got[dtype] = (u.double(), el)
            del rt, st
            if device.type == "cuda":
                torch.cuda.empty_cache()

        u32, t32 = got[torch.float32]
        u64, t64 = got[torch.float64]
        rows.append({
            "shape": f"{T}x{H}x{W}",
            "fp32_ms": round(t32 * 1000, 1),
            "fp64_ms": round(t64 * 1000, 1),
            "fp64_over_fp32": round(t64 / t32, 2),
            "fp32_rel_err": float(torch.linalg.vector_norm(u32 - u64)
                                  / torch.linalg.vector_norm(u64)),
        })
        del got, u32, u64
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def end_to_end(shapes, device):
    """Full per-clip score precompute, GPU vs CPU. One run each -- these are slow."""
    import torch

    from fast_diffusion.model import fp_torch, fp_video

    config = {
        "diffusion": {"num_timesteps": N_TIMESTEPS, "sigma": SIGMA, "dh": 1.0,
                      "solve_tolerance": 1e-4, "max_fp_iterations": 10,
                      "stencil": "upwind", "inner_tolerance": INNER_TOL,
                      "max_inner_sweeps": 100},
        "misc": {"eps": 1e-5},
    }
    rows = []
    for T, H, W in shapes:
        rng = np.random.default_rng(0)
        m0 = rng.random((CHANNELS, T, H, W))

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        sc_g, info_g = fp_torch.compute_scores_clip_torch(
            config, m0, device=device, dtype=torch.float32)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_gpu = time.perf_counter() - t0

        # The CPU path at 32x256x256 takes hours; skip it and extrapolate instead.
        if T * H * W <= 262_144:
            t0 = time.perf_counter()
            sc_c, info_c = fp_video.compute_scores_clip(config, m0, scheme="line")
            t_cpu = time.perf_counter() - t0
            rel = float(np.linalg.norm(sc_g - sc_c) / np.linalg.norm(sc_c))
        else:
            t_cpu, rel, info_c = float("nan"), float("nan"), {"iterations": None}

        rows.append({
            "shape": f"{T}x{H}x{W}",
            "gpu_s": round(t_gpu, 1),
            "cpu_s": None if np.isnan(t_cpu) else round(t_cpu, 1),
            "speedup": None if np.isnan(t_cpu) else round(t_cpu / t_gpu, 2),
            "gpu_iters": info_g["iterations"],
            "cpu_iters": info_c["iterations"],
            "converged": info_g["converged"],
            "gpu_vs_cpu_rel_err": None if np.isnan(rel) else rel,
        })
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def show(title, rows):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not rows:
        print("  (no rows)")
        return
    keys = list(rows[0])
    widths = [max(len(k), max(len(str(r[k])) for r in rows)) for k in keys]
    print("  " + "  ".join(k.rjust(w) for k, w in zip(keys, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(r[k]).rjust(w) for k, w in zip(keys, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed runs per measurement; the median is reported")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--no-largest", action="store_true",
                    help="skip 32x256x256, which is out of scope for the paper")
    ap.add_argument("--skip-end-to-end", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force the CPU device")
    args = ap.parse_args()

    torch = _torch()
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    shapes = SHAPES[:-1] if args.no_largest else SHAPES
    small = [s for s in shapes if s[0] * s[1] * s[2] <= 262_144]

    print(f"device: {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()

    from fast_diffusion.model import fp_torch
    print()
    print("Equivalence of the torch backend against the numpy reference:")
    fp_torch.tests_equivalence(seed=0, shape=(6, 16, 16))

    results = {
        "solver_scaling": solver_scaling(args.repeats, shapes, device),
        "channel_batching": channel_batching(args.repeats, small, device),
        "dtype_cost": dtype_cost(args.repeats, shapes, device),
    }
    if not args.skip_end_to_end:
        results["end_to_end"] = end_to_end(small, device)

    show("SINGLE-VOLUME SOLVE: numpy CPU vs torch GPU (fp32)",
         results["solver_scaling"])
    show("CHANNEL BATCHING: C sequential solves vs one batched solve",
         results["channel_batching"])
    show("DTYPE COST ON DEVICE (fp64 arithmetic is 1/64 rate on Ada)",
         results["dtype_cost"])
    if "end_to_end" in results:
        show("END-TO-END PER-CLIP SCORE PRECOMPUTE", results["end_to_end"])

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in results.items():
        write_csv(os.path.join(args.out_dir, f"solver_{name}.csv"), rows)
    meta = {
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0)
                        if device.type == "cuda" else "cpu"),
        "repeats": args.repeats,
        "sigma": SIGMA,
        "num_timesteps": N_TIMESTEPS,
        "inner_tolerance": INNER_TOL,
        "channels": CHANNELS,
        "torch": torch.__version__,
    }
    with open(os.path.join(args.out_dir, "solver_benchmark_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print()
    print(f"wrote CSVs and metadata to {args.out_dir}/")


if __name__ == "__main__":
    main()
