"""
reproduce_all.py — Single-entry reproduction script for RRPR 2026 companion paper.

Runs all experiments (main + ablations) across multiple seeds, collects results,
and generates a summary CSV.

Usage:
    python reproduce_all.py                  # Run everything
    python reproduce_all.py --main-only      # Only main configs (no ablations)
    python reproduce_all.py --ablations-only # Only ablation configs
    python reproduce_all.py --seeds 9 42 123 # Custom seed list
    python reproduce_all.py --profile        # Enable MSE/SSIM profiling over time
"""

import subprocess
import sys
import os
import csv
import argparse
import time
from pathlib import Path

FAST_DIFFUSION_DIR = os.path.join(os.path.dirname(__file__), "fast_diffusion")
COMPARISONS_DIR = os.path.join(os.path.dirname(__file__), "comparisons")

MAIN_CONFIGS = [
    "cifar1.yml",
    "cifar3_unconditional.yml",
    "cifar3_conditional.yml",
    "celeb1.yml",
    "celeb3_unconditional.yml",
    "inet3_conditional.yml",
]

ABLATION_CONFIGS = [
    # Grid spacing (dh)
    "ablations/cifar1_dh05.yml",
    "ablations/cifar1_dh2.yml",
    "ablations/cifar1_dh4.yml",
    # Timesteps (N)
    "ablations/cifar1_N5.yml",
    "ablations/cifar1_N10.yml",
    "ablations/cifar1_N50.yml",
    # Solve tolerance
    "ablations/cifar1_tol1e4.yml",
    "ablations/cifar1_tol1e6.yml",
    "ablations/cifar1_tol1e10.yml",
    # Sigma
    "ablations/cifar1_sigma3.yml",
    "ablations/cifar1_sigma10.yml",
    "ablations/cifar1_sigma25.yml",
]

COMPARISON_CONFIGS = [
    "cifar1_ddpm.yml",
    "cifar1_ddim.yml",
    "cifar3_ddpm.yml",
    "cifar3_ddim.yml",
    "celeb1_ddpm.yml",
    "celeb1_ddim.yml",
    "inet3_ddpm.yml",
]

DEFAULT_SEEDS = [9, 42, 123]


def run_experiment(working_dir, config, seed=None, profile=False):
    """Run a single experiment. Returns (success, wall_time, save_folder)."""
    cmd = [sys.executable, "run.py", "--config", config]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if profile:
        cmd += ["--profile"]

    config_base = Path(config).stem
    folder_name = config_base
    if seed is not None:
        folder_name += f"_seed{seed}"
    save_folder = os.path.join(working_dir, "saves", folder_name)

    # Skip if already completed
    if os.path.exists(os.path.join(save_folder, "model.pth")):
        print(f"\n{'='*60}")
        print(f"SKIPPING (already complete): {folder_name}")
        print(f"{'='*60}")
        timing = collect_timing(save_folder)
        wall = float(timing.get("total", 0))
        return True, wall, save_folder

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"  Working dir: {working_dir}")
    print(f"  Save folder: {save_folder}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=working_dir, capture_output=False, text=True)
        wall = time.time() - t0
        success = result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        wall = time.time() - t0
        success = False

    return success, wall, save_folder


def collect_timing(save_folder):
    """Read timing.csv from a save folder if it exists."""
    timing_path = os.path.join(save_folder, "timing.csv")
    timing = {}
    if os.path.exists(timing_path):
        with open(timing_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    timing[row[0]] = row[1]
    return timing


def main():
    parser = argparse.ArgumentParser(description="Reproduce all RRPR experiments")
    parser.add_argument("--main-only", action="store_true", help="Run only main configs")
    parser.add_argument("--ablations-only", action="store_true", help="Run only ablation configs")
    parser.add_argument("--comparisons-only", action="store_true", help="Run only comparison baselines")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"Seeds to use (default: {DEFAULT_SEEDS})")
    parser.add_argument("--profile", action="store_true", help="Enable MSE/SSIM profiling")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # Determine which configs to run
    fd_configs = []
    if not args.comparisons_only:
        if not args.ablations_only:
            fd_configs += MAIN_CONFIGS
        if not args.main_only:
            fd_configs += ABLATION_CONFIGS

    comp_configs = []
    if not args.main_only and not args.ablations_only:
        comp_configs = COMPARISON_CONFIGS

    results = []
    total_start = time.time()

    # Fast diffusion experiments (with seed sweep)
    for config in fd_configs:
        for seed in args.seeds:
            if args.dry_run:
                print(f"[DRY RUN] cd {FAST_DIFFUSION_DIR} && python run.py --config {config} --seed {seed}")
                continue
            success, wall, save_folder = run_experiment(
                FAST_DIFFUSION_DIR, config, seed=seed, profile=args.profile
            )
            timing = collect_timing(save_folder) if success else {}
            results.append({
                "type": "fast_diffusion",
                "config": config,
                "seed": seed,
                "success": success,
                "wall_time": f"{wall:.2f}",
                "kde_time": timing.get("kde_init", ""),
                "fp_time": timing.get("fp_solve", ""),
                "fp_iters": timing.get("fp_iterations", ""),
                "train_time": timing.get("training", ""),
            })

    # Comparison baselines (single seed — they use their own config seeds)
    for config in comp_configs:
        if args.dry_run:
            print(f"[DRY RUN] cd {COMPARISONS_DIR} && python run.py --config {config}")
            continue
        success, wall, save_folder = run_experiment(
            COMPARISONS_DIR, config, profile=args.profile
        )
        results.append({
            "type": "comparison",
            "config": config,
            "seed": "",
            "success": success,
            "wall_time": f"{wall:.2f}",
            "kde_time": "",
            "fp_time": "",
            "fp_iters": "",
            "train_time": "",
        })

    if args.dry_run:
        return

    # Write summary
    total_wall = time.time() - total_start
    summary_path = os.path.join(os.path.dirname(__file__), "reproduction_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["type", "config", "seed", "success", "wall_time",
                       "kde_time", "fp_time", "fp_iters", "train_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"All experiments complete. Total wall time: {total_wall:.1f}s")
    print(f"Summary written to: {summary_path}")
    n_success = sum(1 for r in results if r["success"])
    print(f"Results: {n_success}/{len(results)} succeeded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
