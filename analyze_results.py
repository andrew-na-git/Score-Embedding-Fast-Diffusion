"""
analyze_results.py — Aggregate experiment results and generate figures/tables for the paper.

Usage:
    python analyze_results.py                        # Analyze all results in fast_diffusion/saves
    python analyze_results.py --output figures       # Output directory for figures
    python analyze_results.py --latex                 # Also print LaTeX table markup
"""

import os
import csv
import argparse
import glob
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def find_save_folders(base_dir):
    """Find all save folders that contain model.pth."""
    folders = []
    for root, dirs, files in os.walk(base_dir):
        if "model.pth" in files:
            folders.append(root)
    return sorted(folders)


def load_timing(save_folder):
    """Load timing.csv if it exists."""
    path = os.path.join(save_folder, "timing.csv")
    timing = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        timing[row[0]] = float(row[1])
                    except ValueError:
                        timing[row[0]] = row[1]
    return timing


def load_convergence(save_folder):
    """Load convergence_log.csv if it exists."""
    path = os.path.join(save_folder, "convergence_log.csv")
    data = []
    if os.path.exists(path):
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "iteration": int(row["iteration"]),
                    "data_idx": int(row["data_idx"]),
                    "residual": float(row["residual"]),
                    "wall_time_s": float(row["wall_time_s"]),
                })
    return data


def load_metrics(save_folder):
    """Load metrics from model.pth."""
    path = os.path.join(save_folder, "model.pth")
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    result["config"] = state.get("config", {})
    result["diffusion_time"] = state.get("diffusion_time", None)
    result["train_time"] = state.get("train_time", None)
    metrics = state.get("metrics", {})
    losses = metrics.get("losses", None)
    if losses is not None:
        if isinstance(losses, np.ndarray):
            result["final_loss"] = float(np.mean(losses[:, -1]))
        elif isinstance(losses, list) and len(losses) > 0:
            result["final_loss"] = float(losses[-1])
    result["mse"] = metrics.get("mse", [])
    result["ssim"] = metrics.get("ssim", [])
    return result


def parse_folder_name(folder_name):
    """Extract config base name and seed from folder name like 'cifar1_seed42'."""
    m = re.match(r"^(.+?)_seed(\d+)$", folder_name)
    if m:
        return m.group(1), int(m.group(2))
    return folder_name, None


def group_by_config(save_folders):
    """Group save folders by config base name (stripping seed suffix)."""
    groups = defaultdict(list)
    for folder in save_folders:
        name = os.path.basename(folder)
        base, seed = parse_folder_name(name)
        groups[base].append({"folder": folder, "seed": seed, "name": name})
    return dict(groups)


# ── Table generation ──

def generate_main_results_table(groups, latex=False):
    """Generate main results table with mean±std across seeds."""
    rows = []
    for config_name, entries in sorted(groups.items()):
        # Skip ablation configs
        if any(x in config_name for x in ["_dh", "_N5", "_N10", "_N50", "_tol", "_sigma"]):
            continue
        losses, diff_times, train_times, total_times = [], [], [], []
        for entry in entries:
            m = load_metrics(entry["folder"])
            if m is None:
                continue
            if "final_loss" in m:
                losses.append(m["final_loss"])
            if m["diffusion_time"] is not None:
                diff_times.append(m["diffusion_time"])
            if m["train_time"] is not None:
                train_times.append(m["train_time"])
                if m["diffusion_time"] is not None:
                    total_times.append(m["diffusion_time"] + m["train_time"])

        row = {"config": config_name, "n_seeds": len(entries)}
        for label, vals in [("loss", losses), ("fp_time", diff_times),
                            ("train_time", train_times), ("total_time", total_times)]:
            if vals:
                row[f"{label}_mean"] = np.mean(vals)
                row[f"{label}_std"] = np.std(vals)
            else:
                row[f"{label}_mean"] = None
                row[f"{label}_std"] = None
        rows.append(row)

    # Print
    print("\n" + "=" * 80)
    print("MAIN RESULTS (mean ± std across seeds)")
    print("=" * 80)
    header = f"{'Config':<30} {'Seeds':>5} {'Loss':>16} {'FP Time (s)':>16} {'Train (s)':>16} {'Total (s)':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(key):
            m, s = r.get(f"{key}_mean"), r.get(f"{key}_std")
            if m is None:
                return "—"
            if s is not None and s > 0:
                return f"{m:.4f}±{s:.4f}"
            return f"{m:.4f}"
        print(f"{r['config']:<30} {r['n_seeds']:>5} {fmt('loss'):>16} {fmt('fp_time'):>16} {fmt('train_time'):>16} {fmt('total_time'):>16}")

    if latex:
        print("\n% LaTeX table")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Config & Loss & FP Solve (s) & Training (s) & Total (s) \\\\")
        print("\\midrule")
        for r in rows:
            def lfmt(key):
                m, s = r.get(f"{key}_mean"), r.get(f"{key}_std")
                if m is None:
                    return "---"
                if s is not None and s > 0:
                    return f"${m:.4f} \\pm {s:.4f}$"
                return f"${m:.4f}$"
            print(f"{r['config']} & {lfmt('loss')} & {lfmt('fp_time')} & {lfmt('train_time')} & {lfmt('total_time')} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    return rows


def generate_ablation_table(groups, param_name, config_pattern, extract_param, latex=False):
    """Generate ablation table for a specific parameter sweep."""
    rows = []
    for config_name, entries in sorted(groups.items()):
        if not re.search(config_pattern, config_name):
            continue
        param_val = extract_param(config_name, entries)
        losses, fp_times, fp_iters = [], [], []
        for entry in entries:
            m = load_metrics(entry["folder"])
            t = load_timing(entry["folder"])
            if m and "final_loss" in m:
                losses.append(m["final_loss"])
            if t:
                if "fp_solve" in t:
                    fp_times.append(t["fp_solve"])
                if "fp_iterations" in t:
                    fp_iters.append(t["fp_iterations"])
        row = {"param": param_val, "config": config_name, "n_seeds": len(entries)}
        for label, vals in [("loss", losses), ("fp_time", fp_times), ("fp_iters", fp_iters)]:
            if vals:
                row[f"{label}_mean"] = np.mean(vals)
                row[f"{label}_std"] = np.std(vals)
            else:
                row[f"{label}_mean"] = None
                row[f"{label}_std"] = None
        rows.append(row)

    rows.sort(key=lambda r: r["param"])

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {param_name}")
    print(f"{'=' * 60}")
    header = f"{'Value':>10} {'Seeds':>5} {'Loss':>16} {'FP Time (s)':>16} {'FP Iters':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(key):
            m, s = r.get(f"{key}_mean"), r.get(f"{key}_std")
            if m is None:
                return "—"
            if s is not None and s > 0:
                return f"{m:.4f}±{s:.4f}"
            return f"{m:.4f}"
        print(f"{r['param']:>10} {r['n_seeds']:>5} {fmt('loss'):>16} {fmt('fp_time'):>16} {fmt('fp_iters'):>10}")

    return rows


# ── Figure generation ──

def plot_convergence(groups, output_dir):
    """Plot FP residual convergence curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for config_name, entries in sorted(groups.items()):
        # Only plot main configs (first seed)
        if any(x in config_name for x in ["_dh", "_N5", "_N10", "_N50", "_tol", "_sigma"]):
            continue
        entry = entries[0]
        conv = load_convergence(entry["folder"])
        if not conv:
            continue
        # Aggregate: max residual per iteration
        iters = defaultdict(list)
        for c in conv:
            iters[c["iteration"]].append(c["residual"])
        its = sorted(iters.keys())
        residuals = [max(iters[i]) for i in its]
        ax.semilogy(its, residuals, label=config_name, marker="o", markersize=3)
        plotted = True

    if plotted:
        ax.set_xlabel("Fixed-Point Iteration")
        ax.set_ylabel("Residual (max over data)")
        ax.set_title("FP Solver Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, "convergence.pdf")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_runtime_breakdown(groups, output_dir):
    """Stacked bar chart of runtime breakdown for main configs."""
    for config_name, entries in sorted(groups.items()):
        if any(x in config_name for x in ["_dh", "_N5", "_N10", "_N50", "_tol", "_sigma"]):
            continue
        kde_times, fp_times, train_times = [], [], []
        for entry in entries:
            t = load_timing(entry["folder"])
            if t:
                kde_times.append(t.get("kde_init", 0))
                fp_times.append(t.get("fp_solve", 0))
                train_times.append(t.get("training", 0))
        if not kde_times:
            continue

        means = [np.mean(kde_times), np.mean(fp_times), np.mean(train_times)]
        stds  = [np.std(kde_times),  np.std(fp_times),  np.std(train_times)]
        if sum(means) == 0:
            continue

        labels = ["KDE Init", "FP Solve", "Training"]
        colors = ["#4c78a8", "#f58518", "#54a24b"]

        fig, ax = plt.subplots(figsize=(4, 3))
        bottom = 0.0
        for label, mean, std, color in zip(labels, means, stds, colors):
            ax.bar([config_name], [mean], bottom=bottom, label=label, color=color,
                   yerr=[[0], [std]], capsize=4, error_kw={"elinewidth": 1})
            bottom += mean
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Runtime Breakdown: {config_name}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, f"runtime_{config_name}.pdf")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close(fig)


def plot_ablation_sweep(groups, param_name, config_pattern, extract_param, output_dir):
    """Bar chart for an ablation sweep."""
    rows = []
    for config_name, entries in sorted(groups.items()):
        if not re.search(config_pattern, config_name):
            continue
        param_val = extract_param(config_name, entries)
        losses, fp_times = [], []
        for entry in entries:
            m = load_metrics(entry["folder"])
            t = load_timing(entry["folder"])
            if m and "final_loss" in m:
                losses.append(m["final_loss"])
            if t and "fp_solve" in t:
                fp_times.append(t["fp_solve"])
        if losses:
            rows.append({"param": param_val, "loss_mean": np.mean(losses), "loss_std": np.std(losses),
                          "fp_mean": np.mean(fp_times) if fp_times else 0,
                          "fp_std": np.std(fp_times) if fp_times else 0})
    if not rows:
        return

    rows.sort(key=lambda r: r["param"])
    params = [str(r["param"]) for r in rows]
    loss_means = [r["loss_mean"] for r in rows]
    loss_stds = [r["loss_std"] for r in rows]
    fp_means = [r["fp_mean"] for r in rows]
    fp_stds = [r["fp_std"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(params))
    ax1.bar(x, loss_means, yerr=loss_stds, capsize=4, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(params)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Final Loss")
    ax1.set_title(f"Loss vs {param_name}")

    ax2.bar(x, fp_means, yerr=fp_stds, capsize=4, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(params)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel("FP Solve Time (s)")
    ax2.set_title(f"FP Time vs {param_name}")

    fig.tight_layout()
    safe_name = param_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(output_dir, f"ablation_{safe_name}.pdf")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Param extractors for ablations ──

def extract_dh(config_name, entries):
    m = re.search(r"dh(\d+\.?\d*)", config_name)
    if m:
        val = m.group(1)
        return float(val.replace("05", "0.5")) if val == "05" else float(val)
    # baseline
    if entries:
        met = load_metrics(entries[0]["folder"])
        if met:
            return met["config"].get("diffusion", {}).get("dh", 1)
    return 1


def extract_N(config_name, entries):
    m = re.search(r"_N(\d+)", config_name)
    if m:
        return int(m.group(1))
    if entries:
        met = load_metrics(entries[0]["folder"])
        if met:
            return met["config"].get("diffusion", {}).get("num_timesteps", 20)
    return 20


def extract_tol(config_name, entries):
    m = re.search(r"tol(\d+e\d+)", config_name)
    if m:
        return float(m.group(1).replace("e", "e-"))
    if entries:
        met = load_metrics(entries[0]["folder"])
        if met:
            return float(met["config"].get("diffusion", {}).get("solve_tolerance", 2e-8))
    return 2e-8


def extract_sigma(config_name, entries):
    m = re.search(r"sigma(\d+)", config_name)
    if m:
        return int(m.group(1))
    if entries:
        met = load_metrics(entries[0]["folder"])
        if met:
            return met["config"].get("diffusion", {}).get("sigma", 5)
    return 5


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--saves-dir", default="fast_diffusion/saves",
                        help="Base directory containing save folders")
    parser.add_argument("--output", default="figures",
                        help="Output directory for figures")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table markup")
    args = parser.parse_args()

    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_folders = find_save_folders(args.saves_dir)
    if not save_folders:
        print(f"No results found in {args.saves_dir}")
        return

    print(f"Found {len(save_folders)} result folders in {args.saves_dir}")
    groups = group_by_config(save_folders)
    print(f"Grouped into {len(groups)} configs: {list(groups.keys())}")

    # Main results table
    generate_main_results_table(groups, latex=args.latex)

    # Ablation tables
    # Include baseline cifar1 in each ablation pattern
    generate_ablation_table(groups, "Grid Spacing (dh)", r"cifar1(_dh|$)", extract_dh, args.latex)
    generate_ablation_table(groups, "Timesteps (N)", r"cifar1(_N\d|$)", extract_N, args.latex)
    generate_ablation_table(groups, "Solve Tolerance", r"cifar1(_tol|$)", extract_tol, args.latex)
    generate_ablation_table(groups, "Sigma", r"cifar1(_sigma|$)", extract_sigma, args.latex)

    # Figures
    plot_convergence(groups, output_dir)
    plot_runtime_breakdown(groups, output_dir)
    plot_ablation_sweep(groups, "Grid Spacing (dh)", r"cifar1(_dh|$)", extract_dh, output_dir)
    plot_ablation_sweep(groups, "Timesteps (N)", r"cifar1(_N\d|$)", extract_N, output_dir)
    plot_ablation_sweep(groups, "Solve Tolerance", r"cifar1(_tol|$)", extract_tol, output_dir)
    plot_ablation_sweep(groups, "Sigma", r"cifar1(_sigma|$)", extract_sigma, output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
