"""
evaluate_fid.py — Compute FID scores across all experiments by loading saved models,
generating samples, and comparing against ground truth datasets.

Also compiles a visual grid of all generated samples for side-by-side comparison.

Usage:
    python evaluate_fid.py                              # Evaluate all saves
    python evaluate_fid.py --saves-dir fast_diffusion/saves
    python evaluate_fid.py --num-samples 50             # Generate more samples for FID
    python evaluate_fid.py --compile-only               # Only compile image grids, skip FID
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fast_diffusion"))
sys.path.insert(0, os.path.dirname(__file__))

from network.network import Net
from fast_diffusion.model.sample import unconditional_sample, conditional_sample
from torchmetrics.image.fid import FrechetInceptionDistance

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_experiment(save_folder):
    """Load model, config, and ground truth from a save folder."""
    model_path = os.path.join(save_folder, "model.pth")
    if not os.path.exists(model_path):
        return None

    state = torch.load(model_path, map_location=device, weights_only=False)
    config = state["config"]

    model = Net(config)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    dataset_path = os.path.join(save_folder, "dataset.npy")
    ground_truths = np.load(dataset_path) if os.path.exists(dataset_path) else None

    return {"model": model, "config": config, "state": state, "ground_truths": ground_truths}


def generate_samples(model, config, ground_truths, num_samples=1):
    """Generate samples from a trained model, one image at a time.

    Returns:
        numpy array of shape (n_images, C, H, W) — final-timestep images only.
    """
    sample_method = config["sample"]["type"]
    with torch.no_grad():
        if sample_method == "unconditional":
            n_images = config["data_loader"]["num_images"]
            final_images = []
            for i in range(n_images):
                samples, _ = unconditional_sample(model, config, img_idx=i)
                # samples: (timesteps, 1, C, H, W) — take last timestep, squeeze image dim
                final_images.append(samples[-1, 0])
            return np.stack(final_images, axis=0)  # (n_images, C, H, W)
        else:
            conditional_weight = config["sample"]["conditional_weight"]
            gt_tensor = torch.from_numpy(ground_truths) if isinstance(ground_truths, np.ndarray) else ground_truths
            final_images = []
            for i in range(len(gt_tensor)):
                single_gt = gt_tensor[i:i+1]
                samples, _ = conditional_sample(model, config, single_gt, conditional_weight, img_idx=i)
                final_images.append(samples[-1, 0])
            return np.stack(final_images, axis=0)  # (n_images, C, H, W)


def compute_fid(ground_truths_nchw, generated_nchw, feature_dim=2048):
    """Compute FID between ground truth and generated images using torchmetrics.

    Args:
        ground_truths_nchw: numpy (N, C, H, W) in [0, 1]
        generated_nchw:     numpy (M, C, H, W), any scale — will be normalised

    FID requires at least `feature_dim` images on each side for a reliable
    covariance estimate.  With feature_dim=64 that threshold is ~64 images;
    with 2048 you need thousands.  We tile the GT to match the generated count.
    Returns None if there are too few samples to compute FID.
    """
    if len(ground_truths_nchw) < 2 and len(generated_nchw) < 2:
        return None  # both sides have only 1 sample — cannot compute FID

    fid_metric = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(device)

    gt = torch.from_numpy(ground_truths_nchw).float()
    gen = torch.from_numpy(generated_nchw).float()
    # Normalise generated to [0, 1]
    gen = (gen - gen.min()) / (gen.max() - gen.min() + 1e-8)

    # Tile the smaller set so cardinalities are equal (avoids rank-deficient covariance)
    if gt.shape[0] < gen.shape[0]:
        reps = int(np.ceil(gen.shape[0] / gt.shape[0]))
        gt = gt.repeat(reps, 1, 1, 1)[: gen.shape[0]]
    elif gen.shape[0] < gt.shape[0]:
        reps = int(np.ceil(gt.shape[0] / gen.shape[0]))
        gen = gen.repeat(reps, 1, 1, 1)[: gt.shape[0]]

    fid_metric.update(gt.to(device), real=True)
    fid_metric.update(gen.to(device), real=False)
    score = fid_metric.compute().item()
    fid_metric.reset()
    return score


def compile_image_grid(results, output_path):
    """Create a grid comparing ground truth and generated samples across experiments."""
    n_experiments = len(results)
    if n_experiments == 0:
        return

    # Determine grid layout: each row is one experiment
    # Columns: ground truth images | generated images
    max_images = max(r["n_images"] for r in results)

    fig, axes = plt.subplots(n_experiments, max_images * 2,
                              figsize=(3 * max_images * 2, 3 * n_experiments),
                              squeeze=False)

    for row_idx, r in enumerate(results):
        gt = r["ground_truths"]
        gen = r["generated"]
        name = r["name"]
        n_img = r["n_images"]

        for col_idx in range(max_images):
            ax_gt = axes[row_idx, col_idx]
            ax_gen = axes[row_idx, max_images + col_idx]

            if col_idx < n_img and col_idx < len(gen):
                # Ground truth
                img_gt = np.clip(gt[col_idx].transpose(1, 2, 0), 0, 1)
                ax_gt.imshow(img_gt)
                if row_idx == 0:
                    ax_gt.set_title(f"GT {col_idx + 1}", fontsize=9)

                # Generated — gen is (n_images, C, H, W)
                img_gen = gen[col_idx]
                img_gen = (img_gen - img_gen.min()) / (img_gen.max() - img_gen.min() + 1e-8)
                img_gen = np.clip(img_gen.transpose(1, 2, 0), 0, 1)
                ax_gen.imshow(img_gen)
                if row_idx == 0:
                    ax_gen.set_title(f"Gen {col_idx + 1}", fontsize=9)
            else:
                ax_gt.axis("off")
                ax_gen.axis("off")

            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            ax_gen.set_xticks([])
            ax_gen.set_yticks([])

        # Row label
        axes[row_idx, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=80, va="center")

    fig.suptitle("Ground Truth vs Generated Samples", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved image grid: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID and compile sample images")
    parser.add_argument("--saves-dir", default="fast_diffusion/saves",
                        help="Directory containing experiment save folders")
    parser.add_argument("--output", default="figures",
                        help="Output directory for figures and FID results")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile image grids, skip FID computation")
    parser.add_argument("--feature-dim", type=int, default=2048,
                        help="InceptionV3 feature dimension for FID (64, 192, 768, or 2048). "
                             "Use 64 for small datasets (1-3 images); 2048 for large batches.")
    args = parser.parse_args()

    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all save folders with model.pth
    save_folders = []
    for entry in sorted(os.listdir(args.saves_dir)):
        full = os.path.join(args.saves_dir, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "model.pth")):
            save_folders.append((entry, full))

    if not save_folders:
        print(f"No experiments found in {args.saves_dir}")
        return

    print(f"Found {len(save_folders)} experiments in {args.saves_dir}")
    print(f"Using device: {device}")

    results = []
    fid_rows = []

    for name, folder in save_folders:
        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")

        exp = load_experiment(folder)
        if exp is None:
            print(f"  Skipping (no model.pth)")
            continue

        # Check if samples already exist
        samples_path = os.path.join(folder, "samples.npy")
        if os.path.exists(samples_path):
            print(f"  Loading existing samples from {samples_path}")
            raw = np.load(samples_path)
            # Support legacy format (timesteps, n_images, C, H, W) — extract last timestep
            if raw.ndim == 5:
                raw = raw[-1]  # (n_images, C, H, W)
        else:
            print(f"  Generating samples...")
            raw = generate_samples(exp["model"], exp["config"], exp["ground_truths"])
            np.save(samples_path, raw)

        # raw is now (n_images, C, H, W) — final-timestep images
        gt = exp["ground_truths"]  # (n_train, C, H, W)
        n_train = gt.shape[0] if gt is not None else 0

        results.append({
            "name": name,
            "ground_truths": gt,
            "generated": raw,       # (n_images, C, H, W) for image grid
            "generated_all": raw,   # same — used for pooled FID
            "n_images": n_train,
        })

        # FID using last-timestep generated images only
        if not args.compile_only and gt is not None:
            fid_score = compute_fid(gt, raw, feature_dim=args.feature_dim)
            if fid_score is not None:
                print(f"  FID ({len(raw)} gen vs tiled GT, "
                      f"feature={args.feature_dim}): {fid_score:.4f}")
                fid_rows.append({"experiment": name, "fid": fid_score,
                                 "n_generated": len(raw), "n_train": n_train})
            else:
                print(f"  FID skipped: need >=2 samples (have {len(raw)} gen, {n_train} GT)")
                fid_rows.append({"experiment": name, "fid": "ERROR",
                                 "n_generated": len(raw), "n_train": n_train})

    # ------------------------------------------------------------------ #
    # Pooled FID per config (across all seeds combined)                   #
    # ------------------------------------------------------------------ #
    import re
    if not args.compile_only:
        config_groups = defaultdict(list)
        for r in results:
            base = re.sub(r"_seed\d+$", "", r["name"])
            config_groups[base].append(r)

        pooled_rows = []
        for base, group in config_groups.items():
            all_gen = np.concatenate([g["generated_all"] for g in group if g["generated_all"] is not None], axis=0)
            gt_any = group[0]["ground_truths"]
            n_seeds = len(group)
            pooled_fid = compute_fid(gt_any, all_gen, feature_dim=args.feature_dim)
            if pooled_fid is not None:
                print(f"  Pooled FID [{base}] ({len(all_gen)} imgs, {n_seeds} seeds): "
                      f"{pooled_fid:.4f}")
                pooled_rows.append({"config": base, "fid_pooled": pooled_fid,
                                    "n_generated": len(all_gen), "n_seeds": n_seeds})
            else:
                print(f"  Pooled FID [{base}]: skipped (too few samples)")

        # Global FID: only configs with matching spatial resolution
        try:
            # Group by spatial size to avoid cross-resolution concatenation
            by_size = defaultdict(list)
            for r in results:
                if r["generated_all"] is not None and r["ground_truths"] is not None:
                    h = r["generated_all"].shape[2]
                    by_size[h].append(r)
            for h, group in by_size.items():
                all_gen_global = np.concatenate([r["generated_all"] for r in group], axis=0)
                all_gt_global  = np.concatenate([r["ground_truths"]  for r in group], axis=0)
                global_fid = compute_fid(all_gt_global, all_gen_global, feature_dim=args.feature_dim)
                if global_fid is not None:
                    print(f"\n  GLOBAL FID ({h}x{h} images, all configs+seeds) "
                          f"({len(all_gen_global)} gen imgs, feature={args.feature_dim}): "
                          f"{global_fid:.4f}")
                    pooled_rows.append({"config": f"ALL_{h}x{h}", "fid_pooled": global_fid,
                                        "n_generated": len(all_gen_global), "n_seeds": len(group)})
        except Exception as e:
            print(f"  Global FID failed: {e}")

        if pooled_rows:
            pooled_path = os.path.join(output_dir, "fid_pooled.csv")
            with open(pooled_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["config", "fid_pooled",
                                                       "n_generated", "n_seeds"])
                writer.writeheader()
                writer.writerows(pooled_rows)
            print(f"\nPooled FID saved to: {pooled_path}")
            print(f"\n{'='*60}")
            print(f"POOLED FID (all seeds, feature={args.feature_dim})")
            print(f"{'='*60}")
            print(f"{'Config':<40} {'FID':>10} {'N gen':>8}")
            print("-" * 60)
            for r in pooled_rows:
                print(f"{r['config']:<40} {r['fid_pooled']:>10.4f} {r['n_generated']:>8}")

    # Save per-seed FID results
    if fid_rows:
        fid_path = os.path.join(output_dir, "fid_scores.csv")
        with open(fid_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "fid", "n_generated", "n_train"])
            writer.writeheader()
            writer.writerows(fid_rows)
        print(f"\nPer-seed FID scores saved to: {fid_path}")

        # Print summary table
        print(f"\n{'='*60}")
        print(f"PER-SEED FID (feature={args.feature_dim})")
        print(f"{'='*60}")
        print(f"{'Experiment':<40} {'FID':>10} {'N gen':>8}")
        print("-" * 62)
        for r in fid_rows:
            fid_str = f"{r['fid']:.4f}" if isinstance(r['fid'], float) else r['fid']
            print(f"{r['experiment']:<40} {fid_str:>10} {r['n_generated']:>8}")

    # Compile image grids
    if results:
        # Grid 1: All experiments
        grid_path = os.path.join(output_dir, "all_samples_grid.pdf")
        compile_image_grid(results, grid_path)

        # Grid 2: Group by config base (show seed variation)
        groups = defaultdict(list)
        for r in results:
            base = re.sub(r"_seed\d+$", "", r["name"])
            groups[base].append(r)

        for base, group in groups.items():
            if len(group) > 1:
                grid_path = os.path.join(output_dir, f"seeds_{base}.pdf")
                compile_image_grid(group, grid_path)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
