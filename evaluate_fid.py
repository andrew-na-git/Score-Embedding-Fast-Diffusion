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
    """Generate samples from a trained model."""
    sample_method = config["sample"]["type"]
    with torch.no_grad():
        if sample_method == "unconditional":
            samples, n_eval = unconditional_sample(model, config)
        else:
            conditional_weight = config["sample"]["conditional_weight"]
            gt_tensor = torch.from_numpy(ground_truths) if isinstance(ground_truths, np.ndarray) else ground_truths
            samples, n_eval = conditional_sample(model, config, gt_tensor, conditional_weight)
    return samples  # shape: (timesteps, n_images, C, H, W)


def compute_fid(ground_truths, generated, feature_dim=2048):
    """Compute FID between ground truth and generated images using torchmetrics.

    Both inputs should be numpy arrays of shape (N, C, H, W) with values in [0, 1].
    Images are converted to uint8 tensors as required by FrechetInceptionDistance.
    """
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(device)

    # Ground truth — replicate if few images to get stable statistics
    gt = torch.from_numpy(ground_truths).float()
    if gt.shape[0] < 2:
        gt = gt.repeat(2, 1, 1, 1)  # FID needs at least 2 images

    gen = torch.from_numpy(generated).float()
    # Normalize to [0, 1]
    gen = (gen - gen.min()) / (gen.max() - gen.min() + 1e-8)

    if gen.shape[0] < 2:
        gen = gen.repeat(2, 1, 1, 1)

    fid.update(gt.to(device), real=True)
    fid.update(gen.to(device), real=False)

    score = fid.compute().item()
    fid.reset()
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

            if col_idx < n_img:
                # Ground truth
                img_gt = np.clip(gt[col_idx].transpose(1, 2, 0), 0, 1)
                ax_gt.imshow(img_gt)
                if row_idx == 0:
                    ax_gt.set_title(f"GT {col_idx + 1}", fontsize=9)

                # Generated
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
                        help="InceptionV3 feature dimension for FID (64, 192, 768, or 2048)")
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
            samples = np.load(samples_path)
            generated = samples[-1]  # last timestep = final output
        else:
            print(f"  Generating samples...")
            samples = generate_samples(exp["model"], exp["config"], exp["ground_truths"])
            generated = samples[-1]

        gt = exp["ground_truths"]
        n_images = gt.shape[0] if gt is not None else 0

        results.append({
            "name": name,
            "ground_truths": gt,
            "generated": generated,
            "n_images": n_images,
        })

        # FID
        if not args.compile_only and gt is not None:
            try:
                fid_score = compute_fid(gt, generated, feature_dim=args.feature_dim)
                print(f"  FID: {fid_score:.4f}")
                fid_rows.append({"experiment": name, "fid": fid_score, "n_images": n_images})
            except Exception as e:
                print(f"  FID failed: {e}")
                fid_rows.append({"experiment": name, "fid": "ERROR", "n_images": n_images})

    # Save FID results
    if fid_rows:
        fid_path = os.path.join(output_dir, "fid_scores.csv")
        with open(fid_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "fid", "n_images"])
            writer.writeheader()
            writer.writerows(fid_rows)
        print(f"\nFID scores saved to: {fid_path}")

        # Print summary table
        print(f"\n{'='*50}")
        print("FID SCORES")
        print(f"{'='*50}")
        print(f"{'Experiment':<40} {'FID':>10} {'Images':>8}")
        print("-" * 60)
        for r in fid_rows:
            fid_str = f"{r['fid']:.4f}" if isinstance(r['fid'], float) else r['fid']
            print(f"{r['experiment']:<40} {fid_str:>10} {r['n_images']:>8}")

    # Compile image grids
    if results:
        # Grid 1: All experiments
        grid_path = os.path.join(output_dir, "all_samples_grid.pdf")
        compile_image_grid(results, grid_path)

        # Grid 2: Group by config base (show seed variation)
        groups = defaultdict(list)
        import re
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
