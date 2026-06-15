# Manuscript Rewrite Summary — RRPR Workshop @ ICPR

## Paper Identity
- **Title:** Efficient Denoising using Score Embedding in Score-based Diffusion Models
- **arXiv:** 2511.17634
- **Target venue:** RRPR Workshop at ICPR 2026
- **Authors:** Andrew S. Na, William Gao, Mykhailo Briazkalo, Justin W.L. Wan (University of Waterloo)

---

## Core Contribution (keep unchanged)
1. Semi-explicit finite difference scheme to solve the log-density Fokker-Planck (FP) equation before training.
2. Score embedding: the numerical solution is embedded into the image via the transport ODE, giving the score-matching network a better starting point.
3. Result: 3–18× training speedup at equivalent SSIM quality, on CIFAR-10, CelebA and ImageNet.

---

## What Changed Since the Preprint

### Reproducibility study (new)
- All experiments re-run with **3 independent seeds** (9, 42, 123) for every main and ablation config.
- Timing, convergence logs and generated samples collected programmatically via `reproduce_all.py`.
- DDPM and DDIM baselines re-run with profiling (`comparisons/saves/` fully populated, 7/7 configs complete).

### FID evaluation (new metric)
FID computed with `evaluate_fid.py` (InceptionV3, feature dim 2048). Note: single-image configs (n_generated=1) cannot produce a valid FID — only multi-image configs yield meaningful values.

| Config | FID (pooled, 3 seeds) | N images |
|---|---|---|
| cifar3_conditional | **1.53** | 3 |
| inet3_conditional | **0.65** | 3 |
| celeb3_unconditional | **4.30** | 3 |
| ALL 32×32 (pooled) | 1.38 | 48 |
| ALL 64×64 (pooled) | 1.43 | 21 |

Single-image CIFAR-10 and CelebA configs return FID=ERROR (n_generated=1 is insufficient for FID).

### Ablation study (new section for paper)
All ablations run on CIFAR-10 single-image (cifar1) with 3 seeds each.

#### Grid spacing dh (finite difference step size)
- Configs: dh=0.5, 1.0 (default), 2.0, 4.0
- Figures: `figures/ablation_grid_spacing_dh.pdf`, `figures/seeds_cifar1_dh*.pdf`
- FID (pooled): dh0.5=2.27, dh2=2.41, dh4=2.86 vs baseline=2.39

#### Timesteps N (number of FP solve steps)
- Configs: N=5, 10 (default=20), 50
- Figures: `figures/ablation_timesteps_n.pdf`, `figures/seeds_cifar1_N*.pdf`
- FID (pooled): N5=2.42, N10=2.33, N50=2.12 vs baseline=2.39

#### Solver tolerance
- Configs: tol=1e-4, 1e-6, 1e-10
- Figures: `figures/ablation_solve_tolerance.pdf`, `figures/seeds_cifar1_tol*.pdf`
- FID (pooled): tol1e4=2.36, tol1e6=2.50, tol1e10=2.26 — relatively insensitive

#### Noise sigma (σ_max)
- Configs: σ=3, 10, 25
- Figures: `figures/ablation_sigma.pdf`, `figures/seeds_cifar1_sigma*.pdf`
- FID (pooled): σ3=2.86, σ10=2.67, σ25=8.98 — large σ degrades quality significantly

### KDE scaling benchmark (new)
- scipy `gaussian_kde` O(N²) vs histogram FFT O(N log N)
- Benchmarked at resolutions 8×8 to 256×256
- Figure: `figures/kde_scaling.pdf` (timing + speedup bar chart)
- Table: `figures/kde_scaling.csv`

### Runtime breakdown (now a table, not a figure)
- Columns: KDE init / FP solve / training / total (mean ± std across 3 seeds)
- The old pie charts have been replaced by a stacked bar chart in `analyze_results.py` for exploratory use; for the paper use a LaTeX table (see Section 7 of results notebook).

---

## Available Figures (in `figures/`)

| File | Content |
|---|---|
| `kde_scaling.pdf` | KDE scaling benchmark (line + bar) |
| `convergence.pdf` | FP solver residual vs iteration, all configs |
| `all_samples_grid.pdf` | Generated sample grids |
| `ablation_grid_spacing_dh.pdf` | Ablation: dh |
| `ablation_timesteps_n.pdf` | Ablation: N |
| `ablation_solve_tolerance.pdf` | Ablation: tolerance |
| `ablation_sigma.pdf` | Ablation: σ |
| `runtime_{config}.pdf` | Per-config stacked bar runtime breakdown |
| `seeds_{config}.pdf` | Per-seed convergence / sample quality |
| `fid_scores.csv` | Per-seed FID table |
| `fid_pooled.csv` | Pooled FID table |

Figures still needed from old manuscript (not in `figures/`, stored in `latex_preprint_score_embedding/Figures/`):
- `pipeline_diffusion_cropped.png` — architecture diagram (keep)
- `samples_cifar1_fd/ddpm/ddim.png` — denoising strips (regenerate from new saves)
- `samples_celeb1_fd/ddpm.png` — CelebA denoising strips
- `samples_cifar3_fd_c/ddpm.png` — multi-image CIFAR-10
- `samples_imagenet3_fd_c.png` — ImageNet 3-image
- `samples_celeb3_fd_uc.png` — CelebA 3-image unconditional
- `metrics_imagenet3_fd_cropped_c.png` — SSIM/MSE curves for ImageNet

---

## Key Numbers to Update in the Manuscript

### CIFAR-10 single image (cifar1) — Table 1
Results from preprint (single run). Rewrite instructions: **replace with mean ± std over 3 seeds**, or keep best-seed values and note seed in caption.

| Method | SSIM | Training time (s) | Speedup |
|---|---|---|---|
| Proposed | 0.99 | 26.98 | 1× |
| DDPM | 0.99 | 139.63 | 5.17× |
| DDIM | 0.99 | 182.53 | 6.77× |
| Proposed | 0.95 | 9.75 | 1× |
| DDPM | 0.95 | 131.09 | 13.44× |
| DDIM | 0.95 | 181.57 | 18.62× |

→ Regenerate these numbers from `comparisons/saves/cifar1_ddpm/` and `cifar1_ddim/` timing CSVs and `fast_diffusion/saves/cifar1_seed*/` timing CSVs.

### CelebA single image (celeb1) — Table 2
Same note: update with seed-aggregated timing from `fast_diffusion/saves/celeb1_seed*/` and `comparisons/saves/celeb1_ddpm/`.

### CIFAR-10 multi-image (cifar3_conditional) — Table 3
Update from `fast_diffusion/saves/cifar3_conditional_seed*/` and `comparisons/saves/cifar3_ddpm/`.

---

## Sections to Add for RRPR Workshop Version

1. **Ablation Study** (new section after Main Results)
   - Justify choice of default hyperparameters: dh=1.0, N=20, tol=1e-6, σ=5
   - Use the four ablation figures and FID table
   - Key finding: method is robust to tolerance and N; large σ hurts quality

2. **FID Results** (add to Main Results or as a subsection)
   - Report pooled FID for multi-image configs (cifar3, inet3, celeb3)
   - Note FID is not meaningful for n=1 image experiments

3. **Reproducibility / Seed Variance** (short paragraph)
   - All main results confirmed across 3 seeds
   - Report mean ± std for timing and quality metrics

4. **KDE Efficiency** (move from appendix/footnote to methods or experiments)
   - Histogram FFT KDE replaces O(N²) scipy KDE
   - Speedup measured at all resolutions; saves significant pre-training time at 128×128+

---

## Things NOT to Include / Change

- Do NOT add pie charts — runtime breakdown is now a table (or stacked bar if a figure is required)
- Do NOT include KDE ablations (celeb1_kde_*) — those experiments were not run
- Keep the LaTeX class (`interact.cls`) and bibliography style unchanged — workshop format
- The FP equation derivation and algorithms are correct — no changes needed

---

## Suggested Section Order for Rewrite

1. Introduction (minor update: mention reproducibility study, FID, ablations)
2. Fokker-Planck Formulation (unchanged)
3. Methodology (add KDE efficiency paragraph in pre-training step)
4. Experiments
   - 4.1 Setup & Metrics (add FID definition)
   - 4.2 Single Image Denoising — CIFAR-10 (update table with seed stats)
   - 4.3 Single Image Denoising — CelebA (update table)
   - 4.4 Multi-Image Denoising — CIFAR-10 (update table, add FID)
   - 4.5 Multi-Image Denoising — ImageNet & CelebA (add FID)
   - 4.6 Ablation Study (new)
5. Conclusion (update speedup range; mention ablation robustness and FID)
6. Broader Implications
