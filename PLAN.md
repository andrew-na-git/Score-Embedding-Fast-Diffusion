# Reproducibility Companion Paper — Project Plan
**Paper:** "Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection" (arXiv:2511.17634)  
**Venue:** RRPR 2026 Workshop @ ICPR — Track 2 (Companion Reproducibility Paper)  
**Deadline:** July 6, 2026  
**Format:** Springer LNCS, 12–15 pages

---

## What the Original Paper Claims
1. **Score embedding speedup (21×–115× over DDPM):** Pre-compute log-density scores by solving the Fokker-Planck PDE (finite difference → sparse linear system `A·m = b`) and embedding via probability flow ODE. Gives the network exact score supervision instead of noisy estimates.
2. **Krylov projection speedup (15.8%–43.7% over SpSolve):** For N images, exploit structural similarity between FP matrices by building a Krylov subspace from a "seed" image and projecting all "target" images into it for a warm start.
3. **Quality under fixed budget:** Under same wall-clock time, method produces coherent images where DDPM produces noise (SSIM used as quality metric, FID explicitly omitted by authors).

## Reproducibility Focus
Reproduce **contribution 1 (FP score embedding)** as the core, since:
- BiCGSTAB / SpSolve are well-established algorithms — the solver itself is not novel
- The FP discretisation (5-point stencil, boundary conditions, KDE initialisation) is the paper-specific component
- The Krylov projection is an optimisation of that solve — BiCGSTAB is the valid fallback

**Our additional contribution:** Replace the scipy O(N²) KDE initialisation with a histogram FFT approach (O(N log N)), ~250× faster, with negligible effect on FP convergence quality.

---

## Status

### ✅ Completed
- [x] Seed control (`--seed` CLI, numpy + torch + cuda seeding) in `fast_diffusion/run.py` and `comparisons/run.py`
- [x] Timing instrumentation: `kde_init`, `fp_solve`, `fp_iterations`, `training`, `total` → `timing.csv` per run
- [x] Convergence logging: residual norm per FP iteration → `convergence_log.csv` per run
- [x] Fixed `fids` undefined variable bug in `fast_diffusion/model/train.py`
- [x] Fixed `v2.ToImageTensor()` → torchvision v2 API in `data/Dataset.py`
- [x] Fixed `torch.load(weights_only=...)` for PyTorch 2.6+ in both `create_report.py` files
- [x] KDE replaced: scipy `gaussian_kde` (O(N²)) → histogram + `ndimage.gaussian_filter` FFT (O(N log N)) in `fast_diffusion/model/kfp.py`
- [x] 12 parameter ablation configs (dh, N, tol, sigma sweeps) in `fast_diffusion/configs/ablations/`
- [x] 3 KDE ablation configs in `fast_diffusion/configs/ablations/`
- [x] `reproduce_all.py`: single-command reproduction across seeds
- [x] `analyze_results.py`: aggregate timing/convergence → LaTeX tables + matplotlib figures
- [x] `evaluate_fid.py`: FID per seed, pooled per config, global across all samples
- [x] `Dockerfile` + `.dockerignore`
- [x] `README.md` updated with full methodology, seed flags, ablation table, pipeline description
- [x] Rotation augmentation (`rotate_augment: true` in config) in `data/Dataset.py`
- [x] 18 main experiments run (6 configs × 3 seeds) with new histogram KDE
- [x] FID computed: per-seed + pooled per config + global (all configs × all seeds)
- [x] `figures/` generated: convergence.pdf, runtime_*.pdf, ablation_*.pdf, seeds_*.pdf, all_samples_grid.pdf

### 🔲 Remaining Experiments
- [ ] **Ablation runs:** `python reproduce_all.py --ablations-only --seeds 9 42 123`
  - dh sweep: cifar1_dh05, cifar1_dh2, cifar1_dh4
  - N sweep: cifar1_N5, cifar1_N10, cifar1_N50
  - tol sweep: cifar1_tol1e4, cifar1_tol1e6, cifar1_tol1e10
  - sigma sweep: cifar1_sigma3, cifar1_sigma10, cifar1_sigma25
  - KDE comparison: celeb1_kde_scipy, celeb1_kde_sklearn, celeb1_kde_histogram
- [ ] **Comparison baselines:** `python reproduce_all.py --comparisons-only`
  - DDPM: celeb1, celeb1_ddim, cifar1, cifar1_ddim, cifar3, inet3
  - Reproduce Table 1/2/3 speedup claims from original paper
- [ ] **Scaling benchmark:** Time FP solve (scipy KDE vs histogram KDE) vs image resolution N
  - Sizes: 8×8, 16×16, 32×32, 64×64, 128×128
  - Goal: generate O(N²) vs O(N log N) scaling plot for paper

### 🔲 Analysis
- [ ] Run `python analyze_results.py --latex` on completed ablation results
- [ ] Run `python evaluate_fid.py` after ablation runs to include ablation FID
- [ ] Compute SSIM/PSNR per experiment to reproduce Table 1–3 metrics
- [ ] Build speedup table: our method vs DDPM, compare to paper's claimed 21×–115×
- [ ] Build KDE timing table: scipy vs histogram across resolutions

### 🔲 Paper Writing (LNCS format, ~12–15 pages)
Target structure:

| Section | Content | Status |
|---|---|---|
| Abstract | Companion framing, FP focus, KDE contribution | 🔲 |
| 1. Introduction | Original paper summary, reproducibility motivation, our contributions | 🔲 |
| 2. Methodology | FP discretisation (eqs. 5–25), KDE initialisation, score embedding, training | 🔲 |
| 3. KDE Improvement | Scipy O(N²) → histogram FFT O(N log N), bandwidth (Scott's rule), timing comparison | 🔲 |
| 4. Reproducibility Setup | Hardware, seeds, configs, timing instrumentation, SSIM early stopping | 🔲 |
| 5. Experiments | Speedup vs DDPM (reproduce Tables 1–3), ablation sweeps, seed variance | 🔲 |
| 6. FID Study | Per-seed, per-config pooled, global FID; note on few-shot FID limitations | 🔲 |
| 7. Discussion | What reproduced, what didn't, sensitivity to dh/tol/sigma, seed stability | 🔲 |
| 8. Conclusion | Summary, KDE contribution, reproducibility findings | 🔲 |
| References | Cite arXiv:2511.17634, Na et al., Song et al., BiCGSTAB, DDPM | 🔲 |

---

## Key Results So Far

### FID Scores (feature=2048, 50 samples/experiment)
| Config | Seed 9 | Seed 42 | Seed 123 | Pooled |
|---|---|---|---|---|
| cifar1 | 234.9 | 218.1 | 412.7 | 452.8 |
| cifar3_unconditional | 201.4 | 180.7 | 182.2 | 306.3 |
| cifar3_conditional | 138.7 | 117.6 | 162.2 | 276.2 |
| celeb1 | 451.7 | 409.3 | 356.2 | 409.1 |
| celeb3_unconditional | 149.3 | 126.6 | 131.8 | 138.2 |
| inet3_conditional | 296.0 | 277.2 | 268.8 | 281.6 |

*Note: FID computed against tiled training images (no held-out test set in few-shot setting).*

### KDE Speedup (histogram FFT vs scipy)
- Old scipy KDE on celeb3 seed 9: ~37s KDE init, total ~8167s
- New histogram KDE: ~0.06s per image → ~250× speedup on KDE step
- FP solve quality: preserved (convergence logs show same iteration counts)

---

## File Structure
```
Score-Embedding-Fast-Diffusion/
├── fast_diffusion/
│   ├── model/
│   │   ├── kfp.py            ← FP solver + histogram KDE (main contribution)
│   │   ├── train.py          ← seed control, timing CSV
│   │   └── dataloader.py
│   ├── configs/
│   │   ├── *.yml             ← main experiment configs
│   │   └── ablations/        ← 15 ablation configs
│   ├── saves/                ← experiment outputs (timing.csv, convergence_log.csv, ...)
│   └── run.py                ← --seed CLI
├── comparisons/              ← DDPM/DDIM baselines
├── data/
│   └── Dataset.py            ← rotate_augment support
├── network/                  ← U-Net architecture (unchanged)
├── figures/                  ← generated plots and grids
├── reproduce_all.py          ← single-command reproduction
├── analyze_results.py        ← LaTeX tables + matplotlib figures
├── evaluate_fid.py           ← FID per-seed, per-config, global
├── Dockerfile
├── requirements.txt
├── README.md
└── PLAN.md                   ← this file
```

---

## Hardware
- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU (CUDA 12.4, compute 8.9)
- Python 3.13.9, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124
- Original paper: MacBook Air M4 CPU + NVIDIA V100 GPU

*Hardware difference is a reproducibility finding: results may differ from paper due to different GPU architecture and CPU-based score pre-computation vs GPU.*
