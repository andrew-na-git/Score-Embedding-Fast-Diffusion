# EUROGRAPHICS 2027 — Submission TODO

Tracking checklist for *Sequential Score Pre-computation for Video Diffusion via
Upwind Fokker–Planck Solves, KL-Triggered Keyframing, and Krylov-Projected
Physical Control*.

**Scope (contribution framing):** the pipeline is made practical by two
mechanisms on a stable FP-solver foundation — **(1) importance sampling** (KL
keyframe SIS) to speed up the KDE score solve, and **(2) a Krylov projection** to
impose physical control (flow-consistency) on the inpainting. Both are validated.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · **(BLOCKER)** = paper
cannot be submitted without it.

**Claim discipline (non-negotiable):** every speedup is fitting/pre-compute
wall-clock, never sampling; every quality comparison at matched wall-clock;
nothing implies volumetric/4D support (domain is 2D dynamic video, T×H×W).

---

## 0. Done (measured, written, reproducible)

- [x] Video FP solver: upwind + line Gauss–Seidel, matches unsplit to 4.6e-9
- [x] Solver feasibility: 26.6× over `spsolve` @16×64×64; GPU 5.36× e2e @128² (Table 1)
- [x] Unconditional stability: M-matrix margin exactly 1.0 (Table 2)
- [x] Artificial-diffusion cost measured: order 0.99 vs 2.04, `|s|h/2` to 4 digits (Eq. 4)
- [x] KL keyframe trigger: 7.54× / 2591× / 0 false positives (Table 3) — *speed pillar*
- [x] Krylov-projected control (flow-consistency): implemented (`constraints.py`,
      wired into `inpaint.py`, `run_video.py`, config) + validated (`tests_constraints`):
      adjoints exact (0 / 1e-8 fp32), known-region projection = blend (4.4e-16),
      masked flow residual 4.97e2 → 1.2e-12, observed pixels bit-exact (drift 0.0),
      warm start 15 → 3 CG iters (40× lower residual at 3-iter budget) — *control pillar* (Table 5)
- [x] Control augmentation → **constrained control problem**: KKT/saddle form + metric
      W (free_mask = W⁻¹) + Tikhonov regularised control (ridge λ⁻¹). Validated: ridge=0
      bit-matches hard projection; ridge 0.5 cuts control effort 16.9→12.1 and CG iters
      20→15. CG-convergence figure (`benchmark_krylov_control.py` → `figures/krylov_control_convergence.pdf`).
- [x] FVD made comparable (canonical Kinetics-400 I3D)
- [x] LaTeX skeleton compiles on stock `article` class, **8 pp**, 39 refs, 0 warnings

**Page budget to 10 pp (full paper):** currently 8 pp of written-and-real content
(3 validated mechanisms + control augmentation + assembly derivation + figure). The
last ~2 pp are already *scaffolded* and fill once the campaign lands: Table 4
(inpainting quality), the ablation sweep (incl. control on/off + λ⁻¹ sweep), the
user study, and the teaser + a qualitative results figure. No prose padding — the
remaining space is real results, not text.

---

## 1. Experiments — the critical gap

### 1.1 Headline quality result **(BLOCKER)**

> **Gate run (2026-08-25): GO.** Synthetic epoch sweep {60, 200, 500}, 2 clips,
> reduced sampler (n_steps=120), video vs per_frame:
> dPSNR −0.09 → +0.21 → +0.24 dB; masked-warp gain −2.8% → +6.7% → +8.2%.
> The 60-epoch negative was undertraining (zero-init temporal layers); the
> temporal advantage crosses zero and grows monotonically on both metrics.
> Caveat: single seed, synthetic constant-motion (copy_prev trivially wins at
> ~19.9 dB); the real verdict must come from DAVIS. Artifacts:
> `saves/video/gate{60,200,500}`.

- [ ] Full-training-budget inpainting run on DAVIS clips
- [ ] Full-training-budget run on synthetic clips
- [ ] Multi-seed (9, 42, 123) for statistical strength; single-seed gate is not enough
- [ ] Decide + document: positive result, or reported as honest negative
      (gate says positive trend; confirm it holds on DAVIS at full budget)
- [ ] Fill Table 4 (`tab:inpaint`): masked PSNR/SSIM, masked warping error, FVD, fit time
- [ ] Confirm primary metric is **masked warping error** (coherence), PSNR secondary

### 1.2 Baselines — matched wall-clock **(BLOCKER)**
- [ ] Internal: per-frame independent FP solve (ESS = 1)
- [ ] DPM-Solver++
- [ ] Consistency models
- [ ] Rectified flow
- [ ] ≥1 video-native baseline, labelled as trained/different setting

### 1.3 Ablation campaign (harness + configs exist, not run)
- [ ] KL threshold sweep incl. 0 (= always keyframe) → cost-vs-flicker headline figure
- [ ] Warp source: RAFT / block-matching / identity
- [ ] History length L ∈ {1, 2, 4}
- [ ] Stencil: upwind vs central (where stable)
- [ ] Grid spacing dh ∈ {1.0, 0.5, 0.25} (only lever on artificial diffusion)
- [ ] Line Gauss–Seidel vs unsplit reference
- [ ] Temporal-pair density coordinates (the global-statistic limitation)
- [ ] **Krylov control on/off**: masked warping error with vs without flow-consistency
      projection, at matched wall-clock; sweep `constraint_weight` ∈ {0, 0.5, 1.0}
      and `cg_maxiter`. (Framework validated; quality effect is the pending number.)

### 1.4 User study (EG deliverable)
- [ ] Design 2AFC on temporal stability (ours vs per-frame)
- [ ] Determine n, run, analyze
- [ ] Write up + release data

### 1.5 Supplemental video (effectively mandatory)
- [ ] Object-removal before/after reel; build incrementally, not at the end

---

## 2. Writing TODOs (in `main.tex`)

- [ ] Teaser: render object-removal strip, uncomment `\teaser`
- [ ] Pipeline figure: redraw for video path (frame axis, warped-proposal arrow, KL gate)
- [ ] Background §5: condense PDE derivation to ≤1 column; assembly → supplemental
- [ ] Related work: make fitting-vs-sampling orthogonality explicit
- [ ] Related work: add 1–2 recent (2024–2026) video-inpainting / zero-shot methods
- [ ] Implementation: final backbone, torch version, single-GPU budget, checkpointing @128²
- [ ] Setup: finalize clip list/counts, state seeds (9, 42, 123)
- [ ] Conclusion: one-sentence headline result — write ONLY once the number exists
- [ ] Acknowledgements: funding/compute

---

## 3. Citations (`references.bib`)

- [ ] Verify ~20 Section-B entries tagged `TODO(cite): verify` (year/venue/pages/authors):
      RAFT, FVD, I3D, DAVIS, consistency models, rectified flow, DPM-Solver++,
      Deep Image Prior, R(2+1)D, LPIPS, KID, SMC/ESS, numerics texts
- [ ] Verify 4 new control-scope entries tagged `TODO(cite): verify`:
      Saad (iterative methods), Hestenes–Stiefel (CG), RePaint (Lugmayr 2022), DPS (Chung 2023)
- [ ] Related work: add an optical-flow-consistent video editing/inpainting method as closest prior
- [ ] Add ≥1 recent video-diffusion inpainting/object-removal method as context baseline

---

## 4. Repository / reproducibility (reviewer-facing integrity)

- [ ] `saves/` is gitignored/absent → commit timing CSVs or ship a manifest
- [ ] Regenerate headline speedup table from a real run (skip-bug fixed in code; needs run)
- [ ] Confirm all figure-source scripts committed + re-runnable (`benchmark_*.py`)

---

## 5. Kit swap at submission (mechanical — grep `EG-KIT:`)

- [ ] Add `egpubDL.cls` + `eg-alpha-doi.bst` beside `main.tex`
- [ ] `\documentclass{egpubDL}`
- [ ] Replace generic author block with EG `\author[...]{\parbox...}` block
- [ ] Uncomment `\teaser` and `\begin{classification}` CCS block
- [ ] `\bibliographystyle{eg-alpha-doi}`

---

## Critical path

`1.1 (full-budget run)` → decides positive vs negative-result paper → gates
`1.2 baselines` and `1.3 ablations` → then `1.4 user study` + `1.5 video` →
finalize writing (§2), citations (§3), reproducibility (§4), kit swap (§5).
