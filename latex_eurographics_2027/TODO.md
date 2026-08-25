# EUROGRAPHICS 2027 — Submission TODO

Tracking checklist for *Sequential Score Pre-computation for Video Diffusion via
Upwind Fokker–Planck Solves and KL-Triggered Keyframing*.

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
- [x] KL keyframe trigger: 7.54× / 2591× / 0 false positives (Table 3)
- [x] FVD made comparable (canonical Kinetics-400 I3D)
- [x] LaTeX skeleton compiles on stock `article` class, 6 pp, 33 refs, 0 warnings

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
