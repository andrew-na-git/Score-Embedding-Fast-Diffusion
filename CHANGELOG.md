# Changelog

All work extending the image-based score-precomputation pipeline to video, for the
Eurographics full-paper submission. Newest first.

Every measured number below was produced by a script in this repository and can be
regenerated: `benchmark_solver.py`, `benchmark_stencil.py`,
`benchmark_keyframe_trigger.py`, `benchmark_density.py`, `run_video.py`. Where an
earlier claim was found to be wrong it is listed under **Corrections** with the
corrected value, rather than quietly replaced.

---

## Scope fixed to 2D dynamic video; warm start removed; FVD made comparable

### Removed

* **The cross-clip score warm start is gone** -- `compute_scores_clip`,
  `compute_scores_clip_torch`, `score_precompute`, `train_video`, `run_video.py`
  (`--measure-warm-start`) and `benchmark_warm_start.py`. It measured **1.12x** against
  a cold-start control (4.00 vs 4.50 iterations), which does not justify a mechanism,
  its config surface and its correctness caveats.

  The cause was structural, not a tuning failure: upwinding already converges in 4-5
  iterations from cold, so the stability work had consumed the headroom the warm start
  existed to exploit. The two contributions were in direct tension and the solver won.
  Recorded in the `train_video` module docstring and PLAN.md 8.3b so it is not
  reintroduced. **The KL keyframe trigger is independent of this and survives.**

* **The `r3d_18` FVD backend is gone.** `fvd()` no longer takes a `backbone` argument
  at all. It had a latent `KeyError: 'mean'` on
  `weights.transforms.keywords["mean"]` and had never actually been executed. Rather
  than fix a path whose output cannot be compared with any published number, the path
  was deleted: a non-comparable FVD is not a weaker result but a different quantity
  wearing the same name.

### Added

* **`benchmark_keyframe_trigger.py`** -- replaces `benchmark_warm_start.py` and
  measures the mechanism that survived, on three axes separately because a trigger
  needs all three:

  | property | measured |
  |---|---|
  | cost: incremental correction vs full re-estimate | 3.3 ms vs 24.5 ms/frame = **7.54x** |
  | discrimination: KL at cut vs mean within-shot | **33931x** |
  | discrimination: KL at cut vs *worst* within-shot frame | **2591x** |
  | false positives on a no-cut control | **0** in 23 frames |
  | detection | fires exactly once, at the cut |
  | ESS on the same sequence (the rejected trigger) | **0.871x** |

  This 7.54x is what the efficiency claim now rests on, in place of 1.12x.

* **`benchmark_stencil.py`** -- what upwinding buys and what it costs, in space. The
  discrete operator is applied to a smooth analytic field and compared against the
  exact continuous operator, which isolates truncation error without needing an exact
  PDE solution. Boundary cells are excluded from every norm, since Neumann ghost-node
  folding is first-order by construction and would otherwise report order 1 for both
  stencils.

  * **Observed spatial order: upwind 0.99, central 2.04.**
  * **Artificial diffusion** fitted by projection onto the Laplacian matches the
    `|v|h/2` prediction to 0.987-0.9998 across four grids. Reduced to config
    quantities: `D_num / D = |s| h / 2`. **`g` cancels, so sigma is irrelevant** -- the
    only levers are `diffusion.dh` and the score magnitude. At the shipped `dh = 1`
    with `|s| ~ 1` the artificial diffusion is **~50% of the physical diffusion**.
  * **Three distinct central-difference thresholds**, which are easy to conflate:
    M-matrix (sign) condition at `|s|h <= 2`; diagonal dominance to `~2.53`; actual
    divergence between 2.6 and 3.0. Upwind holds a margin of **exactly 1.0** at every
    magnitude tested including `|s| = 25`.

* **`fast_diffusion/model/evaluate_video.py`: FVD on canonical Kinetics-400 I3D** --
  `load_i3d`, `i3d_features`, `DEFAULT_I3D_PATH`, `MIN_I3D_FRAMES`. 400-d pre-softmax
  logits, the layer FVD is defined on, confirmed by assertion. Preprocessing is
  delegated to the module's own `rescale`/`resize` flags rather than reimplemented,
  because FVD is as sensitive to preprocessing as to weights.

  Validated: ordering same-process 2.14 < shifted 89.78 < corrupted 947.0; identical
  sets -7.6e-06. Guards verified to fire for sample count < 64, T < 9, and missing
  weights.

* **`data/VideoDataset.py`: `DavisVideoDataset`** -- DAVIS 2017 with real per-object
  segmentation masks, wired in as `dataset: davis` with `davis_inpaint.yml`. 70 clips
  of 16 frames load with 0 skipped; mask coverage ranges **0.3% to 54.6%** against a
  synthetic box's fixed ~6%. Exposes `real_mask`, `sequence_name` and `mask_report`.

  Masks are resized **nearest-neighbour, never bicubic**. Bicubic on a DAVIS
  annotation was measured to produce fractional boundary values ranging to
  `[-0.166, 1.173]` -- outside `[0, 1]` entirely -- which would put every masked
  metric on the wrong support.

* **`download_assets.py`** -- fetches and verifies the I3D weights (sha256
  `bec6519f66ea534e...`, feature width asserted at 400) and DAVIS 2017 480p (sha256
  `e3d0b5b77c3d031b...`, 90 sequences / 6208 frames, clearing `MIN_FVD_SAMPLES = 64`).
  `assets/` added to `.gitignore`; without it 1.64 GB would have been committed.

* **`run_video.py`: `build_mask` prefers a dataset-supplied real mask** over a
  synthetic one and returns the source name, which is recorded on every results row.
  Masked metrics are not comparable across mask families, so a table mixing a 6%
  synthetic box with a 55% DAVIS object mask says nothing.

### Changed -- scope

* **Scope fixed to 2D dynamic video (`T x H x W`) throughout.** PLAN.md 7 moves from
  "deferred" to decided-out-of-scope, and 1 gains an explicit scope paragraph.
  README gains a video section stating it. `fp_video`, `fp_torch` and `network3d`
  docstrings now say that "3D" means the number of axes in the linear system or the
  use of `Conv3d`, never volumetric data -- the FP grid is (time, value, value), and
  two of those axes are pixel-*value* axes rather than spatial ones.

* **PLAN.md 1** no longer lists cross-frame warm starts as a core contribution, and
  no longer describes the keyframe trigger as ESS-based. The risk table and milestone
  table follow.

* **`synth_contiguous.yml` and `SyntheticVideoDataset.contiguous`** retargeted from
  the deleted warm start to the density estimator's incremental correction, which is
  what `contiguous` actually affects now.

### Corrections

* **`scipy.linalg.sqrtm` no longer accepts `disp`**, so `_frechet_distance` raised
  `TypeError` outright -- FVD and KID were both broken on any current SciPy. Now
  handles either return convention, adds an ill-conditioning offset, and **refuses**
  a large imaginary component rather than silently taking the real part.

* **I3D `.view()` failure.** The module's resize path calls `.view()`, which fails on
  the non-contiguous tensor a `permute` produces. Fixed with an explicit
  `.contiguous()`; without it every I3D feature call raised.

* **`benchmark_stencil` divergence test was too permissive.** `np.isfinite` reported a
  solve returning **1.79e+143** as successful. Replaced with `mx < 10 * |rhs|.max()`,
  which is justified because the FP step is a contraction on a non-negative rhs.

* **`benchmark_stencil` column was mislabelled.** `predicted_central_ok` predicted the
  M-matrix sign condition, not solvability -- central still solves at `|s| = 2.1` with
  a positive margin. Renamed `m_matrix_predicted`, and all three thresholds are now
  reported separately.

* **A DAVIS check of my own was vacuous.** The `objects='all'` vs `'largest'`
  comparison ran on `bear`, a single-object sequence, where the two are the same set
  by definition and agreed trivially. Rerun on `bike-packing` (2 objects), where
  `'largest'` correctly selects a strict subset: 0.1002 vs 0.1649 coverage.

---

## Video extension

### Added -- solver

* **`fast_diffusion/model/fp_video.py`** — numpy reference solver for the
  spatio-temporal Fokker-Planck equation on a (T, H, W) grid. Full-diagonal line
  Gauss-Seidel relaxation, upwind/central/legacy stencils, Neumann boundaries by
  ghost-node folding, batched Thomas kernel, stability diagnostics
  (`stability_margin`, `max_stable_score`, `check_config_stability`), and
  `splitting_error` / `unsplit_solve` to verify the relaxation converges to the exact
  unsplit solution (measured 4.6e-9).
* **`fast_diffusion/model/fp_torch.py`** — GPU backend mirroring `fp_video`. Kept as a
  separate implementation rather than an abstraction over both array libraries, and
  pinned to the reference by `tests_equivalence()`, which reproduces it **bit-for-bit**
  (rel diff 0.000e+00 across coefficients, diagonal, operator application, both
  stencils' solves and the gradient).
* **`benchmark_solver.py`** — regenerates the solver cost tables with repeats and
  medians. Single-shot timings on this workload vary by >2x between runs, so single
  measurements are not reportable.

### Added -- sequential importance sampling

* **`fast_diffusion/model/density.py`** — three real density estimators
  (`histogram`, `scipy`, `sklearn`) behind `estimate_log_density`, plus
  `SequentialDensityEstimator`, which propagates a running log-density grid across
  frames with an importance correction and triggers a full re-estimate (keyframe) on
  KL divergence. `effective_sample_size` is retained as a **diagnostic only**.
* **`benchmark_warm_start.py`** *(since REMOVED — see the top entry)* — measured the cross-clip warm start against a
  cold-start control, on both contiguous and independent clips.

### Added -- inpainting (the target task)

* **`fast_diffusion/model/inpaint.py`** — mask generation (`static_box`,
  `moving_box`, `stroke`, `dilate_mask`, `mask_coverage`) and the masked
  probability-flow sampler (`pf_ode_inpaint`, `autoregressive_inpaint`), with
  `tests_masked_sampling()` asserting the invariants that make any reported number
  trustworthy: an all-zero mask returns the reference exactly, and observed pixels are
  preserved bit-exactly for any mask.
* **Masked metrics in `evaluate_video.py`** — `masked_psnr`, `masked_mse`,
  `masked_lpips`, `seam_error`, `masked_warping_error`, `evaluate_inpainting`.
  `evaluate_inpainting` also reports whole-frame PSNR and mask coverage, and warns if
  observed pixels deviate at all.
* **`run_video.py`** — pipeline driver: precompute, fit, inpaint, evaluate. Runs two
  baselines by default (`per_frame`, `copy_prev`) and prints an explicit warning when
  the temporal path fails to beat the per-frame baseline.
* **Configs** in `fast_diffusion/configs/video/`: `synth_inpaint.yml`,
  `synth_inpaint_cut.yml` (positive control for the keyframe trigger),
  `synth_contiguous.yml` (positive control for the warm start), `folder_inpaint.yml`.

### Added -- supporting modules

* `fast_diffusion/model/flow.py` — RAFT / block-matching / identity flow,
  forward-backward consistency masks, sample warping.
* `fast_diffusion/model/sample_video.py` — autoregressive clip sampling and
  `weight_schedule` (constant / decay / ramp), with a single-solve baseline.
* `fast_diffusion/model/score_store.py` — raw memmap, temporal-SVD and int8 stores for
  the score fields, which reach 2.0 GB per clip at 20x3x16x128x128.
* `fast_diffusion/model/train_video.py` — score precompute with warm starts, plus the
  video fitting loop.
* `fast_diffusion/model/evaluate_video.py` — FVD / KID / warping error / warped LPIPS
  with sample-count guards.
* `network/network3d.py` — (2+1)D factorised spatio-temporal UNet with
  zero-initialised temporal layers (verified: output difference with and without the
  temporal path is exactly 0.0e+00 at initialisation).
* `data/VideoDataset.py` — folder / video-file / synthetic clip datasets. The
  synthetic one exposes `true_flow()` and a `contiguous` mode.

### Changed

* **`fast_diffusion/model/kfp.py`** — `construct_A` now takes `bc` and `stencil`;
  Neumann boundaries are applied explicitly and the assembly is node-indexed.
* **`fast_diffusion/model/loss.py`** — `slice_wasserstein_loss` renamed to
  `denoising_score_matching_loss`; old name kept as a deprecated alias that warns.
  Added `video_score_matching_loss`.
* **`reproduce_all.py`** — three-valued run status, nonzero exit on failure, `--force`.
* **`README.md`** — encoding repaired (the file was invalid UTF-8 mojibake); the
  density-estimator section rewritten around measured numbers.
* **`requirements.txt`** — added `lpips`, `Pillow`, `av`; added `requirements-lock.txt`.

### Removed

* `evaluate_fid.py` — computed FID over as few as 1-3 images. FID bias scales roughly
  as 1/n and its covariance term is not well conditioned below a few thousand samples,
  so the reported 0.65 / 1.53 / 4.30 carried no information about sample quality.
* Old `PLAN.md`, `SUMMARY.md`, all of `figures/`, `reproduction_summary.csv`, and
  notebook outputs — cleared for the new experiment campaign.
  (`latex_preprint_score_embedding/Figures/` was deliberately **kept**; it is the
  archived preprint source, not experimental output.)

---

## Bugs found and fixed

Listed because several were silent — they produced plausible numbers rather than
errors, which is the failure mode that reaches publication.

| Area | Defect | Consequence | Fix |
|---|---|---|---|
| `kfp.construct_A` | `sparse.diags(a[1:], 1)` coupled each row's end to the next row's start, and the row stride used `H` where the array stride is `W` | wrong boundary conditions everywhere; a linear ramp's gradient came out as 0.5 and -7 instead of 1.0 | explicit Neumann folding; ramp gradient now exactly 1.0 |
| `kfp` assembly | neighbour-indexed off-diagonals (`A[p,p+1] = a[p+1]`) | row `p` mixed drift from three different nodes; upwinding produced `inf` and a stability margin of -89.3 | node-indexed assembly; margin exactly 1.0 |
| splitting scheme | per-factor diagonal ownership: each factor carried `1 + diag_extra` while the operator diagonal is `1 + 3*diag_extra` | line relaxation diverged, relative error 2.5e+91 | full-diagonal line relaxation |
| Strang splitting | `(I + L/2)^-2 != (I + L)^-1` at second order | measured *worse* than plain ADI (2.1e-2 vs 1.8e-2) | removed rather than shipped |
| `thomas_batch` | no pivot guard | silent NaN propagation | `check=True` with a diagnostic `LinAlgError` |
| fixed-point iteration | no stability constraint | divergence to max abs 7.4e+68; `cifar1.yml` (sigma=5, N=20 → sigma^2 dt = 1.25) was outside the stable region | upwinding, which is unconditionally stable here |
| `diffusion.kde_method` | set in three ablation configs, read by nothing — `score_samples` never received `config` | the three `celeb1_kde_*` ablations were identical runs | real dispatch via `density.py` |
| keyframe trigger | ESS-based | **blind**: separated a hard scene cut from within-shot variation by ±0.0002, because ESS measures weight *variance*, which stays negligible when the log-ratio is near-constant across the support | KL divergence; ~300x separation |
| keyframe trigger | purely adaptive median threshold | 3 spurious keyframes on the no-cut control, as the median baseline fell to ~1e-6 | `max(kl_factor * baseline, kl_floor)` |
| `benchmark_density.py` | no warm-up call | lazy `scipy.stats` import (~10 s) landed in the first timed measurement, reporting 10,056 ms for a 64-sample problem | warm-up call before timing |
| `reproduce_all.py` | skipped runs returned `True` with `wall_time = timing.get("total", 0)` | skipped runs reported as instantaneous successes; matches the committed CSV of all-zero wall times | three-valued `status`, empty wall time for skips, nonzero exit on failure |
| `network3d.VideoNet.forward` | tiled `labels` and `clip_idx` across the flattened `B*T` batch but not `frame_idx` | outright shape error (24 vs 6) whenever a caller passed a per-frame index vector | tile `frame_idx` with `repeat(b)` — note `repeat`, not `repeat_interleave`, given the flattening order |
| `fp_torch.tests_equivalence` | compared the `central` stencil at a score magnitude where it diverges | **vacuous pass**: both backends diverged identically, their difference was exactly 0, the denominator overflowed to `inf`, and `0/inf` reported as a perfect match | each stencil tested inside its stable range, plus an explicit finiteness assertion |
| `fp_torch.thomas_batch_torch` | device-side pivot check inside the recursion | a device-to-host sync per step serialised the pipeline, costing ~2.5x | off by default; redundant under `upwind`, which is a diagonally dominant M-matrix |
| `fp_torch.fp_solve_torch` | precision warning fired at exactly the tolerance it recommended | contradictory advice | threshold lowered to `2*eps` and the message corrected |
| `evaluate_video.masked_psnr` | divided by zero on an exact fill | `ZeroDivisionError` on the best possible input | returns `inf`, documented as non-averageable |
| `inpaint.stroke_mask` | control points drawn uniformly over the frame | strokes traversed the frame, giving 28% coverage against the box masks' 6% — inpainting error is not comparable across coverages | local random walk; coverage now in the same range |
| video configs | `flow_method: true` | YAML parses bare `true` as boolean `True`, so the downstream string comparison silently failed | value renamed to `ground_truth` |
| `train_video.score_precompute` | disqualified the warm start on *any* keyframe, including the unavoidable one at frame 0 | **warm starts never once engaged**, so the paper's core mechanism went unmeasured | separate the bootstrap keyframe from genuine boundary and interior cuts |
| `inpaint.pf_ode_inpaint` | no output range constraint | an underfitted network produced fills of order 10 against data in [0, 1] | `clamp_output`, applied identically to the method and every baseline |

---

## Corrections to previously claimed numbers

Each of these was stated in the repository or in an earlier revision of the plan, and
each is withdrawn in favour of a measurement.

* **"~250x faster than exact KDE at 64x64"** — the committed CSV timed exact KDE only
  to 32x32 and left 64x64 empty. Measured: **slower below 48x48** (0.37x at 32x32),
  crossover at 48x48, **6.2x at 64x64**. The histogram estimator's cost is constant in
  resolution; exact KDE is O(N²).
* **"187x speedup over `spsolve`"** — measured with the divergent per-factor scheme and
  a near-zero score field. Real figure: **26.6x** at n = 65,536.
* **"18 min per clip on CPU at 16x128x128"** — a per-solve extrapolation that assumed
  every solve uses the full sweep budget. Measured end-to-end: **5.0 min** at
  `solve_tolerance` 1e-4.
* **"Ada fp64 is 1/64 the fp32 rate, so float64 on GPU is slower than the CPU path"** —
  true of the hardware, false of this solver. fp64 costs **0.85-1.33x** fp32 here,
  which is itself the evidence that the solver is kernel-launch bound rather than
  arithmetic bound. fp32 remains the default because it is never slower, not because
  fp64 is unaffordable.
* **"Porting the sweep to torch should recover a large factor"** — size-dependent, and
  a **net loss** below ~64x64: 0.36x end-to-end at 8x32x32, 1.36x at 16x64x64, 5.36x at
  16x128x128. The backend now defaults to `auto` and picks numpy below 96 px.

---

## Open issues

* **The warm start's measured benefit is 1.12x in iterations** (4.00 vs 4.50 on
  contiguous clips), which is weak for a headline claim. The cause is structural:
  upwinding made the cold-start iteration converge in 4-5 iterations, leaving little
  for a warm start to save. The two contributions are in tension. See PLAN.md §8.3b for
  the three options; the recommended framing leads with unconditional stability, which
  is a strong and independently measured result.
* **The temporal path is not yet beating the per-frame baseline** (-0.09 dB masked PSNR
  at 60 epochs), and the trivial `copy_prev` baseline beats both by ~10 dB. Expected at
  this training budget — the temporal layers are zero-initialised and start at exactly
  zero contribution — but it is unresolved and no temporal-consistency claim can be made
  until it is. `run_video.py` prints a warning whenever this holds.
* **`kl_floor = 1e-3` is resolution- and clip-length-dependent.** Tuned at 64x64 / T=16;
  at 48x48 / T=8 it produced one false-positive interior keyframe (KL 0.00103). Should be
  expressed relative to the within-shot KL scale rather than as an absolute constant.
* **Upwinding is first-order in space**, introducing numerical diffusion of order
  |v|h/2. The cost must be quantified and reported, not just the stability benefit.
* **CUDA graph capture is not implemented.** A sweep issues on the order of a thousand
  small kernels and the solver is launch-bound, so this is where the remaining GPU
  headroom is.
* **FVD uses an `r3d_18` backbone**, so its values are **not comparable** with published
  I3D-based FVD. State the backbone next to any figure.
* **4D / dynamic-3D is deferred**; scope is 2D dynamic video (T x H x W).
