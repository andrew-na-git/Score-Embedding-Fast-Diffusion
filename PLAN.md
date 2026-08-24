# Eurographics Full Paper -- Project Plan

**Working title:** Sequential Score Pre-computation for Video Diffusion via
Importance-Sampled Fokker-Planck Initialisation

**Venue:** Eurographics (full paper track, proceedings in Computer Graphics Forum)
**Status:** invited to submit a full paper
**Deliverables:** paper (CGF format), supplemental video (effectively mandatory),
code artifact, user study data

---

## 1. Framing

The method in this repository is **not** a generative model of a data
distribution. It is a **per-instance score pre-computation and fitting
accelerator**: scores are computed per training image by solving the log-density
Fokker-Planck (FP) equation, and the network is indexed by image identity
(`nn.Embedding(num_images)`). Reported SSIM is measured against the *training*
images, and the speedup is in *training/fitting* wall-clock, not sampling.

The Eurographics paper commits to that regime and extends it along the axis where
it has a genuine advantage no static-image method can claim: **temporal
continuation**.

**Scope: 2D dynamic video.** The domain is `T x H x W` -- two spatial dimensions
plus time -- throughout. The FP solve runs on a three-axis grid, but two of those
axes are pixel-*value* axes and the third is time; this is 2D imagery in motion, not
volumetric or 4D data. Dynamic 3D and 4D representations are explicitly out of scope
(section 7), and nothing in the paper should be phrased so that a reader infers
otherwise. The one place the wording matters most is the solver description: "3D
operator" in `fp_video` refers to the three grid axes of the linear system, not to
three spatial dimensions.

**Core contribution.** Consecutive video frames have nearly identical FP operators.
The mechanism that exploits this is:

**Sequential importance sampling with a KL-triggered keyframe rule.** Frame `k-1`'s
converged density is used as a proposal; particles are advected by the estimated
motion field and reweighted, so only the residual -- content the warped predecessor
cannot explain -- needs fresh estimation. Temporal coherence holds *by construction*,
not via a consistency penalty. A KL statistic on the importance weights decides when
the proposal has stopped being usable and a full re-estimate is required.

Measured (`benchmark_keyframe_trigger.py`, 24 frames at 64x64): the incremental
correction is **7.54x cheaper per frame** than re-estimating (3.3 ms vs 24.5 ms); the
KL statistic separates an injected cut from within-shot variation by **2591x** against
the worst within-shot frame (33931x on the mean); and it fires **zero** false
positives across 23 control frames while detecting the cut exactly. ESS, the
originally proposed trigger, moves by only 0.871x on the same sequence -- see 8.5.

**Removed: cross-frame warm starts.** An earlier revision also warm-started each
clip's fixed-point iteration from the previous clip's converged score field. Measured
against a cold-start control it was worth 1.12x, and it is gone. The cause was
structural rather than a tuning failure: upwinding already converges in 4-5 iterations
from cold, so the stability work had consumed the headroom the warm start existed to
exploit. Section 8.3b records the measurement. The KL trigger is independent of this
and survives.

**Claim discipline.** Every speedup claim must state that it is fitting-time,
and every comparison must be at matched wall-clock.

---

## 2. Corrections required before any new work

These are prerequisites. Two of them are integrity issues that would sink the
paper if a reviewer opened the repository.

### 2.1 Density-estimator claims (blocking)

- `diffusion.kde_method` is set in `configs/ablations/celeb1_kde_{scipy,sklearn,histogram}.yml`
  but **read by no code**. `score_samples()` never receives `config`, and
  `_kde_log_density` is hardcoded. Those three ablations would produce identical
  results under three names. There is no scipy or sklearn code path in the repo.
- `figures/kde_scaling.csv` and `.pdf` are **orphaned**: no committed script
  generates them.
- The README and the `_kde_log_density` docstring claim "~250x faster for 64x64".
  The only committed measurement says **2.5x at 32x32**, and 64x64 scipy was
  never timed. Below 24x24 the histogram estimator is *slower* (0.05x at 8x8),
  because the 256^2 grid cost is independent of image resolution.

**Action:** implement the real dispatch (`density.py`), commit the benchmark
generator (`benchmark_density.py`), and restate the claim honestly: *cost is
constant in image resolution, crossover at ~24x24, scipy intractable past 32x32*.

### 2.2 Boundary conditions (blocking)

`construct_A` builds `sparse.diags(a[1:], 1)` on the flattened `H*W` grid with no
boundary masking, so the end of each row is coupled to the start of the next; and
`dm[..., 0]` / `dm[..., -1]` substitute a hard zero outside the domain. In 2D this
is an edge artifact. With a temporal axis added it leaks density **across frame
boundaries**, producing exactly the flicker a graphics reviewer looks for in the
supplemental video. Proper Neumann conditions on the true grid are required
first.

### 2.3 Naming -- RESOLVED

`slice_wasserstein_loss` (`fast_diffusion/model/loss.py`) is standard weighted
denoising score matching, `(score * std + z)^2`. No slicing, no Wasserstein
distance. Renamed to `denoising_score_matching_loss`; the old name remains as a
deprecated alias that warns, and `train.py` uses the new name. The video equivalent
is `video_score_matching_loss`, which sums over (T, C, H, W) to match the image
path's sum-over-pixels convention -- averaging over T instead would rescale the loss
by 1/T and make learning rates non-transferable between the two paths.

### 2.4 Evaluation artifacts

Remove FID-over-3-images entirely (`evaluate_fid.py` deleted). FID is severely
biased below a few thousand samples; the reported 0.65 / 1.53 / 4.30 values are
not meaningful. `reproduction_summary.csv` records `wall_time = 0.00` for every
baseline and contains no `fast_diffusion` rows, so the headline speedup table is
not currently regenerable.

The cause is fixed: `reproduce_all.run_experiment` returned `True` with
`wall_time = timing.get("total", 0)` for runs it *skipped*, making skipped and
completed runs indistinguishable and reporting the skips as instantaneous successes.
It now returns a three-valued `status` (`ok` / `skipped` / `failed`), writes an empty
wall time for skipped runs rather than zero, exits nonzero on any failure, and takes
`--force` to re-run and actually time completed configs.

---

## 3. Method work

| # | Item | Module | Notes |
|---|---|---|---|
| M1 | Density-estimator dispatch | `fast_diffusion/model/density.py` | histogram / scipy / sklearn, config-driven |
| M2 | Neumann BCs, 2D | `fast_diffusion/model/kfp.py` | prerequisite for M4 |
| M3 | Optical flow + warping | `fast_diffusion/model/flow.py` | RAFT via torchvision; forward/backward consistency masks |
| M4 | Line-relaxation FP solve on (T,H,W) | `fast_diffusion/model/fp_video.py` | batched Thomas; `spsolve` on T*H*W is intractable. Full-diagonal line Gauss-Seidel, not per-factor ADI |
| M5 | Sequential IS initialiser + KL keyframing | `fast_diffusion/model/density.py` | replaces per-frame `score_samples`; ESS was measured blind and rejected (8.5) |
| ~~M6~~ | ~~Cross-frame Krylov warm starts~~ | -- | **Removed.** Measured at 1.12x and deleted; see 8.3b |
| M7 | Score-field storage | `fast_diffusion/model/score_store.py` | ~250 MB per 16-frame 256^2 clip at N=20; needs temporal compression or streaming |
| M8 | Spatio-temporal backbone | `network/network3d.py` | factorised (2+1)D conv + temporal attention |
| M9 | Autoregressive conditional sampler | `fast_diffusion/model/sample_video.py` | per-frame weight schedule replacing scalar `conditional_weight`; batched PF-ODE |

### Dimensionality caveat (must appear in the paper)

Importance sampling degenerates in high dimensions. We do **not** perform IS in
the full `H*W`-dimensional pixel space -- only on the low-dimensional
density-estimation coordinates. State this explicitly; a reviewer will otherwise
invoke the curse of dimensionality, correctly.

---

## 4. Experiments

### 4.1 Ablations (replaces the dh / N / tol / sigma sweeps)

| Axis | Values | Question answered |
|---|---|---|
| KL threshold | `kl_factor` sweep, plus 0 (= always keyframe) | cost vs. flicker tradeoff; headline figure |
| Warp source | RAFT / block matching / identity (no warp) | does motion-aware IS matter, or is reweighting enough |
| History length L | 1, 2, 4 | how much conditioning context is useful |
| Stencil | upwind / central (where central is stable) | the accuracy price of unconditional stability; measured in 8.3c |
| Grid spacing `dh` | 1.0, 0.5, 0.25 | artificial diffusion is `\|s\|dh/2`, so this is the only lever on it |
| Relaxation | line Gauss-Seidel / unsplit reference | numerical validation of M4 |

The warm-start ablation (`cold / warped-m only / warped-m + Krylov`) is dropped with
the mechanism itself; see 8.3b.

`ESS threshold = 1.0` degenerates to per-frame independent solves, which is the
correct internal baseline: it isolates the contribution of the sequential scheme
from everything else.

### 4.2 Baselines

Matched wall-clock, all of them. DDPM/DDIM alone is not defensible in a current
submission.

- Internal: per-frame independent FP solve (ESS = 1.0)
- Fast samplers: DPM-Solver, DPM-Solver++
- Distilled / few-step: consistency models, rectified flow
- Video-native: at least one video diffusion baseline appropriate to the chosen task

### 4.3 Metrics

- FVD (state the backbone and its known caveats)
- Per-frame KID (preferred over FID at achievable sample counts)
- **Temporal consistency:** flow-warped LPIPS / warping error -- the metric the
  contribution is actually about
- Fitting wall-clock, FP iteration count per frame, ESS trace over time
- User study: two-alternative forced choice on temporal stability

### 4.4 Target task: per-instance video inpainting / object removal

**Selected and implemented.** Fit the score prior to the *observed* content of
a single clip and sample the masked region. No external training data; the
per-instance setting is the claim, not a limitation.

Implementation: `fast_diffusion/model/inpaint.py` (masks + masked probability-flow
sampler), masked metrics in `evaluate_video.py`, driver `run_video.py`, configs in
`fast_diffusion/configs/video/`. The masked sampler uses a fixed-step integrator
rather than `solve_ivp` because the known region must be projected in *between*
steps and `solve_ivp` exposes no hook to modify its state; projecting inside the
right-hand side is a different operation and leaves the known region free to drift.

Why this task and not the alternatives:

- **It matches what the code actually does.** Scores are precomputed per clip and
  the network is indexed by clip identity. Inpainting is the task where fitting to
  one instance is the accepted methodology (the Deep Image Prior lineage), so the
  regime needs no apology.
- **The contribution and the metric coincide.** Temporal incoherence inside the
  synthesised region is the standard failure mode of video inpainting, and it is
  exactly what sequential importance sampling with a KL-triggered keyframe rule
  addresses. Warping error measured *inside the mask* is a clean, targeted number.
- **The baselines are fair.** Compare against zero-shot / internal-learning
  methods, where a per-instance fit is the same category. Trained video inpainting
  models can be reported as context, clearly labelled as a different setting
  rather than as a claim.
- **The keyframe trigger acquires physical meaning.** Disocclusion and scene change are
  precisely when the warped proposal fails, so the keyframe rate becomes an
  interpretable quantity rather than a tuning artefact.
- **Strong visual payoff.** Object removal before/after is exactly the kind of
  supplemental video a graphics audience responds to.
- **It fits the compute budget.** A handful of clips at 16x64x64 (3.8 min of FP
  precompute each) plus a small 128x128 showcase set -- not a dataset.

Rejected alternatives: denoising and temporal super-resolution both face very
strong *trained* baselines that a per-instance fit will not beat, and the
comparison invites the fitting-time / sampling-time confusion. Neural video
compression drags in rate-distortion evaluation as a whole second discipline. 4D
reconstruction regularisation is deferred with the rest of the 4D scope.

**Secondary, diagnostic only:** high-noise denoising on the same clips, purely
because ground truth is free there and PSNR/SSIM give a quantitative sanity check
on the prior. Not a second contribution, and not a second results section.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| ~~Split solve too slow at useful resolutions~~ | **Resolved.** 26.6x over `spsolve`; 3.8 min/clip at 16x64x64. Section 8.3. |
| ~~Stability constraint `sigma^2 dt <= 0.5`~~ | **Resolved by upwinding.** Margin is exactly 1 for every configuration. Section 8.2. |
| Upwind numerical diffusion smooths the score field | **Measured, section 8.3c.** Order 0.99 vs 2.04; relative added diffusion is exactly `\|s\|h/2`, so ~50% at the shipped `dh=1`, and sigma-independent. Reported as a cost, with `dh` as the documented lever. |
| Solver cost at 128x128 (18 min/clip) limits the experiment count | Port `thomas_batch` to torch on GPU before the campaign -- 1024-2048 independent lines is an ideal GPU workload. Section 8.3. |
| 8 GB VRAM caps clip size during training, not just precompute | Budget VRAM for the (2+1)D backbone at 16x64x64 first; use gradient checkpointing or shorter windows at 128x128. |
| IS weight degeneracy on most real sequences | KL trigger makes this graceful rather than fatal (ESS was measured blind, 8.5); report keyframe rate as a result, not a failure |
| Flow errors dominate density errors | The identity-warp ablation quantifies this directly |
| Reviewers read fitting-time speedup as sampling-time speedup | State it in the abstract, the contributions list, and every table caption |
| ~~FVD not comparable with published values~~ | **Resolved.** Canonical Kinetics-400 I3D, 400-d pre-softmax features, no substitute backbone available. Section 8.3d. |
| Temporal path does not beat the per-frame baseline at short training budgets | Real and unresolved: -0.09 dB masked PSNR at 60 epochs, and trivial `copy_prev` beats both by ~10 dB. Expected, since temporal layers start at exactly zero contribution, but it must be shown resolved at the full budget or reported as a negative result. `run_video.py` warns whenever it holds. |
| "Why not just add a temporal consistency loss?" | The keyframe-trigger and no-warp ablations must answer this quantitatively |

---

## 6. Sequencing

1. ~~**Feasibility test (gate).**~~ **DONE -- passed.** See Section 8.
2. Prerequisites: M1, M2 (done), plus `benchmark_density.py` (done) and the
   README correction (done).
3. M3, M4 (done), then M5 (done), M6 -- the core contribution.
4. M7, M8, M9 -- needed for generation results.
5. Evaluation harness (`evaluate_video.py`), then ablations, then baselines.
6. User study.
7. Writing; supplemental video throughout, not at the end.

---

## 7. Open decisions

- ~~**Dynamic 3D / 4D.**~~ **DECIDED: out of scope.** The paper is **2D dynamic
  video** only -- `T x H x W`, two spatial dimensions plus time, with real object and
  camera motion. Extending the FP domain to a deforming or unstructured 4D
  representation (deformable Gaussians, dynamic NeRF) is a separate contribution and
  will not be attempted here. Wording discipline follows from this: "the 3D operator"
  in `fp_video` means the three axes of the linear system, two of which are pixel-value
  axes and one of which is time. Nothing in the paper should let a reader infer
  volumetric support.
- ~~**Drift discretisation.**~~ **RESOLVED.** Upwinding is implemented and is the
  default (`diffusion.stencil: upwind`). Settling it required correcting the
  inherited sign convention: expanding a legacy row shows the diffusion term
  multiplying the antisymmetric neighbour combination and the drift the symmetric
  one, i.e. the two roles swapped relative to implicit Euler. Only the `a`
  (+direction) coefficient differs. `stencil='central'` implements the
  sign-consistent second-order version, `stencil='legacy'` the inherited one for
  reproducing published numbers. See Section 8.1-8.2.

---

## 8. Measured results from the gate

All numbers produced during scaffolding, on CPU. Reproduce with
`benchmark_density.py` and the helpers in `fast_diffusion/model/fp_video.py`.

### 8.1 The solver: upwind line relaxation

The scheme is `stencil='upwind'` + `scheme='line'` (line Gauss-Seidel over
directions). Reaching that took three corrections, each caught by measurement:

1. **A removed Strang variant.** Composing two implicit half-solves,
   `(I + L/2)^-2`, does not agree with `(I + L)^-1` at second order; it measured
   strictly worse than plain ADI (2.1e-2 vs 1.8e-2). Deleted rather than shipped.
2. **Node indexing.** The inherited assembly is neighbour-indexed
   (`A[p, p+1] = a[p+1]`), so row p mixes drift values from three different nodes.
   Under that convention no row-wise dominance statement exists and upwinding is
   meaningless, because the sign of `v` selecting the upwind direction for row p is
   not the sign used to build row p's off-diagonals. Row p must take all its
   coefficients from node p. Measured effect: 2D row dominance margin went from
   **-89.3 to exactly 1.0**.
3. **Diagonal ownership.** Giving each Gauss-Seidel factor only `1 + diag_extra`
   while the operator diagonal is `1 + 3 diag_extra` left the remainder carrying
   `2 diag_extra`; the iteration matrix norm approached 2 and the solve diverged
   (relative error 2.5e+91). Every tridiagonal solve must retain the **full**
   diagonal, making the remainder ratio `2 d / (1 + 2 d) < 1` unconditionally.

Accuracy: line relaxation converges to the *exact* unsplit solution --
`splitting_error` measures 4.6e-9 against `unsplit_solve` at an inner tolerance of
1e-8. There is no splitting error to defend in review. Single-pass ADI is retained
as a cheap ablation point and carries ~6e-2.

### 8.2 Unconditional stability, and what it costs

Upwinding makes the operator an M-matrix whose row dominance margin is **exactly
1 for every configuration**. Measured margins, upwind vs central:

| sigma | N | \|s\| | upwind | central |
|---|---|---|---|---|
| 2 | 20 | 20 | 1.0000 | -4.40 |
| 5 | 20 | 20 | 1.0000 | -32.75 |
| 25 | 4 | 500 | 1.0000 | -116717.75 |

The consequence is that the `sigma^2 dt <= 0.5` restriction is **gone**. All 12
configurations across sigma in {2,5,10,25} and N in {4,20,50} now converge in 4
outer iterations, including sigma=25, N=4 at `sigma^2 dt = 156`, which previously
broke down catastrophically. `check_config_stability` is retained but only applies
to the central and legacy stencils.

The M-matrix property also gives a discrete maximum principle, so the scheme
cannot produce spurious oscillation.

**The cost is real and must be reported.** Upwinding stiffens the operator in the
drift-dominated regime (`diag_extra` grows like `2 dh |v|`), so line relaxation
needs roughly 30 sweeps instead of a handful. Convergence is linear at about 5
sweeps per decade, which makes the inner tolerance the dominant cost knob:

| inner tol | sweeps | ms/solve (16x64x64) | rel err vs exact |
|---|---|---|---|
| 1e-2 | 10 | 283 | 8.2e-3 |
| 1e-4 | 20 | 507 | 7.6e-5 |
| 1e-6 | 30 | 875 | 7.2e-7 |
| 1e-8 | 40 | 1041 | 6.9e-9 |

`diffusion.inner_tolerance` exposes this; the default is 1e-6.

Accuracy trade: upwinding is first-order accurate in space against central
differencing's second order, adding numerical diffusion of `|v| h / 2`. **This is now
measured rather than asserted -- see 8.3c**: observed order 0.99 vs 2.04, the fitted
artificial diffusion matching `|v|h/2` to four digits, and a relative added diffusion
of `|s| h / 2` in which sigma cancels entirely. At the shipped `dh = 1` that is about
50% of the physical diffusion. Do not present unconditional stability as free.

### 8.3 Feasibility, measured honestly

Upwind line relaxation, inner tolerance 1e-6, |s| ~ 5, single-threaded numpy on
CPU:

| clip | n | sweeps | ms/solve | `spsolve` | speedup | s per fixed-point iteration |
|---|---|---|---|---|---|---|
| 8x32x32 | 8,192 | 29 | 68 | 241 ms | 3.5x | 4.1 |
| 16x64x64 | 65,536 | 30 | 798 | 21,201 ms | **26.6x** | 47.9 |
| 16x128x128 | 262,144 | 30 | 4,746 | intractable | -- | 284.8 |
| 32x256x256 | 2,097,152 | 31 | 57,114 | intractable | -- | 3,426.8 |

At 4 outer iterations that is **3.8 min per clip at 16x64x64** and about 19 min at
16x128x128 by this per-solve extrapolation. 32x256x256 is about 3.8 hours per clip and
is out of scope. Note the extrapolation overstates the cost: the measured end-to-end
CPU figure at 16x128x128 is 5.0 min at `solve_tolerance` 1e-4 (the extrapolation
assumes every solve needs the full sweep budget). Use the end-to-end table below.

> An earlier revision of this plan quoted 187x and 0.7 s per fixed-point
> iteration. Those numbers were measured with the divergent per-factor scheme and
> a near-zero score field, and are withdrawn. The figures above supersede them.

**Decision: experiments at 16x64x64, with 16x128x128 for a small showcase set.**

**GPU port: done, and the payoff is size-dependent.** `fast_diffusion/model/fp_torch.py`
mirrors the numpy solver in torch and is pinned to it by `tests_equivalence()`, which
reproduces the reference **bit-for-bit** (rel diff 0.000e+00 on coefficients, diagonal,
operator application, both stencils' solves, and the gradient). `benchmark_solver.py`
regenerates the table below; medians of 3, RTX 2000 Ada Laptop.

Single-volume solve, upwind, inner tolerance 1e-6:

| clip | n | numpy CPU | torch GPU fp32 | speedup | GPU vs CPU rel err |
|---|---|---|---|---|---|
| 8x32x32 | 8,192 | 164 ms | 1,065 ms | **0.15x** | 5.7e-07 |
| 16x64x64 | 65,536 | 1,056 ms | 1,310 ms | **0.81x** | 4.4e-07 |
| 16x128x128 | 262,144 | 5,973 ms | 2,255 ms | **2.65x** | 4.5e-07 |
| 32x256x256 | 2,097,152 | 83,565 ms | 9,454 ms | **8.84x** | 2.1e-07 |

End-to-end per-clip score precompute (N=20, C=3, 4 outer iterations):

| clip | GPU | CPU | speedup |
|---|---|---|---|
| 8x32x32 | 12.7 s | 4.6 s | **0.36x** |
| 16x64x64 | 31.1 s | 42.2 s | **1.36x** |
| 16x128x128 | 55.7 s | 298.5 s | **5.36x** |

**The GPU is a net loss below ~64x64 and only clearly wins at 128x128.** That is the
opposite of the assumption in the previous revision of this section, so
`train_video._select_backend` defaults to `auto` and picks numpy below 96 px.

Two diagnostics explain the shape of that table and both changed the implementation:

* **The solver is kernel-launch bound, not arithmetic bound.** fp64 costs 0.85-1.33x
  fp32 across all sizes. Ada fp64 arithmetic runs at 1/64 the fp32 rate, so
  near-parity means arithmetic is a negligible share of runtime. The earlier concern
  that fp64 would be 64x slower and force fp32 was therefore misplaced: fp32 is kept
  as the default only because it is never slower, not because fp64 is unaffordable.
  fp32's deviation from fp64 is ~9e-8, an order of magnitude below the 1e-6 inner
  tolerance.
* **Device-side pivot checking cost ~2.5x.** Testing a Thomas pivot on device forces a
  device-to-host synchronisation, and doing it inside the recursion serialises the
  pipeline (1,701 -> 677 ms at 8x32x32). It is off by default in the torch backend;
  under `upwind` the matrix is a diagonally dominant M-matrix so no pivot can vanish
  and the check is provably redundant.
* **Channel batching gains 2.1-3.3x at zero numerical cost.** The C colour channels
  are independent, so batching them into the tridiagonal batch dimension does the same
  arithmetic with 1/C the launches. Measured rel diff between batched and sequential:
  exactly 0.

Remaining headroom is in launch count -- a sweep issues on the order of a thousand
small kernels. CUDA graph capture of the sweep body is the obvious next step and is
**not implemented**.

> An earlier revision of this plan quoted 187x and 0.7 s per fixed-point
> iteration. Those numbers were measured with the divergent per-factor scheme and
> a near-zero score field, and are withdrawn. The figures above supersede them.
> The same revision claimed 18 min per clip at 16x128x128 on CPU; the measured
> end-to-end figure is 298.5 s (5.0 min) at `solve_tolerance` 1e-4.

### 8.3b The warm start was measured, and removed

The cross-clip score warm start was the mechanism section 1 originally rested its
speedup claim on. It was measured against a cold-start control (4 clips of 8x48x48,
N=10):

| setting | clips warm-started | warm iters | cold iters | iteration saving | wall clock |
|---|---|---|---|---|---|
| contiguous windows | 2 of 4 | 4.00 | 4.50 | **1.12x** | 1.09x |
| independent scenes | 0 of 4 | -- | -- | -- | -- |

**It has been removed from the codebase.** 1.12x does not justify a mechanism, its
config surface, and its correctness caveats. The cause was structural, not a tuning
failure: upwinding made the fixed-point iteration converge in 4-5 iterations from a
cold start, so there was almost nothing left to save. The two contributions were in
direct tension and the more stable solver won -- it consumed the headroom the warm
start existed to exploit.

What this costs the paper: nothing that was load-bearing. The efficiency claim now
rests on the KL-triggered incremental density correction, which measures **7.54x**
(section 8.5) rather than 1.12x, and the stability result (section 8.2), which is
independent of both.

What survives from the second row of that table: the trigger correctly declining to
reuse a proposal across a genuine discontinuity. That behaviour is now reported as
part of the keyframe trigger in 8.5, where it belongs.

Recorded here so it is not reintroduced: warm-starting the FP iteration is not a
promising direction *given* upwinding. It would only pay at much tighter
`solve_tolerance`, where cold-start iteration counts are high enough to leave
something to save. That was never measured.

### 8.3c Upwinding in space: the benefit and the exact cost

`benchmark_stencil.py`. The discrete operator is applied to a smooth analytic field
and compared against the exact continuous operator, which isolates truncation error
without needing an exact PDE solution. Boundary cells are excluded from every norm --
the Neumann ghost-node folding is first-order by construction, so including them
would report order 1 for both stencils and hide the difference.

**Observed spatial order** (L2 of the operator error, interior only):

| n | h | cell Peclet | upwind L2 | order | central L2 | order |
|---|---|---|---|---|---|---|
| 17 | 0.0625 | 1.000 | 1.02e+01 | -- | 9.22e-01 | -- |
| 33 | 0.0312 | 0.500 | 5.22e+00 | 0.97 | 2.20e-01 | 2.07 |
| 65 | 0.0156 | 0.250 | 2.62e+00 | 0.99 | 5.35e-02 | 2.04 |
| 129 | 0.0078 | 0.125 | 1.31e+00 | 1.00 | 1.32e-02 | 2.02 |

**Upwind 0.99, central 2.04.** Exactly the textbook orders, which is the confirmation
that the shipped assembly is doing what it claims.

**Artificial diffusion.** The difference between the two operators is the artificial
diffusion term; least-squares projecting it onto the Laplacian recovers its
coefficient without assuming the leading-order analysis:

| h | fitted `D_num` | predicted `\|v\|h/2` | ratio | `D_num / D` |
|---|---|---|---|---|
| 0.0625 | 2.47e-01 | 2.50e-01 | 0.987 | 0.494 |
| 0.0312 | 1.25e-01 | 1.25e-01 | 0.997 | 0.249 |
| 0.0156 | 6.24e-02 | 6.25e-02 | 0.999 | 0.125 |
| 0.0078 | 3.12e-02 | 3.12e-02 | 1.000 | 0.062 |

The prediction is confirmed to 4 digits at the finest grid. Reducing it to the
quantities a config actually sets:

```
D_num / D  =  |v| h / (2 D)  =  |s| h / 2
```

**`g` cancels, so sigma is irrelevant to the artificial diffusion.** The only levers
are `diffusion.dh` and the score magnitude. This matters practically: at the shipped
`dh = 1.0` with a measured score range of order `|s| ~ 1`, the artificial diffusion is
**~50% of the physical diffusion**. That is not a rounding error and must be stated.
Halving `dh` halves it, at 4x the solve cost per axis-refinement.

**Stability boundary** (sigma=5, N=20, h=1). Three thresholds that are easy to
conflate:

| `\|s\|h` | upwind margin | central margin | central solves | central max\|u\| |
|---|---|---|---|---|
| 2.0 | 1.0000 | 1.0000 | yes | 7.84e-01 |
| 2.1 | 1.0000 | 0.8125 | yes | 7.98e-01 |
| 2.5 | 1.0000 | 0.0625 | yes | 8.53e-01 |
| 2.6 | 1.0000 | -0.1250 | yes | 8.67e-01 |
| 3.0 | 1.0000 | -0.8750 | **no** | 1.15e+04 |
| 25.0 | 1.0000 | -42.1250 | **no** | 1.79e+143 |

1. **M-matrix (sign) condition: `|s|h <= 2`.** Past this an off-diagonal changes sign
   and the discrete maximum principle is lost, so spurious oscillation becomes
   possible even while the solve still converges.
2. **Diagonal dominance: `|s|h ~ 2.53`.** The shared identity term sustains dominance
   past the sign condition.
3. **Actual divergence: between 2.6 and 3.0.**

Upwind holds a margin of **exactly 1.0 at every magnitude tested, including
`|s| = 25`**. Reporting only one of the three central thresholds would misstate where
central is usable, so all three are given. Note that "central diverges" and "central
loses its guarantee" are 30% apart in `|s|h`, and the gap is where oscillation lives.

### 8.3d FVD is now comparable with published values

The earlier implementation computed FVD from torchvision's `r3d_18` because the
canonical Kinetics-400 I3D weights are not redistributable. That was documented as
non-comparable, which was honest but useless: an FVD that cannot be placed next to
any published FVD is not a baseline comparison.

Resolved by downloading the canonical weights (`download_assets.py`):

- I3D TorchScript, 51,235,320 bytes, sha256 `bec6519f66ea534e...`, feature width
  verified at **400** (pre-softmax logits, the layer FVD is defined on).
- Preprocessing is delegated to the module's own `rescale` / `resize` flags rather
  than reimplemented, since FVD is as sensitive to preprocessing as to weights.
- DAVIS 2017 480p trainval, sha256 `e3d0b5b77c3d031b...`, 90 sequences / 6208 frames,
  which clears `MIN_FVD_SAMPLES = 64` with one clip each.

Validated behaviour: ordering holds across same-process (2.14) < systematically
shifted (89.78) < heavily corrupted (947.0); identical sets give -7.6e-06.

Guards, each verified to fire: sample count below 64; fewer than 9 frames per clip
(the I3D temporal stack collapses at T=8 and fails inside the TorchScript
interpreter); missing weights. **`fvd()` has no backbone parameter at all** -- the
`r3d_18` path was deleted rather than kept as a fallback, because a non-comparable FVD
is not a weaker result but a different quantity wearing the same name.

Two bugs found while doing this, both of which would have produced wrong or absent
numbers on a current install:

- `scipy.linalg.sqrtm` has dropped its `disp` keyword, so the Frechet distance raised
  `TypeError` outright. Now handles either return convention, with an
  ill-conditioning offset and a guard that refuses a large imaginary component instead
  of silently taking the real part.
- The I3D resize path calls `.view()`, which fails on the non-contiguous tensor a
  `permute` produces. Fixed with an explicit `.contiguous()`.

**The inpainting comparison now runs on DAVIS with real object masks**
(`davis_inpaint.yml`, `DavisVideoDataset`). Real masks are irregular, track real
motion, and vary in coverage from 0.3% to 54.6% across sequences against a synthetic
box's fixed ~6%. Masks are resized nearest-neighbour, never bicubic: bicubic on a
DAVIS annotation produces fractional boundary values ranging to `[-0.166, 1.173]`,
i.e. outside `[0, 1]` entirely, which would put every masked metric on the wrong
support.

### 8.4 Density-estimator claims, corrected and re-measured

The committed `figures/kde_scaling.csv` timed exact KDE only up to 32x32 and left
64x64 empty, yet the README and docstring claimed "~250x faster at 64x64".
`benchmark_density.py` now regenerates the table (a warmup call was needed: lazy
`scipy.stats` import cost ~10 s and was landing in the first timed measurement,
producing a nonsensical 10,056 ms for a 64-sample problem). Measured:

| resolution | N | histogram | scipy exact | speedup |
|---|---|---|---|---|
| 8x8 | 64 | 43.6 ms | 0.5 ms | 0.01x |
| 32x32 | 1,024 | 27.6 ms | 10.1 ms | 0.37x |
| 48x48 | 2,304 | 23.7 ms | 44.8 ms | 1.89x |
| 64x64 | 4,096 | 23.0 ms | 142.9 ms | **6.20x** |
| 256x256 | 65,536 | 23.3 ms | intractable | -- |

Histogram cost is constant in resolution (fixed 256^2 grid); exact KDE is O(N^2),
confirmed by the 16x cost rise per 4x rise in N. The honest claim is therefore:
**slower below 48x48, crossover at 48x48, 6.2x at 64x64, advantage growing
quadratically thereafter.** Not 250x.

### 8.5 The keyframe trigger is KL, not ESS

The ESS-based trigger in the original design was implemented and measured to be
**blind**. On a synthetic clip with an injected scene cut it separated the cut from
within-shot frames by +/-0.0002 -- noise. The reason is structural: ESS measures
importance-weight *variance*, and the log-ratio between two pixel-value-pair
densities stays nearly constant across the support even when the densities differ
materially.

Discrimination of candidate statistics at the cut:

| statistic | cut frame | max within-shot | ratio |
|---|---|---|---|
| KL(target \|\| proposal) | 0.0060 | 0.0004 | **14.6x** |
| grid relative L2 | 0.3097 | 0.0375 | 8.3x |
| ESS | 0.9999 | 0.9997 | none |

The trigger is now `KL(target || proposal)` with an adaptive threshold,
`max(kl_factor * running_median, kl_floor)`. The floor is necessary: on smooth
motion the running median falls to ~1e-6 and a purely relative rule fired 3
spurious keyframes in a 10-frame no-cut control. With the floor, the cut clip fires
exactly once at the cut (KL 0.0077 vs ~0.00005 within-shot, ~150x) and the control
fires only on frame 0. ESS is retained and reported as a diagnostic.

**Re-measured on the current code** (`benchmark_keyframe_trigger.py`, 24 frames at
64x64, cut at frame 12), reporting cost, discrimination and false positives
separately, because a trigger needs all three and the earlier ESS design passed two of
them trivially while failing the third completely:

| property | measurement |
|---|---|
| cost: incremental correction vs full re-estimate | 3.3 ms vs 24.5 ms per frame = **7.54x** |
| discrimination: KL at cut vs mean within-shot | 3.35e-02 vs 9.89e-07 = **33931x** |
| discrimination: KL at cut vs *worst* within-shot frame | 3.35e-02 vs 1.29e-05 = **2591x** |
| false positives on a no-cut control | **0** in 23 frames |
| detection | fires exactly once, at frame 12 |
| ESS on the same sequence | 0.871 at cut vs 1.000 within shot = **0.871x** |

The 7.54x is the paper's efficiency result. It replaces the 1.12x warm start (8.3b),
and unlike that mechanism it is not in tension with the solver work.

The ESS row is included deliberately: it is the originally proposed trigger, measured
on the identical sequence, moving by 13% where KL moves by five orders of magnitude.

**A limitation to state in the paper.** Pixel-value pairs are a *global* statistic,
insensitive to purely spatial rearrangement. A cut that changes layout while
preserving the value distribution is hard to detect from this density. The fix,
where it matters, is to build the density on temporal pairs
`(I_k(p), I_{k-1}(warp(p)))`, which puts motion compensation into the coordinate
system; `flow.warp_samples` provides the plumbing. This should be an ablation axis,
not a footnote.

### 8.6 Repository defects found

- `reproduce_all.py:82-88`: when `model.pth` already exists the run is skipped but
  recorded as `success=True` with `wall_time = float(timing.get("total", 0))`,
  which is 0.00 when the timing CSV has no `total` row. This exactly matches the
  committed `reproduction_summary.csv`, in which all seven baselines show
  `success=True, wall_time=0.00`. Skips must be reported as skips.
- `saves/` is gitignored and absent, so no published number is regenerable from a
  fresh clone. Either commit the timing CSVs or ship a manifest.
- The system Python 3.12 has a **corrupted `torch`**: a namespace directory with no
  `__init__.py`, plus `~-b` and `~ib` folders left by a failed pip uninstall.
  `import torch` succeeds but exposes nothing, which also breaks `scipy.stats` and
  `sklearn` (both probe `torch.Tensor` at import). **Resolved by using `.venv`**
  (Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, RTX 2000
  Ada sm_89, 8.59 GB). The system interpreter is still broken; always use `.venv`.

### 8.7 Environment and VRAM budget

`.venv` verified end to end on GPU. The (2+1)D backbone at ch=64, ch_mult=[1,2,2],
16x64x64, batch 1 uses **2.34 GB peak for forward+backward of 8.59 GB available**,
with 8.3 M parameters. Zero-initialised temporal layers were confirmed to reproduce
the 2D model exactly at initialisation (max difference 0.0e+00), so "no temporal
layers" is a genuine ablation baseline.

Headroom exists at 64x64 for a wider network or a larger batch; at 128x128 expect
roughly 4x the activation memory, so plan for gradient checkpointing or shorter
temporal windows there.
