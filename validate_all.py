"""Consolidated validation of every invariant asserted by the video extension.

Run this after any change to the solver, the estimator, the masked sampler or the
metrics. It is fast (no training) and it is the gate that catches the class of defect
that produces plausible numbers rather than errors.
"""
import numpy as np
import torch

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        FAILURES.append(name)


print("=" * 74)
print("1. torch backend is equivalent to the numpy reference")
print("=" * 74)
from fast_diffusion.model import fp_torch, fp_video
check("fp_torch.tests_equivalence", lambda: fp_torch.tests_equivalence(verbose=False))


def line_relaxation_is_exact():
    """Line relaxation must converge to the exact unsplit solve, not an approximation."""
    rng = np.random.default_rng(0)
    rhs = rng.random((5, 12, 12))
    s = rng.standard_normal((5, 12, 12)) * 5
    u_split = fp_video.fp_solve(rhs, 5.0, s, 0.025, 0.05, scheme="line",
                                stencil="upwind", tol=1e-10)
    u_exact = fp_video.unsplit_solve(rhs, 5.0, s, 0.025, 0.05, stencil="upwind")
    err = np.linalg.norm(u_split - u_exact) / np.linalg.norm(u_exact)
    assert err < 1e-7, f"relative error {err:.2e} vs the unsplit solve"


check("line relaxation == unsplit solve", line_relaxation_is_exact)


def upwind_is_stable_everywhere():
    """The upwind operator must stay an M-matrix for every configuration tried."""
    worst = None
    for sigma in (3.0, 5.0, 10.0, 25.0):
        for N in (4, 10, 20):
            for smax in (1.0, 5.0, 25.0):
                dt = 1.0 / N
                m = fp_video.stability_margin(
                    float(sigma), np.full((4, 8, 8), smax), dt / 2, dt,
                    stencil="upwind")
                worst = m if worst is None else min(worst, m)
    assert worst >= 1.0 - 1e-12, f"worst stability margin {worst}"


check("upwind stability margin >= 1 over (sigma, N, |s|)", upwind_is_stable_everywhere)


def neumann_gradient_is_exact():
    """A linear ramp must have a constant gradient, boundaries included."""
    H = W = 16
    ramp = np.tile(np.arange(W, dtype=np.float64), (H, 1))
    g = fp_video.log_density_gradient_3d(ramp[None], 1.0, axis=-1)[0]
    assert np.allclose(g, 1.0), f"gradient range [{g.min()}, {g.max()}], expected 1.0"


check("Neumann gradient exact on a linear ramp", neumann_gradient_is_exact)

print()
print("=" * 74)
print("2. keyframe trigger discriminates a cut from within-shot variation")
print("=" * 74)
from fast_diffusion.model.density import SequentialDensityEstimator


def kl_separates_cut():
    rng = np.random.default_rng(0)
    est = SequentialDensityEstimator(threshold_mode="absolute", kl_threshold=1e9)
    within, across = [], []
    base = rng.random((2, 4000))
    for i in range(8):
        _, info = est.estimate(np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1))
        if i > 2:
            within.append(info["kl"])
    _, info = est.estimate(rng.random((2, 4000)) ** 3)  # hard change of distribution
    across.append(info["kl"])
    w, a = float(np.mean(within)), float(np.mean(across))
    print(f"        within-shot KL {w:.3e}   at cut {a:.3e}   ratio {a/w:.1f}x")
    assert a > 5 * w, f"KL fails to separate the cut: {a:.3e} vs {w:.3e}"


check("KL separates a distribution change", kl_separates_cut)

print()
print("=" * 74)
print("3. masked sampler preserves observed pixels")
print("=" * 74)
from fast_diffusion.model import inpaint
check("inpaint.tests_masked_sampling",
      lambda: inpaint.tests_masked_sampling(verbose=False))


def masks_have_comparable_coverage():
    """Cross-mask comparisons are meaningless at different coverages."""
    covs = {k: inpaint.make_mask(k, 16, 64, 64)[:].mean()
            for k in ("static_box", "moving_box", "stroke")}
    print("        " + "  ".join(f"{k}={v*100:.1f}%" for k, v in covs.items()))
    lo, hi = min(covs.values()), max(covs.values())
    assert hi / lo < 3.0, f"coverage spread {lo*100:.1f}%-{hi*100:.1f}% is too wide"


check("mask coverages are within 3x of each other", masks_have_comparable_coverage)


def moving_box_is_temporal():
    """The moving box must make some pixels recoverable from other frames."""
    m = inpaint.make_mask("moving_box", 16, 64, 64)
    per_pixel = m[:, 0].sum(0)
    frac = float(np.logical_and(per_pixel > 0, per_pixel < 16).mean())
    s = inpaint.make_mask("static_box", 16, 64, 64)[:, 0].sum(0)
    frac_s = float(np.logical_and(s > 0, s < 16).mean())
    print(f"        moving_box {frac*100:.1f}% recoverable, static_box {frac_s*100:.1f}%")
    assert frac > 0.1, "moving_box is not a temporal problem"
    assert frac_s == 0.0, "static_box should be purely spatial"


check("moving_box requires temporal reasoning", moving_box_is_temporal)

print()
print("=" * 74)
print("4. masked metrics behave as claimed")
print("=" * 74)
from fast_diffusion.model import evaluate_video as ev


def masked_psnr_matches_coverage_prediction():
    """masked and whole-frame PSNR must differ by -10*log10(coverage)."""
    rng = np.random.default_rng(0)
    target = torch.as_tensor(rng.random((8, 3, 32, 32)), dtype=torch.float32)
    hole = inpaint.make_mask("moving_box", 8, 32, 32, size=0.3)
    h = torch.as_tensor(hole)
    damaged = target + h * 0.25
    mp = ev.masked_psnr(damaged, target, hole)
    wp = float(ev.psnr_per_frame(damaged, target).mean())
    cov = float(hole.mean())
    predicted = -10 * np.log10(cov)
    print(f"        gap {wp-mp:.2f} dB, predicted {predicted:.2f} dB at "
          f"{cov*100:.1f}% coverage")
    assert abs((wp - mp) - predicted) < 0.5


check("whole-frame PSNR flatters by -10*log10(coverage)",
      masked_psnr_matches_coverage_prediction)


def masked_metrics_ignore_observed_damage():
    """Damage outside the hole must not change the masked metric."""
    rng = np.random.default_rng(0)
    target = torch.as_tensor(rng.random((8, 3, 32, 32)), dtype=torch.float32)
    hole = inpaint.make_mask("moving_box", 8, 32, 32, size=0.3)
    outside = target + (1.0 - torch.as_tensor(hole)) * 0.5
    assert ev.masked_mse(outside, target, hole) == 0.0
    rep = ev.evaluate_inpainting(outside, target, hole, flow_method="identity")
    assert "warning" in rep, "the observed-pixel leak guard did not fire"


check("masked metrics ignore observed-pixel damage, and the guard fires",
      masked_metrics_ignore_observed_damage)


def seam_error_sees_boundary_defects():
    rng = np.random.default_rng(0)
    target = torch.as_tensor(rng.random((8, 3, 48, 48)), dtype=torch.float32)
    hole = inpaint.make_mask("static_box", 8, 48, 48, size=0.5)
    hn = np.asarray(hole)
    band = np.clip(hn - (1.0 - inpaint.dilate_mask(1.0 - hn, 2)), 0, 1)
    seamy = target + torch.as_tensor(band, dtype=torch.float32) * 0.25
    se = ev.seam_error(seamy, target, hole)
    mse = ev.masked_mse(seamy, target, hole)
    print(f"        seam_error {se:.3e} > hole-wide MSE {mse:.3e}")
    assert se > mse, "seam error is not concentrating on the boundary"
    assert ev.seam_error(target, target, hole) == 0.0


check("seam_error isolates boundary defects", seam_error_sees_boundary_defects)

print()
print("=" * 74)
print("5. network wiring")
print("=" * 74)
from network.network3d import VideoNet

CONFIG = {
    "data_loader": {"image_size": 32, "clip_len": 4, "channels": 3},
    "model": {"max_positions": 10000, "in_channels": 3, "out_channels": 3, "ch": 32,
              "ch_mult": [1, 2], "num_res_blocks": 1, "attention_resolutions": [16],
              "dropout": 0.0, "resample_with_conv": True,
              "embedding_method": "linear", "num_clips": 2, "max_frames": 64},
}


def frame_idx_is_tiled():
    """A per-frame index vector must work with a batch larger than one clip."""
    model = VideoNet(CONFIG).eval()
    B, T = 3, 4
    x = torch.randn(B, T, 3, 32, 32)
    out = model(x, torch.rand(B), clip_idx=torch.zeros(B, dtype=torch.long),
                frame_idx=torch.arange(T), n_frames=T)
    assert out.shape == (B, T, 3, 32, 32), out.shape


check("frame_idx of length T works for batch B>1", frame_idx_is_tiled)


def temporal_layers_start_at_zero():
    """At initialisation the temporal path must contribute exactly nothing.

    Tested as frame-equivariance: with the temporal convolutions, temporal attention
    and frame embedding all zero-initialised, each frame is processed independently,
    so permuting the input frames must permute the output frames identically. Any
    temporal coupling breaks that. All frames are given the same diffusion time so the
    time embedding cannot account for a difference.

    This is the property that makes inflation from a 2D-pretrained checkpoint valid:
    the inflated model starts exactly equal to its spatial counterpart.
    """
    torch.manual_seed(0)
    model = VideoNet(CONFIG).eval()
    B, T = 1, 4
    x = torch.randn(B, T, 3, 32, 32)
    labels = torch.full((B,), 0.5)
    clip_idx = torch.zeros(B, dtype=torch.long)
    perm = torch.tensor([3, 1, 2, 0])

    with torch.no_grad():
        straight = model(x, labels, clip_idx=clip_idx, n_frames=T)
        permuted = model(x[:, perm], labels, clip_idx=clip_idx, n_frames=T)

    diff = float((permuted - straight[:, perm]).abs().max())
    print(f"        max |f(perm x) - perm f(x)| = {diff:.3e}")
    assert diff < 1e-5, (
        f"temporal path is not inert at initialisation (equivariance error {diff:.3e}); "
        "2D-pretrained inflation would not start equal to the spatial model"
    )


check("temporal path is inert at init (frame-equivariance)",
      temporal_layers_start_at_zero)

print()
print("=" * 74)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
print("=" * 74)
