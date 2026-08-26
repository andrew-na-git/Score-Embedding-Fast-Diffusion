"""Krylov control-projection convergence measurement and figure.

Regenerates:
  figures/krylov_control_convergence.pdf   -- CG relative-residual curves
  figures/krylov_control_measurement.json  -- the numbers cited in the paper

All numbers are produced without a trained network: they are properties of the
matrix-free projection in fast_diffusion/model/constraints.py. The synthetic clip
is a rigid translation with exact known flow, so flow-consistency is satisfiable
and the hard projection drives the masked residual to machine zero.
"""
import json
import os

import numpy as np
import torch

from fast_diffusion.model.constraints import FlowConsistency, _adjoint, _flatten, _unflatten


def build_problem(T=5, Ch=3, H=12, W=12, seed=0):
    torch.manual_seed(seed)
    unknown = torch.zeros(T, 1, H, W, dtype=torch.float64)
    unknown[:, :, 3:9, 3:9] = 1.0
    flows = [torch.zeros(2, H, W, dtype=torch.float64) for _ in range(T - 1)]
    for f in flows:
        f[0] = -1.0
    base = torch.randn(Ch, H, W, dtype=torch.float64)
    clip = torch.stack([torch.roll(base, shifts=t, dims=2) for t in range(T)])
    corrupt = clip + unknown * torch.randn_like(clip) * 0.8
    return corrupt, [FlowConsistency(flows, unknown)], unknown


def make_matvec(x, constraints, free_mask, ridge):
    residuals = [c.apply(x) - c.target(x) for c in constraints]
    shapes = [tuple(r.shape) for r in residuals]
    b = _flatten(residuals)

    def ctv(yflat):
        yb = _unflatten(yflat, shapes)
        xadj = None
        for c, y in zip(constraints, yb):
            g = _adjoint(c, x, y)
            xadj = g if xadj is None else xadj + g
        return free_mask * xadj

    def matvec(yflat):
        base = _flatten([c.apply(ctv(yflat)) for c in constraints])
        return base + ridge * yflat if ridge else base

    return matvec, b


def cg_history(matvec, b, x0=None, maxiter=60):
    """CG that records the relative residual after each iteration."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = torch.dot(r, r)
    bnorm = torch.sqrt(torch.dot(b, b)).clamp_min(1e-30)
    hist = [float((torch.sqrt(rs) / bnorm).item())]
    for _ in range(maxiter):
        Ap = matvec(p)
        alpha = rs / torch.dot(p, Ap).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        hist.append(float((torch.sqrt(rs_new) / bnorm).item()))
        if hist[-1] < 1e-12:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, hist


def main():
    os.makedirs("figures", exist_ok=True)
    x, cons, unknown = build_problem()

    mv0, b = make_matvec(x, cons, unknown, ridge=0.0)
    _, cold = cg_history(mv0, b)

    # Warm start: solve a nearby problem, reuse its dual as the initial guess.
    x_near = x + unknown * 1e-2 * torch.randn_like(x)
    mv_near, b_near = make_matvec(x_near, cons, unknown, ridge=0.0)
    y_near, _ = cg_history(mv_near, b_near)
    _, warm = cg_history(mv0, b, x0=y_near)

    mvr, br = make_matvec(x, cons, unknown, ridge=0.5)
    _, ridge = cg_history(mvr, br)

    meas = {
        "cold_iters_to_1e-8": next((i for i, v in enumerate(cold) if v < 1e-8), len(cold)),
        "warm_iters_to_1e-8": next((i for i, v in enumerate(warm) if v < 1e-8), len(warm)),
        "ridge_iters_to_1e-8": next((i for i, v in enumerate(ridge) if v < 1e-8), len(ridge)),
        "cold_curve": cold, "warm_curve": warm, "ridge_curve": ridge,
    }
    with open("figures/krylov_control_measurement.json", "w") as f:
        json.dump(meas, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        ax.semilogy(cold, "o-", ms=3, label="cold start")
        ax.semilogy(warm, "s-", ms=3, label="warm start (prev. dual)")
        ax.semilogy(ridge, "^-", ms=3, label=r"regularised ($\lambda^{-1}{=}0.5$)")
        ax.set_xlabel("CG iteration")
        ax.set_ylabel("relative residual")
        ax.set_title("Krylov control projection: CG convergence", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig("figures/krylov_control_convergence.pdf")
        print("wrote figures/krylov_control_convergence.pdf")
    except Exception as e:  # figure is optional; the JSON is the source of record
        print(f"figure skipped ({e})")

    print(f"iters to 1e-8: cold={meas['cold_iters_to_1e-8']} "
          f"warm={meas['warm_iters_to_1e-8']} ridge={meas['ridge_iters_to_1e-8']}")


if __name__ == "__main__":
    main()
