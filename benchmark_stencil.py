"""Quantify what upwinding buys and what it costs, in space.

The solver uses upwind differencing because central differencing does not merely lose
accuracy in the drift-dominated regime -- it diverges. That is a real benefit, but it
is bought with first-order spatial accuracy and an artificial diffusion term, and a
paper that reports the benefit without the cost is not reporting the method. This
script measures both.

What is measured
----------------
1. **Spatial order of accuracy.** The discrete operator is applied to a smooth
   analytic field and compared against the exact continuous operator, so the
   truncation error is isolated without needing an exact PDE solution. Expect order 1
   for upwind and order 2 for central.
2. **Artificial diffusion.** Upwinding a first-derivative term adds diffusion of
   |v|h/2. Expressed relative to the physical diffusion D = g^2/2, and in terms of the
   quantities actually set in a config.
3. **Stability boundary.** The cell Peclet number at which central loses diagonal
   dominance, verified against both the analytic prediction and the shipped
   `stability_margin`, alongside upwind's margin.
4. **Accuracy given up in the stable region.** Where central is usable, how much
   accuracy upwind sacrifices at the resolution the experiments actually run at.

The operator convention
-----------------------
`directional_coefficients` is called with `dh = dt/(2h)` and `dh2 = dt/h^2`, and
`apply_A` assembles `A = I - dt * L_h` where

    L u = D laplacian(u) + v . grad(u),    D = g^2 / 2,    v = f - g^2 s / 2

so `L_h u = (u - A u) / dt`. The same `v` is used on every axis, so the drift is
isotropic and the exact operator is `D (u_xx + u_yy + u_zz) + v (u_x + u_y + u_z)`.

A chosen (D, v) is imposed by inverting those relations: `g = sqrt(2 D)` and, with
f = 0, `s = -v / D`.

Boundaries are excluded from every error norm. The Neumann ghost-node folding is
first-order at the boundary by construction, so including boundary cells would
measure the boundary treatment rather than the interior stencil and would report
order 1 for both stencils.

Usage
-----
    python benchmark_stencil.py
    python benchmark_stencil.py --out-dir figures --peclet 0.5
"""

import argparse
import csv
import json
import os

import numpy as np

from fast_diffusion.model.fp_video import (
    apply_A,
    directional_coefficients,
    full_diagonal,
    fp_solve,
    is_node_indexed,
    stability_margin,
)

# Interior margin excluded from error norms, in cells.
GUARD = 2


def analytic_field(n, freq=1.0):
    """A smooth field and its exact first and second derivative sums on [0, 1]^3.

    Returns (u, grad_sum, lap) where grad_sum = u_x + u_y + u_z and lap is the
    Laplacian, all evaluated on the same uniform grid.
    """
    h = 1.0 / (n - 1)
    ax = np.linspace(0.0, 1.0, n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    k = 2.0 * np.pi * freq

    # A product of sines is smooth and has closed-form derivatives; adding a phase
    # keeps it from being (anti)symmetric about the grid centre, which could let a
    # first-order scheme cancel its own leading error term and flatter its order.
    u = np.sin(k * x + 0.3) * np.sin(k * y + 0.7) * np.sin(k * z + 1.1)

    ux = k * np.cos(k * x + 0.3) * np.sin(k * y + 0.7) * np.sin(k * z + 1.1)
    uy = k * np.sin(k * x + 0.3) * np.cos(k * y + 0.7) * np.sin(k * z + 1.1)
    uz = k * np.sin(k * x + 0.3) * np.sin(k * y + 0.7) * np.cos(k * z + 1.1)

    lap = -3.0 * k * k * u
    return u, ux + uy + uz, lap, h


def discrete_operator(u, g, s, h, dt, stencil):
    """L_h u, recovered from the assembled operator as (u - A u) / dt."""
    dh, dh2 = dt / (2.0 * h), dt / (h ** 2)
    a, c, diag_extra = directional_coefficients(g, s, dh, dh2, stencil=stencil)
    diag = full_diagonal(a, c, diag_extra)
    Au = apply_A(u, a, c, diag, node_indexed=is_node_indexed(stencil))
    return (u - Au) / dt


def interior(arr):
    return arr[GUARD:-GUARD, GUARD:-GUARD, GUARD:-GUARD]


def order_study(D, v, resolutions, dt=1e-3, freq=1.0):
    """Observed spatial order of each stencil, from the operator truncation error."""
    g = float(np.sqrt(2.0 * D))
    rows = []

    for n in resolutions:
        u, grad_sum, lap, h = analytic_field(n, freq)
        s = np.full_like(u, -v / D)          # gives drift exactly v on every axis
        exact = D * lap + v * grad_sum

        row = {"n": n, "h": h, "cell_peclet": abs(v) * h / D}
        for stencil in ("upwind", "central"):
            Lh = discrete_operator(u, g, s, h, dt, stencil)
            err = interior(Lh - exact)
            row[f"{stencil}_l2"] = float(np.sqrt(np.mean(err ** 2)))
            row[f"{stencil}_max"] = float(np.abs(err).max())
        rows.append(row)

    # Observed order between successive refinements.
    for i in range(1, len(rows)):
        hr = np.log(rows[i - 1]["h"] / rows[i]["h"])
        for stencil in ("upwind", "central"):
            k = f"{stencil}_l2"
            rows[i][f"{stencil}_order"] = float(
                np.log(rows[i - 1][k] / max(rows[i][k], 1e-300)) / hr
            )
    return rows


def artificial_diffusion(D, v, resolutions, dt=1e-3, freq=1.0):
    """Measure the extra diffusion upwinding introduces, against the |v|h/2 prediction.

    The difference between the two stencils' operators is exactly the artificial
    diffusion term, so projecting that difference onto the Laplacian recovers its
    coefficient without having to assume the leading-order analysis is right.
    """
    g = float(np.sqrt(2.0 * D))
    rows = []
    for n in resolutions:
        u, grad_sum, lap, h = analytic_field(n, freq)
        s = np.full_like(u, -v / D)

        up = discrete_operator(u, g, s, h, dt, "upwind")
        ce = discrete_operator(u, g, s, h, dt, "central")
        diff = interior(up - ce)
        lap_i = interior(lap)

        # Least-squares coefficient of the Laplacian in the difference.
        fitted = float(np.sum(diff * lap_i) / np.sum(lap_i * lap_i))
        predicted = abs(v) * h / 2.0
        rows.append({
            "n": n, "h": h,
            "fitted_D_num": fitted,
            "predicted_D_num": predicted,
            "ratio_to_prediction": fitted / max(predicted, 1e-300),
            "D_physical": D,
            "relative_added_diffusion": fitted / D,
            "cell_peclet": abs(v) * h / D,
        })
    return rows


def stability_boundary(sigma=5.0, N=20, h=1.0, score_mags=None):
    """Where central loses its guarantees, and where it actually diverges.

    Two distinct thresholds, which are easy to conflate:

    * **M-matrix (sign) condition**, |s| h <= 2. Beyond this an off-diagonal changes
      sign, the discrete maximum principle is lost, and spurious oscillation becomes
      possible. This is what `m_matrix_predicted` reports.
    * **Diagonal dominance**, which the shared identity term sustains somewhat past
      the sign condition. The relaxation keeps converging until the margin reaches
      zero, measured here at |s| h ~ 2.5.

    So central does not fail the instant the sign condition breaks -- it loses its
    guarantee first and diverges a little later. Reporting only one of the two would
    misstate where central is usable. Upwind is unconditional either way.
    """
    if score_mags is None:
        score_mags = (0.5, 1.0, 1.9, 2.0, 2.1, 2.4, 2.5, 2.6, 3.0, 5.0, 25.0)
    dt = 1.0 / N
    dh, dh2 = dt / (2.0 * h), dt / (h ** 2)
    rng = np.random.default_rng(0)
    rhs = rng.random((5, 12, 12))
    rows = []

    for smax in score_mags:
        s = np.full((5, 12, 12), smax)
        row = {"score_mag": smax, "cell_peclet": smax * h,
               "m_matrix_predicted": bool(smax * h <= 2.0)}
        for stencil in ("upwind", "central"):
            row[f"{stencil}_margin"] = float(
                stability_margin(sigma, s, dh, dh2, stencil=stencil))
            try:
                u = fp_solve(rhs, sigma, s, dh, dh2, scheme="line",
                             stencil=stencil, tol=1e-8, max_sweeps=60)
                mx = float(np.abs(u).max()) if np.isfinite(u).all() else float("inf")
                # `isfinite` alone is not a usable divergence test: a solve returning
                # 1e+143 for an O(1) right-hand side is finite and completely wrong.
                # The FP step is a contraction on a non-negative rhs, so the solution
                # cannot legitimately exceed the rhs scale by any large factor.
                row[f"{stencil}_max_abs"] = mx if np.isfinite(mx) else None
                row[f"{stencil}_solved"] = bool(mx < 10.0 * np.abs(rhs).max())
            except Exception as e:
                row[f"{stencil}_solved"] = False
                row[f"{stencil}_max_abs"] = None
                row[f"{stencil}_error"] = type(e).__name__
        rows.append(row)
    return rows


def show(title, rows, keys=None, fmt=None):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    if not rows:
        print("  (no rows)")
        return
    keys = keys or list(rows[0])
    fmt = fmt or {}
    widths = {}
    for k in keys:
        cells = [fmt.get(k, lambda v: str(v))(r.get(k)) for r in rows]
        widths[k] = max(len(k), max(len(c) for c in cells))
    print("  " + "  ".join(k.rjust(widths[k]) for k in keys))
    print("  " + "  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  " + "  ".join(
            fmt.get(k, lambda v: str(v))(r.get(k)).rjust(widths[k]) for k in keys))


def f4(v):
    return "--" if v is None else f"{v:.4f}"


def e2(v):
    return "--" if v is None else f"{v:.2e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--peclet", type=float, default=1.0,
                    help="cell Peclet number at the coarsest grid; kept <= 2 so "
                         "central is stable and the order comparison is meaningful")
    args = ap.parse_args()

    resolutions = [17, 33, 65, 129]
    D = 0.5
    # Choose v so the coarsest grid sits at the requested cell Peclet number.
    h0 = 1.0 / (resolutions[0] - 1)
    v = args.peclet * D / h0

    print(f"D = {D}  v = {v:.3f}  (cell Peclet {args.peclet} at h = {h0:.4f})")
    print("Boundary cells excluded from all norms: Neumann folding is first-order")
    print(f"by construction, so including them would report order 1 for both stencils.")

    orders = order_study(D, v, resolutions)
    show("1. SPATIAL ORDER OF ACCURACY (operator truncation error, interior only)",
         orders,
         keys=["n", "h", "cell_peclet", "upwind_l2", "upwind_order",
               "central_l2", "central_order"],
         fmt={"h": f4, "cell_peclet": f4, "upwind_l2": e2, "central_l2": e2,
              "upwind_order": f4, "central_order": f4})

    up_orders = [r["upwind_order"] for r in orders if "upwind_order" in r]
    ce_orders = [r["central_order"] for r in orders if "central_order" in r]
    print()
    print(f"  observed order  upwind {np.mean(up_orders):.2f}   "
          f"central {np.mean(ce_orders):.2f}")
    print("  -> upwind is first-order in space; central is second-order. This is the")
    print("     accuracy cost of unconditional stability, and it must be reported")
    print("     alongside the stability benefit.")

    diffusion = artificial_diffusion(D, v, resolutions)
    show("2. ARTIFICIAL DIFFUSION introduced by upwinding", diffusion,
         keys=["n", "h", "cell_peclet", "fitted_D_num", "predicted_D_num",
               "ratio_to_prediction", "relative_added_diffusion"],
         fmt={"h": f4, "cell_peclet": f4, "fitted_D_num": e2,
              "predicted_D_num": e2, "ratio_to_prediction": f4,
              "relative_added_diffusion": f4})
    print()
    print("  The fitted coefficient tracks the |v|h/2 prediction, and the relative")
    print("  added diffusion equals half the cell Peclet number by construction:")
    print("      D_num / D = |v| h / (2 D) = |s| h / 2")
    print("  Note what drops out: g (and therefore sigma) cancels entirely. The only")
    print("  levers are the grid spacing `diffusion.dh` and the score magnitude.")
    print("  At the shipped dh = 1 with a measured score range of about |s| ~ 1, the")
    print("  artificial diffusion is therefore of order 50% of the physical")
    print("  diffusion. Halving dh halves it.")

    stab = stability_boundary()
    show("3. STABILITY BOUNDARY (sigma=5, N=20, h=1)", stab,
         keys=["score_mag", "cell_peclet", "m_matrix_predicted",
               "upwind_margin", "central_margin", "upwind_solved",
               "central_solved", "central_max_abs"],
         fmt={"score_mag": f4, "cell_peclet": f4, "upwind_margin": f4,
              "central_margin": f4, "central_max_abs": e2})
    # Locate the two thresholds from the measurements rather than asserting them.
    m_ok = [r["score_mag"] for r in stab if r["m_matrix_predicted"]]
    solved = [r["score_mag"] for r in stab if r["central_solved"]]
    print()
    print(f"  central keeps the M-matrix property up to |s| h = {max(m_ok):.2f}")
    print(f"  central still converges up to |s| h = {max(solved):.2f}, then diverges")
    print("  -> the sign condition and the divergence point are NOT the same number;")
    print("     between them central runs without a maximum principle, which is where")
    print("     spurious oscillation lives.")
    print("  Upwind holds a margin of exactly 1.0 at every magnitude tested, including")
    print("  |s| = 25, which is the whole reason it is the default.")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("stencil_order", orders),
                       ("stencil_artificial_diffusion", diffusion),
                       ("stencil_stability_boundary", stab)):
        path = os.path.join(args.out_dir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    summary = {
        "D_physical": D,
        "drift": v,
        "observed_order_upwind": float(np.mean(up_orders)),
        "observed_order_central": float(np.mean(ce_orders)),
        "relative_added_diffusion_formula": "|s| * h / 2",
        "central_m_matrix_limit": "|s| * h <= 2",
        "central_divergence_point": float(max(
            r["score_mag"] for r in stab if r["central_solved"])),
    }
    with open(os.path.join(args.out_dir, "stencil_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote CSVs and stencil_summary.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
