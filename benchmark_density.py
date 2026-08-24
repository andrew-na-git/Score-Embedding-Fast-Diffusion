"""Density-estimator scaling benchmark.

Regenerates `figures/kde_scaling.csv` and `.pdf`, which were previously committed
with no generating script. It also exists to correct a claim: the README and the
old `_kde_log_density` docstring stated the histogram estimator is "~250x faster"
than scipy at 64x64, but the committed CSV measured 2.5x at 32x32 and never timed
scipy at 64x64 at all.

What the measurements actually show
-----------------------------------
The histogram estimator bins into a fixed n_bins x n_bins grid and smooths it, so
its cost is dominated by the grid, not the image: it is close to *constant in
image resolution*. scipy's exact Gaussian KDE is O(N^2) in the pixel count. The
consequences are:

  - below roughly 24x24 the histogram estimator is SLOWER, because the fixed grid
    cost exceeds the exact computation on a small point set;
  - the crossover sits near 24x24;
  - past 32x32 scipy becomes impractical and the ratio grows without bound.

That is a good result stated correctly. Run this and quote it; do not quote 250x.

Usage
-----
    python benchmark_density.py                       # default sweep
    python benchmark_density.py --max-scipy 64        # allow scipy up to 64x64
    python benchmark_density.py --repeats 5 --no-plot
"""

import argparse
import csv
import os
import time

import numpy as np

from fast_diffusion.model.density import (
    histogram_log_density, image_channel_to_samples, scipy_log_density,
    sklearn_log_density,
)

DEFAULT_RESOLUTIONS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def make_channel(res, seed=0):
    """A smooth-plus-noise test channel in [0, 1], resembling real image statistics."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float64)
    img = (
        0.5
        + 0.25 * np.sin(2 * np.pi * xx / max(res / 4.0, 1))
        * np.cos(2 * np.pi * yy / max(res / 3.0, 1))
        + 0.05 * rng.standard_normal((res, res))
    )
    return np.clip(img, 0.0, 1.0)


def time_estimator(fn, xy, repeats=3, warmup=True):
    """Median wall-clock over `repeats` calls, in milliseconds.

    A warmup call is essential: `scipy_log_density` and `sklearn_log_density`
    import their backends lazily, and `scipy.stats` alone costs ~10 s to import.
    Without a warmup that import lands in the first timed measurement and produced
    a nonsensical 10,056 ms for a 64-sample problem.
    """
    if warmup:
        fn(xy)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(xy)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples))


def run(resolutions, max_scipy=32, max_sklearn=128, repeats=3, verbose=True):
    """Benchmark each estimator across resolutions.

    `max_scipy` / `max_sklearn` cap where the expensive estimators are attempted;
    entries above the cap are recorded as None rather than silently omitted, so
    the resulting table cannot be misread as a completed comparison.
    """
    rows = []
    for res in resolutions:
        xy = image_channel_to_samples(make_channel(res))
        n = xy.shape[1]

        hist_ms = time_estimator(histogram_log_density, xy, repeats)
        scipy_ms = (
            time_estimator(scipy_log_density, xy, repeats) if res <= max_scipy else None
        )
        sklearn_ms = (
            time_estimator(sklearn_log_density, xy, repeats)
            if res <= max_sklearn else None
        )

        row = {
            "resolution": f"{res}x{res}",
            "n_pixels": n,
            "histogram_ms": round(hist_ms, 3),
            "scipy_ms": None if scipy_ms is None else round(scipy_ms, 3),
            "sklearn_ms": None if sklearn_ms is None else round(sklearn_ms, 3),
            "speedup_vs_scipy": None if scipy_ms is None else round(scipy_ms / hist_ms, 3),
            "speedup_vs_sklearn": (
                None if sklearn_ms is None else round(sklearn_ms / hist_ms, 3)
            ),
        }
        rows.append(row)

        if verbose:
            sc = "n/a" if scipy_ms is None else f"{scipy_ms:9.2f}"
            sk = "n/a" if sklearn_ms is None else f"{sklearn_ms:9.2f}"
            sp = "n/a" if scipy_ms is None else f"{scipy_ms / hist_ms:6.2f}x"
            print(
                f"  {res:4}x{res:<4} n={n:8,}  hist={hist_ms:8.2f} ms  "
                f"scipy={sc} ms  sklearn={sk} ms  hist speedup vs scipy={sp}"
            )

    return rows


def crossover(rows):
    """Smallest resolution at which the histogram estimator becomes faster."""
    for r in rows:
        if r["speedup_vs_scipy"] is not None and r["speedup_vs_scipy"] > 1.0:
            return r["resolution"]
    return None


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "resolution", "n_pixels", "histogram_ms", "scipy_ms", "sklearn_ms",
        "speedup_vs_scipy", "speedup_vs_sklearn",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n_pixels"] for r in rows]
    hist = [r["histogram_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)

    ax1.plot(n, hist, "o-", label="histogram + FFT")
    for key, label in [("scipy_ms", "scipy gaussian_kde"), ("sklearn_ms", "sklearn KernelDensity")]:
        xs = [r["n_pixels"] for r in rows if r[key] is not None]
        ys = [r[key] for r in rows if r[key] is not None]
        if xs:
            ax1.plot(xs, ys, "s--", label=label)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("pixels per channel (N)")
    ax1.set_ylabel("wall-clock (ms)")
    ax1.set_title("Log-density estimation cost")
    ax1.legend()
    ax1.grid(alpha=0.3)

    xs = [r["resolution"] for r in rows if r["speedup_vs_scipy"] is not None]
    ys = [r["speedup_vs_scipy"] for r in rows if r["speedup_vs_scipy"] is not None]
    ax2.bar(xs, ys)
    ax2.axhline(1.0, color="k", ls="--", lw=1, label="parity")
    ax2.set_ylabel("speedup over scipy (x)")
    ax2.set_title("Histogram vs exact KDE\n(below parity = histogram is slower)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--max-scipy", type=int, default=32,
                        help="largest resolution at which to attempt scipy's O(N^2) KDE")
    parser.add_argument("--max-sklearn", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("Density-estimator scaling benchmark")
    print(f"  repeats={args.repeats}  max_scipy={args.max_scipy}x{args.max_scipy}")
    print()

    rows = run(
        args.resolutions, max_scipy=args.max_scipy, max_sklearn=args.max_sklearn,
        repeats=args.repeats,
    )

    csv_path = os.path.join(args.out_dir, "kde_scaling.csv")
    write_csv(rows, csv_path)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        pdf_path = os.path.join(args.out_dir, "kde_scaling.pdf")
        write_plot(rows, pdf_path)
        print(f"wrote {pdf_path}")

    xo = crossover(rows)
    print()
    print(f"Crossover (histogram first becomes faster): {xo}")
    print("Quote the measured crossover and the resolutions actually timed.")
    print("Do NOT quote the old '~250x at 64x64' figure -- it was never measured.")


if __name__ == "__main__":
    main()
