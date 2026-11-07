"""
Per-asset last-day marginal distribution comparison: Real vs Generated, for a
chosen pair (or more) of methods. This is a visual complement to the single
aggregate wasserstein1_marginal number in group_bc_comparison_*.png -- that
number averages over all 15 assets and can hide shape differences (shift,
width, tails) that a histogram makes obvious.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(_HERE), "main_results", "exp2_results")
TICKERS = common.TICKERS
COLORS = {"real": "#555555"}
METHOD_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def load(method):
    d = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[method], "diffusion")
    gen = np.load(os.path.join(d, "generated_samples.npy"))  # (n_test, K, A, T)
    real = np.load(os.path.join(d, "test_data.npz"))["X_test"]  # (n_test, A, T)
    return real, gen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default="s0,s2")
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    real, _ = load(methods[0])
    real_last = real[:, :, -1]  # (n_test, A)
    gens_last = {}
    for m in methods:
        _, gen = load(m)
        gens_last[m] = gen[:, :, :, -1].reshape(-1, gen.shape[2])  # (n_test*K, A)

    n_assets = len(TICKERS)
    ncols = 5
    nrows = int(np.ceil(n_assets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = axes.ravel()

    for a, ticker in enumerate(TICKERS):
        ax = axes[a]
        r = real_last[:, a]
        lo = min(r.min(), *[gens_last[m][:, a].min() for m in methods])
        hi = max(r.max(), *[gens_last[m][:, a].max() for m in methods])
        bins = np.linspace(lo, hi, args.bins + 1)
        ax.hist(r, bins=bins, density=True, histtype="step", color=COLORS["real"],
                linewidth=2, label="Real")
        for i, m in enumerate(methods):
            g = gens_last[m][:, a]
            w1 = wasserstein_distance(r, g)
            ax.hist(g, bins=bins, density=True, histtype="step", color=METHOD_COLORS[i % len(METHOD_COLORS)],
                    linewidth=1.3, label=f"{m.upper()} (W1={w1:.3g})")
        ax.set_title(ticker, fontsize=10)
        ax.tick_params(labelsize=7)
        if a == 0:
            ax.legend(fontsize=6.5, loc="upper right")

    for a in range(n_assets, len(axes)):
        axes[a].axis("off")

    fig.suptitle(f"Last-Day Marginal Returns: Real vs {' vs '.join(m.upper() for m in methods)}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT_DIR, f"marginals_{'_'.join(methods)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
