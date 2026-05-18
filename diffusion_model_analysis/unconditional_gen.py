import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import combinations
from matplotlib.patches import Patch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel

config = get_default_config()
data_processor = DataProcessor(
    csv_path=config.data.csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=config.data.test_days,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)
data_processor.process_all()

# Assets and channel indices driven entirely by config
plot_tickers = config.portfolio.portfolio_tickers
ch_idx = {t: config.data.tickers.index(t) for t in plot_tickers}

# Real marginal distributions — last day of each 64-day window
X_train = data_processor.X_train  # (N_train, seq_len, channels)
X_test  = data_processor.X_test   # (N_test,  seq_len, channels)

real = {
    t: {
        "train": X_train[:, -1, config.data.tickers.index(t)].numpy(),
        "test":  X_test[:,  -1, config.data.tickers.index(t)].numpy(),
    }
    for t in plot_tickers
}

# ── Unconditional generation ─────────────────────────────────────────────────
diffusion_model = DiffusionModel(
    in_channels=config.diffusion.in_channels,
    out_channels=config.diffusion.out_channels,
    sample_size=config.diffusion.sample_size,
    layers_per_block=config.diffusion.layers_per_block,
    block_out_channels=config.diffusion.block_out_channels,
    b_min=config.diffusion.b_min,
    b_max=config.diffusion.b_max,
    device=config.diffusion.device,
)
diffusion_model.load("checkpoints/diffusion_model.pt")

N_samples = 500
print(f"Generating {N_samples} unconditional samples...")
uncond = diffusion_model.sample(
    batch_size=N_samples,
    num_steps=config.diffusion.num_steps,
    stoch=1,
).cpu()  # (N_samples, channels, seq_len)

gen = {
    t: uncond[:, ch_idx[t], -1].numpy()
    for t in plot_tickers
}

# ── Marginal distributions ────────────────────────────────────────────────────
n_assets = len(plot_tickers)
fig, axes = plt.subplots(n_assets, 2, figsize=(14, 5 * n_assets))
if n_assets == 1:
    axes = axes[np.newaxis, :]

for row, ticker in enumerate(plot_tickers):
    for col, split in enumerate(["train", "test"]):
        ax = axes[row, col]
        real_vals = real[ticker][split]
        gen_vals  = gen[ticker]

        for vals, color, label in [
            (real_vals, "darkorange", f"Real {split} (n={len(real_vals)})"),
            (gen_vals,  "steelblue",  f"Unconditional generated (n={len(gen_vals)})"),
        ]:
            kde = gaussian_kde(vals, bw_method="silverman")
            x   = np.linspace(min(real_vals.min(), gen_vals.min()) - 0.5,
                               max(real_vals.max(), gen_vals.max()) + 0.5, 500)
            ax.plot(x, kde(x), color=color, linewidth=2, label=label)
            ax.hist(vals, bins=40, density=True, alpha=0.2, color=color)

        split_label = "In-Sample (Train)" if split == "train" else "Out-of-Sample (Test)"
        ax.set_title(f"{ticker.upper()} — {split_label}", fontsize=11)
        ax.set_xlabel("Standardized Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

fig.suptitle("Unconditional Generation: Learned Distribution vs. Real Data", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.savefig("results/score_function_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Joint distributions (all pairwise combinations) ───────────────────────────
pairs   = list(combinations(plot_tickers, 2))
n_pairs = len(pairs)

fig2, axes2 = plt.subplots(n_pairs, 2, figsize=(14, 6 * n_pairs))
if n_pairs == 1:
    axes2 = axes2[np.newaxis, :]

for row, (t1, t2) in enumerate(pairs):
    for col, split in enumerate(["train", "test"]):
        ax = axes2[row, col]

        real_t1 = real[t1][split]
        real_t2 = real[t2][split]
        gen_t1  = gen[t1]
        gen_t2  = gen[t2]

        x_min = min(real_t1.min(), gen_t1.min()) - 0.5
        x_max = max(real_t1.max(), gen_t1.max()) + 0.5
        y_min = min(real_t2.min(), gen_t2.min()) - 0.5
        y_max = max(real_t2.max(), gen_t2.max()) + 0.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80),
                             np.linspace(y_min, y_max, 80))
        grid = np.vstack([xx.ravel(), yy.ravel()])

        zz_real = gaussian_kde(np.vstack([real_t1, real_t2]),
                               bw_method="silverman")(grid).reshape(xx.shape)
        zz_gen  = gaussian_kde(np.vstack([gen_t1,  gen_t2]),
                               bw_method="silverman")(grid).reshape(xx.shape)

        ax.contourf(xx, yy, zz_real, levels=10, cmap="Oranges", alpha=0.5)
        ax.contour(xx, yy, zz_real,  levels=10, colors="darkorange", linewidths=0.8, alpha=0.8)
        ax.contourf(xx, yy, zz_gen,  levels=10, cmap="Blues",   alpha=0.5)
        ax.contour(xx, yy, zz_gen,   levels=10, colors="steelblue",  linewidths=0.8, alpha=0.8)

        ax.legend(handles=[
            Patch(color="darkorange", alpha=0.7, label=f"Real {split} (n={len(real_t1)})"),
            Patch(color="steelblue",  alpha=0.7, label=f"Generated (n={len(gen_t1)})"),
        ], fontsize=9, loc="upper right")

        split_label = "In-Sample (Train)" if split == "train" else "Out-of-Sample (Test)"
        ax.set_title(f"Joint {t1.upper()} × {t2.upper()} — {split_label}", fontsize=11)
        ax.set_xlabel(f"{t1.upper()} Std Return")
        ax.set_ylabel(f"{t2.upper()} Std Return")
        ax.grid(True, alpha=0.3)

fig2.suptitle("Unconditional Generation: Joint Pairwise Distributions", fontsize=13, fontweight="bold")
fig2.tight_layout()
plt.savefig("results/score_function_joint_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
