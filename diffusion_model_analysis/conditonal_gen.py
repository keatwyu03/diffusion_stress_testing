import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor

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

tickers = config.data.tickers          # all assets, e.g. ["unemp", "sp500", "baa"]
n_assets = len(tickers)

# X shape: (N, T, A)   gen shape: (N, A, T)
X_train = data_processor.X_train
X_test  = data_processor.X_test

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
gen_train = torch.load(os.path.join(_root, 'generated_samples_train.pt'), map_location='cpu')
gen_test  = torch.load(os.path.join(_root, 'generated_samples_test.pt'),  map_location='cpu')


def get_mask(X):
    last_window = X[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        return last_window.sum(dim=1) <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        return (last_window[:, -1] - last_window[:, 0]).abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        return last_window[:, -1].abs() >= config.hfunction.event_threshold


mask_train = get_mask(X_train)
mask_test  = get_mask(X_test)

print(f"Train event windows: {mask_train.sum().item()} / {len(mask_train)}")
print(f"Test  event windows: {mask_test.sum().item()}  / {len(mask_test)}")


def kde_plot(ax, real_vals, gen_vals, real_label, gen_label, xlabel):
    x_min = min(real_vals.min(), gen_vals.min()) - 0.5
    x_max = max(real_vals.max(), gen_vals.max()) + 0.5
    x = np.linspace(x_min, x_max, 500)
    for vals, color, label in [
        (real_vals, "darkorange", real_label),
        (gen_vals,  "steelblue",  gen_label),
    ]:
        kde = gaussian_kde(vals, bw_method="silverman")
        ax.plot(x, kde(x), color=color, linewidth=2, label=label)
        ax.hist(vals, bins=40, density=True, alpha=0.2, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def make_figure(extract_fn, suptitle, filename, xlabel):
    """
    extract_fn(X, mask, gen, ch) -> (real_vals, gen_vals) as numpy arrays
    Produces a (n_assets x 2) figure: left=train, right=test.
    """
    fig, axes = plt.subplots(n_assets, 2, figsize=(14, 4 * n_assets))
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    splits = [
        (0, X_train, mask_train, gen_train, "In-Sample (Train)"),
        (1, X_test,  mask_test,  gen_test,  "Out-of-Sample (Test)"),
    ]

    for row, ticker in enumerate(tickers):
        ch = row
        for col, X, mask, gen, split_label in splits:
            ax = axes[row, col]
            real_vals, gen_vals = extract_fn(X, mask, gen, ch)

            kde_plot(
                ax, real_vals, gen_vals,
                real_label=f"Real event windows (n={len(real_vals)})",
                gen_label =f"Conditional generated (n={len(gen_vals)})",
                xlabel=xlabel,
            )
            ax.set_title(
                f"{ticker.upper()} — {split_label}",
                fontsize=11, fontweight="bold"
            )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
    out = os.path.join(_dir, "results", filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# ── Figure 1: Last-day returns ─────────────────────────────────────────────────
def extract_lastday(X, mask, gen, ch):
    real = X[mask, -1, ch].numpy()       # (N_event,)
    g    = gen[:, ch, -1].numpy()        # (N_gen,)
    return real, g

make_figure(
    extract_fn=extract_lastday,
    suptitle=(
        f"Conditional vs Real — Last-Day Return  "
        f"[event={config.hfunction.event_type}, thr={config.hfunction.event_threshold}]"
    ),
    filename="conditional_lastday.png",
    xlabel="Standardized Return (day 64)",
)

# ── Figure 2: Cumulative returns (sum over full window) ────────────────────────
def extract_cumsum(X, mask, gen, ch):
    real = X[mask, :, ch].sum(dim=1).numpy()    # (N_event,)
    g    = gen[:, ch, :].sum(dim=1).numpy()     # (N_gen,)
    return real, g

make_figure(
    extract_fn=extract_cumsum,
    suptitle=(
        f"Conditional vs Real — Cumulative Return (64-day sum)  "
        f"[event={config.hfunction.event_type}, thr={config.hfunction.event_threshold}]"
    ),
    filename="conditional_cumulative.png",
    xlabel="Cumulative Standardized Return",
)
