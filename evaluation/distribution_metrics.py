import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
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


def wasserstein_lastday(X, mask, gen):
    results = {}
    for ch, ticker in enumerate(tickers):
        real = X[mask, -1, ch].numpy()
        g = gen[:, ch, -1].numpy()
        results[ticker] = wasserstein_distance(real, g)
    return results

w_train = wasserstein_lastday(X_train, mask_train, gen_train)
w_test = wasserstein_lastday(X_test, mask_test, gen_test)

print("\nWasserstein Distance — Last-Day Marginals")
print(f"{'Asset':<10} {'Train':>10} {'Test':>10}")
print("-" * 32)
for ticker in tickers:
    print(f"{ticker:<10} {w_train[ticker]:>10.4f} {w_test[ticker]:>10.4f}")


def fraction(vals):
    s = np.sort(vals)
    n = len(s)
    p = np.arrange(n, 0, -1)/n
    return s, p



def plot_tail_logs():
    os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
    splits = [
        (X_train, mask_train, gen_train, "Train"),
        (X_test,  mask_test,  gen_test,  "Test"),
    ]
    for X, mask, gen, split_label in splits:
        fig, axes = plt.subplots(n_assets, 2, figsize=(12, 4 * n_assets))
        if n_assets == 1:
            axes = [axes]
        
        for ch, ticker, in enumerate(tickers):
            real = np.abs(X[mask, -1, ch].numpy())
            gen = np.abs(X[mask, ch, -1].numpy())

            ax = axes[ch]
            for vals, color, label in [(real, "darkorange", "Real"), (gen, "steelblue", "Generated")]:
                s, p = fraction(vals)
                ax.plot(s, p, color = color, linewidth = 1.5, label = label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{ticker.upper()} — {split_label}", fontsize=10, fontweight="bold")
            ax.set_xlabel("log|return|")
            ax.set_ylabel("log P(|R| > x)")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)


        fig.suptitle(f"Log-Log Tail Plot — Last-Day Returns ({split_label})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = os.path.join(_dir, "results", f"tail_loglog_{split_label.lower()}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved {out}")

plot_tail_logs()

def tail_index(vals):
    s, p = fraction(np.abs(vals))
    log_s = np.log(s[s > 0])
    log_p = np.log(p[p > 0])
    slope, _ = np.polyfit(log_s, log_p, 1)
    return -1 * slope
