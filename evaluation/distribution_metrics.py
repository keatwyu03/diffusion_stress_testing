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
    start_date=config.data.start_date,
    end_date=config.data.end_date,
    train_end_date=config.data.train_end_date,
    window_shift=config.data.window_shift,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)
data_processor.process_all()

tickers = config.data.tickers          # all assets, e.g. ["unemp", "sp500", "baa"]
n_assets = len(tickers) - 1

# X shape: (N, T, A)   gen shape: (N, A, T)
X_train = data_processor.X_train
X_test  = data_processor.X_test

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)

gen_train = torch.load(os.path.join(_root, 'generated_samples_train.pt'), map_location='cpu')
gen_test  = torch.load(os.path.join(_root, 'generated_samples_test.pt'),  map_location='cpu')

# Event masks from the real conditioning series (X has no macro channel), same as
# main.py; the "top X%" fraction is converted to a raw cutoff first. Evaluation
# always uses the HARD event definition regardless of constraint_mode — "soft"
# only changes the h-function's training labels, not what an event is.
event_type  = config.hfunction.event_type
h_threshold = data_processor.get_event_threshold_from_percentile(
    config.hfunction.event_threshold, event_type)
print(f"Event threshold: top {config.hfunction.event_threshold:.1%} -> "
      f"{h_threshold:.4f} std ({event_type})")

def event_mask(Z_start, Z_end):
    if event_type == "abs_change":
        return (Z_end - Z_start).abs() >= h_threshold
    elif event_type == "absval":
        return Z_end.abs() >= h_threshold
    elif event_type == "upper_change":
        return Z_end - Z_start >= h_threshold
    elif event_type == "lower_change":
        return Z_end - Z_start <= -h_threshold
    raise NotImplementedError(f"event_type={event_type!r}")

Zs_tr, Ze_tr, vidx_tr = data_processor.get_z_windows_train_aligned()
Zs_te, Ze_te, vidx_te = data_processor.get_z_windows_test()
X_train_events = X_train[vidx_tr][event_mask(Zs_tr, Ze_tr)]
X_test_events  = X_test[vidx_te][event_mask(Zs_te, Ze_te)]

print(f"Train event windows: {len(X_train_events)} / {len(X_train)}")
print(f"Test  event windows: {len(X_test_events)}  / {len(X_test)}")


def wasserstein_lastday(X_events, gen):
    results = {}
    for ch, ticker in enumerate(tickers[1:]):
        real = X_events[:, -1, ch].numpy()
        g = gen[:, ch, -1].numpy()
        results[ticker] = wasserstein_distance(real, g)
    return results

w_train = wasserstein_lastday(X_train_events, gen_train)
w_test = wasserstein_lastday(X_test_events, gen_test)

print("\nWasserstein Distance — Last-Day Marginals")
print(f"{'Asset':<10} {'Train':>10} {'Test':>10}")
print("-" * 32)
for ticker in tickers[1:]:
    print(f"{ticker:<10} {w_train[ticker]:>10.4f} {w_test[ticker]:>10.4f}")


fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')
table_data = [[ticker, f"{w_train[ticker]:.4f}", f"{w_test[ticker]:.4f}"] for ticker in tickers[1:]]
table = ax.table(
    cellText=table_data,
    colLabels=["Asset", "Train", "Test"],
    loc='center',
    cellLoc='center'
)
table.scale(1, 1.5)
fig.suptitle("Wasserstein Distance — Last-Day Marginals", fontsize=12, fontweight="bold")
os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
out = os.path.join(_dir, "results", "wasserstein_table.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")



def fraction(vals):
    s = np.sort(vals)
    n = len(s)
    p = np.arange(n, 0, -1)/n
    return s, p



def plot_tail_logs():
    os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
    splits = [
        (X_train, mask_train, gen_train, "Train"),
        (X_test,  mask_test,  gen_test,  "Test"),
    ]
    
    fig, axes = plt.subplots(n_assets, 2, figsize=(12, 4 * n_assets))
    if n_assets == 1:
        axes = axes[np.newaxis, :]
        

    for col, (X, mask, gen, split_label) in enumerate(splits):

        for ch, ticker, in enumerate(tickers[1:]):
            real = np.abs(X[mask, -1, ch].numpy())
            gen_vals = np.abs(gen[:, ch, -1].numpy())

            ax = axes[ch, col]
            for vals, color, label in [(real, "darkorange", "Real"), (gen_vals, "steelblue", "Generated")]:
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
    log_p = np.log(p[s > 0])
    slope, _ = np.polyfit(log_s, log_p, 1)
    return -1 * slope
