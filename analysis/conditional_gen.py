import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor

_parser = argparse.ArgumentParser()
_parser.add_argument("--gan", action="store_true",
                      help="load the cGAN baseline's generated samples "
                           "(gan_baseline/gan_results/gan_generated_samples_{train,test}.pt) and "
                           "save outputs to analysis/gan_results/, instead of the diffusion model's "
                           "(generated_samples_{train,test}.pt, saved to analysis/diffusion_results/)")
_args = _parser.parse_args()

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
    event_causal=config.data.event_causal,
    event_lag_gap=config.data.event_lag_gap,
)
data_processor.process_all()

# event_threshold is specified as "top X% of |Z_end - Z_start|", converted here to
# the equivalent raw numeric cutoff (train-only, no leakage) — see main.py for details.
event_top_fraction = config.hfunction.event_threshold
config.hfunction.event_threshold = data_processor.get_event_threshold_from_percentile(event_top_fraction, config.hfunction.event_type)
print(f"Event threshold: top {event_top_fraction:.1%} -> {config.hfunction.event_threshold:.4f} std")

tickers = config.data.tickers          # assets only; conditioning series is separate
n_assets = len(tickers)
plot_tickers = tickers
n_plot       = len(plot_tickers)

# X shape: (N, T, A)   gen shape: (N, A, T)
X_train = data_processor.X_train
X_test  = data_processor.X_test

_dir  = os.path.dirname(os.path.abspath(__file__))  # analysis/
_root = os.path.dirname(_dir)                        # repo root
if _args.gan:
    _gen_dir = os.path.join(_root, 'gan_baseline', 'gan_results')
    _train_name, _test_name = 'gan_generated_samples_train.pt', 'gan_generated_samples_test.pt'
    _results_dir = os.path.join(_dir, "gan_results")
else:
    _gen_dir = _root
    _train_name, _test_name = 'generated_samples_train.pt', 'generated_samples_test.pt'
    _results_dir = os.path.join(_dir, "diffusion_results")
gen_train = torch.load(os.path.join(_gen_dir, _train_name), map_location='cpu')
gen_test  = torch.load(os.path.join(_gen_dir, _test_name),  map_location='cpu')


def get_mask(X, Z_start, Z_end, valid_idx):
    # Event mask must come from the real macro series (Z_start/Z_end from
    # get_z_windows), not from X, which is stock-returns-only and has no
    # macro channel at all.
    if config.hfunction.event_type == "abs_change":
        event_valid = (Z_end - Z_start).abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        event_valid = Z_end.abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "upper_change":
        event_valid = (Z_end - Z_start) >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "lower_change":
        event_valid = (Z_end - Z_start) <= -config.hfunction.event_threshold
    elif config.hfunction.event_type == "start_upper":
        event_valid = Z_start >= config.hfunction.event_threshold
    else:
        raise NotImplementedError(
            f"event_type={config.hfunction.event_type!r} not supported by the "
            "macro-based mask; only 'abs_change', 'absval', 'upper_change', 'lower_change', and "
            "'start_upper' are implemented."
        )
    mask = torch.zeros(X.shape[0], dtype=torch.bool)
    mask[valid_idx] = event_valid
    return mask


Z_start_train, Z_end_train, valid_idx_train = data_processor.get_z_windows_train_aligned()
Z_start_test,  Z_end_test,  valid_idx_test  = data_processor.get_z_windows_test()

mask_train = get_mask(X_train, Z_start_train, Z_end_train, valid_idx_train)
mask_test  = get_mask(X_test,  Z_start_test,  Z_end_test,  valid_idx_test)

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
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 4 * n_plot))
    if n_plot == 1:
        axes = axes[np.newaxis, :]

    splits = [
        (0, X_train, mask_train, gen_train, "In-Sample (Train)"),
        (1, X_test,  mask_test,  gen_test,  "Out-of-Sample (Test)"),
    ]

    for row, (ch, ticker) in enumerate(zip(range(n_assets), plot_tickers)):
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
    os.makedirs(_results_dir, exist_ok=True)
    out = os.path.join(_results_dir, filename)
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
    xlabel=f"Standardized Return (day {config.data.seq_len})",
)

# ── Figure 2: Cumulative returns (sum over full window) ────────────────────────
def extract_cumsum(X, mask, gen, ch):
    real = X[mask, :, ch].sum(dim=1).numpy()    # (N_event,)
    g    = gen[:, ch, :].sum(dim=1).numpy()     # (N_gen,)
    return real, g

make_figure(
    extract_fn=extract_cumsum,
    suptitle=(
        f"Conditional vs Real — Cumulative Return ({config.data.seq_len}-day sum)  "
        f"[event={config.hfunction.event_type}, thr={config.hfunction.event_threshold}]"
    ),
    filename="conditional_cumulative.png",
    xlabel="Cumulative Standardized Return",
)

# ── Diagnostics table ─────────────────────────────────────────────────────────
rows = []
splits = [
    (0, X_train, mask_train, gen_train, "In-Sample (Train)"),
    (1, X_test,  mask_test,  gen_test,  "Out-of-Sample (Test)"),
]
for i, ticker in zip(range(n_assets), plot_tickers):
    for col, X, mask, gen, split_label in splits:
        real_last = X[mask, -1, i].numpy()
        gen_last  = gen[:, i, -1].numpy()
        real_cum  = X[mask, :, i].sum(dim=1).numpy()
        gen_cum   = gen[:, i, :].sum(dim=1).numpy()

        for kind, vals_last, vals_cum in [
            ("real event",  real_last, real_cum),
            ("generated",   gen_last,  gen_cum),
        ]:
            qs_l = np.quantile(vals_last, [.01, .05, .5, .95, .99]).round(3)
            qs_c = np.quantile(vals_cum,  [.01, .05, .5, .95, .99]).round(3)
            rows.append([
                ticker.upper() if kind == "real event" else "",
                split_label if kind == "real event" else "",
                kind,
                f"{vals_last.mean():.3f}", f"{vals_last.std():.3f}", str(qs_l),
                f"{vals_cum.mean():.3f}",  f"{vals_cum.std():.3f}",  str(qs_c),
            ])

col_labels = ["Asset", "Split", "Kind",
              "Mean (last)", "Std (last)", "q[1,5,50,95,99] last day",
              "Mean (cum)",  "Std (cum)",  f"q[1,5,50,95,99] {config.data.seq_len}-day sum"]
fig_d, ax_d = plt.subplots(figsize=(24, 0.45 * len(rows) + 1.5))
ax_d.axis("off")
tbl = ax_d.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(col_labels))))
fig_d.suptitle(
    f"Conditional Generation — Diagnostics  "
    f"[event={config.hfunction.event_type}, thr={config.hfunction.event_threshold}]",
    fontsize=12, fontweight="bold",
)
fig_d.tight_layout()
os.makedirs(_results_dir, exist_ok=True)
out_diag = os.path.join(_results_dir, "conditional_diagnostics.png")
plt.savefig(out_diag, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_diag}")
