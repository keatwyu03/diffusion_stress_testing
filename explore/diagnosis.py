import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_default_config
from data.data_processor import DataProcessor

_cfg        = get_default_config()
h_threshold = _cfg.hfunction.event_threshold
tickers     = _cfg.data.tickers[1:]
csv_path    = _cfg.data.csv_path

dp = DataProcessor(
    csv_path        = csv_path,
    tickers         = _cfg.data.tickers,
    seq_len         = _cfg.data.seq_len,
    test_days       = _cfg.data.test_days,
    start_date      = _cfg.data.start_date,
    end_date        = _cfg.data.end_date,
    train_end_date  = _cfg.data.train_end_date,
    window_shift    = _cfg.data.window_shift,
    winsorize_lower = _cfg.data.winsorize_lower,
    winsorize_upper = _cfg.data.winsorize_upper,
)

dp.load_returns()
# tickers[0] is already the chosen conditioning series (latent state or raw
# macro variable) — import_data.py bakes it into the csv at build time
dp.r_dw = dp.df[_cfg.data.tickers[1:]]  # standardize/sequence on stock cols only
dp.standardize()
dp.winsorize()
dp.make_sequences()
dp.train_test_split()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnosis_plots")
os.makedirs(save_dir, exist_ok=True)

# ── Winsorized standardized log returns (time series, one panel per asset) ────
test_start_date = dp.y_dates_test[0]
n_wins_assets = len(dp.df_z_wins.columns)
fig_wins, axes_wins = plt.subplots(n_wins_assets, 1, figsize=(14, 3 * n_wins_assets), sharex=True)
if n_wins_assets == 1:
    axes_wins = [axes_wins]
for ax, col in zip(axes_wins, dp.df_z_wins.columns):
    ax.plot(dp.df_z_wins.index, dp.df_z_wins[col], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(test_start_date, color="red", linestyle="--", linewidth=1.2, label="Test start")
    ax.set_title(col.upper(), fontsize=10, fontweight="bold")
    ax.set_ylabel("Standardized Return")
    ax.legend(fontsize=8)
axes_wins[-1].set_xlabel("Date")
fig_wins.suptitle("Winsorized Standardized Log Returns", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "winsorized_standardized_returns.png"), dpi=150, bbox_inches="tight")

sq_resid = dp.df_z_wins ** 2
n_sq_assets = len(sq_resid.columns)

# ── ACF of squared residuals (one panel per asset) — volatility clustering ────
acf_n_lags = 20
fig_acf, axes_acf = plt.subplots(n_sq_assets, 1, figsize=(14, 3 * n_sq_assets))
if n_sq_assets == 1:
    axes_acf = [axes_acf]
for ax, col in zip(axes_acf, sq_resid.columns):
    series = sq_resid[col].values
    plot_acf(series, lags=acf_n_lags, ax=ax, title=col.upper())
    ci_bound = 1.96 / np.sqrt(len(series))
    ax.axhline(ci_bound, color="red", linestyle="--", linewidth=1, label="95% rejection boundary")
    ax.axhline(-ci_bound, color="red", linestyle="--", linewidth=1)
    ax.legend(fontsize=8)
    ax.set_ylabel("ACF")
axes_acf[-1].set_xlabel("Lag")
fig_acf.suptitle("ACF of Squared Residuals — Volatility Clustering", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "acf_squared_residuals.png"), dpi=150, bbox_inches="tight")

Z_start, Z_end, valid_idx = dp.get_z_windows()

# event_threshold is specified as "top X% of change" (e.g. 0.10 = top 10%),
# converted here to the equivalent raw numeric cutoff — see main.py for details.
event_top_fraction = h_threshold
event_type = _cfg.hfunction.event_type
h_threshold = dp.get_event_threshold_from_percentile(event_top_fraction, event_type)
print(f"Event threshold: top {event_top_fraction:.1%} -> {h_threshold:.4f} std ({event_type})")


def _event_metric_and_mask(z_start, z_end, event_type, threshold):
    if event_type == "abs_change":
        metric = abs(z_end - z_start)
        return metric, metric >= threshold
    elif event_type == "absval":
        metric = abs(z_end)
        return metric, metric >= threshold
    elif event_type == "upper_change":
        metric = z_end - z_start
        return metric, metric >= threshold
    elif event_type == "lower_change":
        metric = z_end - z_start
        return metric, metric <= -threshold
    else:
        raise NotImplementedError(f"event_type={event_type!r} not supported here.")


n_train_windows = len(dp.X_train)
n_valid         = len(valid_idx)
train_metric_t, train_event_mask_t = _event_metric_and_mask(Z_start, Z_end, event_type, h_threshold)
train_metric    = train_metric_t.numpy()
n_events        = int(train_event_mask_t.sum())

print(f"Training windows:              {n_train_windows}")
print(f"Valid macro windows:           {n_valid}  ({100*n_valid/n_train_windows:.1f}%)")
print(f"Events ({event_type} vs {h_threshold:.4f}): {n_events} / {n_valid}  ({100*n_events/n_valid:.1f}%)")

# Test window event count
cfg = _cfg
macro_col = cfg.data.tickers[0]
macro_raw = dp.df[macro_col]
z_mean = macro_raw.iloc[:-cfg.data.test_days].dropna().mean()
z_std  = macro_raw.iloc[:-cfg.data.test_days].dropna().std()
macro_std_vals = ((macro_raw - z_mean) / z_std).values
n_total = len(macro_std_vals)
n_train = n_total - cfg.data.test_days

test_metric, test_end_dates, test_event_list = [], [], []
test_events, test_valid = 0, 0
for i in range(n_train, n_total - dp.seq_len + 1):
    z_start = macro_std_vals[i]
    end_idx = i + dp.seq_len - 1
    z_end   = macro_std_vals[end_idx]
    if np.isnan(z_start) or np.isnan(z_end):
        continue
    m, is_event = _event_metric_and_mask(float(z_start), float(z_end), event_type, h_threshold)
    test_metric.append(m)
    test_event_list.append(bool(is_event))
    test_end_dates.append(dp.df.index[end_idx])
    test_valid += 1
    if is_event:
        test_events += 1

n_test_windows = len(dp.X_test)
print(f"\nTest windows:                  {n_test_windows}")
print(f"Valid macro windows:           {test_valid}  ({100*test_valid/max(n_test_windows,1):.1f}%)")
print(f"Events ({event_type} vs {h_threshold:.4f}): {test_events} / {max(test_valid,1)}  ({100*test_events/max(test_valid,1):.1f}%)")

test_metric      = np.array(test_metric)
test_event_mask  = np.array(test_event_list)
train_end_dates  = [dp.df.index[int(i) + dp.seq_len - 1] for i in valid_idx.numpy()]
train_event_mask = train_event_mask_t.numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

metric_label = {"abs_change": "|ΔZ|", "absval": "|Z_end|",
                "upper_change": "ΔZ", "lower_change": "ΔZ"}[event_type]

for ax, dates, changes, event_mask, title, n_total_win in [
    (ax1, train_end_dates, train_metric, train_event_mask, "Train", n_train_windows),
    (ax2, test_end_dates,  test_metric,  test_event_mask,  "Test",  n_test_windows),
]:
    n_ev = int(event_mask.sum())
    n_v  = len(changes)
    ax.scatter(dates, changes, s=10, color="steelblue", label="valid window")
    ax.scatter(
        [d for d, e in zip(dates, event_mask) if e],
        changes[event_mask],
        s=15, color="red", label=f"event ({event_type})"
    )
    ax.axhline(h_threshold, color="red", linestyle="--")
    if event_type == "lower_change":
        ax.axhline(-h_threshold, color="red", linestyle="--")
    ax.legend()
    ax.set_title(f"{metric_label} — {title} windows")
    ax.annotate(
        f"Valid: {n_v} / {n_total_win}  ({100*n_v/max(n_total_win,1):.1f}%)\n"
        f"Events: {n_ev} / {n_v}  ({100*n_ev/max(n_v,1):.1f}%)",
        xy=(0.5, -0.12), xycoords='axes fraction', ha='center'
    )

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "event_detection.png"), dpi=150, bbox_inches="tight")

# ── Correlation matrices — last-day returns, unconditional vs conditional ─────
plot_tickers = tickers  # _cfg.data.tickers[1:], the stock columns

# event masks aligned 1:1 with X_train / X_test
Zs_tr, Ze_tr, vidx_tr = dp.get_z_windows_train_aligned()
Zs_te, Ze_te, vidx_te = dp.get_z_windows_test()
_, mask_tr = _event_metric_and_mask(Zs_tr, Ze_tr, event_type, h_threshold)
_, mask_te = _event_metric_and_mask(Zs_te, Ze_te, event_type, h_threshold)

train_last_day = dp.X_train[:, -1, :].numpy()
test_last_day  = dp.X_test[:, -1, :].numpy()
cond_train_last_day = dp.X_train[vidx_tr][mask_tr][:, -1, :].numpy()
cond_test_last_day  = dp.X_test[vidx_te][mask_te][:, -1, :].numpy()

fig_corr, axes_corr = plt.subplots(2, 2, figsize=(12, 10))
tick_lbl = [t.upper() for t in plot_tickers]

for ax, data, title in [
    (axes_corr[0, 0], train_last_day,      "Train — unconditional"),
    (axes_corr[0, 1], cond_train_last_day, "Train — conditional (events)"),
    (axes_corr[1, 0], test_last_day,       "Test — unconditional"),
    (axes_corr[1, 1], cond_test_last_day,  "Test — conditional (events)"),
]:
    corr = np.corrcoef(data.T)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(plot_tickers))); ax.set_xticklabels(tick_lbl)
    ax.set_yticks(range(len(plot_tickers))); ax.set_yticklabels(tick_lbl)
    ax.set_title(f"{title} — Last-Day Return Correlation (n={data.shape[0]})",
                 fontsize=10, fontweight="bold")
    for r in range(len(plot_tickers)):
        for c in range(len(plot_tickers)):
            ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center",
                     fontweight="bold", color="white" if abs(corr[r, c]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlation_matrices.png"), dpi=150, bbox_inches="tight")
