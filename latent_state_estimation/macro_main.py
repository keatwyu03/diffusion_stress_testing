import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from tracking_regression import TrackingRegression
from state_space import StateSpace

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
sys.path.insert(0, _root)
from config import get_default_config
from data.data_processor import DataProcessor

df_gm = pd.read_csv(os.path.join(_dir, "growth_macro.csv"), index_col=0, parse_dates=True)
df_gd = pd.read_csv(os.path.join(_dir, "growth_daily.csv"), index_col=0, parse_dates=True)

df_im = pd.read_csv(os.path.join(_dir, "inflation_macro.csv"), index_col=0, parse_dates=True)
df_id = pd.read_csv(os.path.join(_dir, "inflation_daily.csv"), index_col=0, parse_dates=True)

param_names = ["b0", "b1", "b2", "a0", "a1", "log_var_y"]

states = {}
for name, macro, daily in [("growth", df_gm, df_gd), ("inflation", df_im, df_id)]:
    tr = TrackingRegression(macro, daily)
    ut = tr.fit()

    ss = StateSpace(y=tr.factor, x=ut)
    ss.fit()

    print(f"\n=== {name} ===")
    print(f"PCA explained variance: {tr.explained_var:.3f}")
    print(f"loglik: {-ss.res.fun:.2f}  converged: {ss.res.success}")
    print(pd.Series(ss.params, index=param_names))

    states[name] = ss.filtered_states()

df_states = pd.concat(states, axis=1)

# standardized average: put both states on the same scale, then average
z = (df_states - df_states.mean()) / df_states.std()
df_states["combined"] = z.mean(axis=1)

df_states.to_csv(os.path.join(_dir, "latent_states.csv"))

fig, axes = plt.subplots(len(df_states.columns), 1, figsize=(14, 12), sharex=True)
for ax, name in zip(axes, df_states.columns):
    ax.plot(df_states.index, df_states[name], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"filtered latent state — {name}")
plt.tight_layout()
plt.savefig(os.path.join(_dir, "latent_states.png"), dpi=150, bbox_inches="tight")


# ── diagnosis: combined latent state as the conditioning macro variable ──────
_cfg = get_default_config()
event_type = _cfg.hfunction.event_type
tickers = _cfg.data.tickers[1:]
macro_col = _cfg.data.tickers[0]

dp = DataProcessor(
    csv_path        = os.path.join(_root, _cfg.data.csv_path),
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
# swap the FRED conditioning series for the combined latent state
dp.df[macro_col] = df_states["combined"].reindex(dp.df.index)
dp.r_dw = dp.df[tickers]
dp.standardize()
dp.winsorize()
dp.make_sequences()
dp.train_test_split()

h_threshold = dp.get_event_threshold_from_percentile(_cfg.hfunction.event_threshold, event_type)
print(f"\nEvent threshold: top {_cfg.hfunction.event_threshold:.1%} -> {h_threshold:.4f} std ({event_type})")


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


# windows aligned 1:1 with X_train / X_test
Zs_tr, Ze_tr, vidx_tr = dp.get_z_windows_train_aligned()
Zs_te, Ze_te, vidx_te = dp.get_z_windows_test()

metric_tr, mask_tr = _event_metric_and_mask(Zs_tr.numpy(), Ze_tr.numpy(), event_type, h_threshold)
metric_te, mask_te = _event_metric_and_mask(Zs_te.numpy(), Ze_te.numpy(), event_type, h_threshold)

split_idx = dp._sequence_split_idx()
shift = dp.window_shift
dates_tr = [dp.df_z.index[int(p) * shift + dp.seq_len - 1] for p in vidx_tr.numpy()]
dates_te = [dp.df_z.index[(split_idx + int(p)) * shift + dp.seq_len - 1] for p in vidx_te.numpy()]

save_dir = os.path.join(_dir, "diagnosis_plots")
os.makedirs(save_dir, exist_ok=True)

# ── event detection scatter (train vs test), diagnosis.py style ──────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

metric_label = {"abs_change": "|ΔZ|", "absval": "|Z_end|",
                "upper_change": "ΔZ", "lower_change": "ΔZ"}[event_type]

for ax, dates, changes, event_mask, title, n_total_win in [
    (ax1, dates_tr, metric_tr, mask_tr, "Train", len(dp.X_train)),
    (ax2, dates_te, metric_te, mask_te, "Test",  len(dp.X_test)),
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
    ax.set_title(f"{metric_label} — {title} windows (latent state condition)")
    ax.annotate(
        f"Valid: {n_v} / {n_total_win}  ({100*n_v/max(n_total_win,1):.1f}%)\n"
        f"Events: {n_ev} / {n_v}  ({100*n_ev/max(n_v,1):.1f}%)",
        xy=(0.5, -0.12), xycoords='axes fraction', ha='center'
    )
    print(f"{title}: valid {n_v}/{n_total_win}, events {n_ev}/{n_v} ({100*n_ev/max(n_v,1):.1f}%)")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "event_detection.png"), dpi=150, bbox_inches="tight")

# ── correlation matrices — last-day returns, conditional vs unconditional ────
last_tr = dp.X_train[:, -1, :].numpy()
last_te = dp.X_test[:, -1, :].numpy()

cond_tr = dp.X_train[vidx_tr][mask_tr][:, -1, :].numpy()
cond_te = dp.X_test[vidx_te][mask_te][:, -1, :].numpy()

fig_corr, axes_corr = plt.subplots(2, 2, figsize=(12, 10))
tick_lbl = [t.upper() for t in tickers]

panels = [
    (axes_corr[0, 0], last_tr, "Train — unconditional"),
    (axes_corr[0, 1], cond_tr, "Train — conditional (events)"),
    (axes_corr[1, 0], last_te, "Test — unconditional"),
    (axes_corr[1, 1], cond_te, "Test — conditional (events)"),
]

for ax, data, title in panels:
    corr = np.corrcoef(data.T)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(tickers))); ax.set_xticklabels(tick_lbl)
    ax.set_yticks(range(len(tickers))); ax.set_yticklabels(tick_lbl)
    ax.set_title(f"{title} (n={data.shape[0]})", fontsize=10, fontweight="bold")
    for r in range(len(tickers)):
        for c in range(len(tickers)):
            ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center",
                    fontweight="bold", color="white" if abs(corr[r, c]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlation_matrices.png"), dpi=150, bbox_inches="tight")
