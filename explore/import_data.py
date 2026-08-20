import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_default_config
from data import DataProcessor
from latent_state_estimation.macro_main import LatentStateEstimator

_cfg = get_default_config()

# Conditioning column, written as column 0 of the CSV. Named for what it IS
# (the conditioning series) rather than how it was produced — it comes from
# either state_space or tracking_regression. DataProcessor picks up the first
# column positionally, so the name only has to avoid clashing with a ticker.
cond_event = "m_t"

# All macro inputs (growth + inflation panels, monthly and daily) are imported
# separately by latent_state_estimation/macro_importer.py into the *_macro.csv /
# *_daily.csv files the estimator reads — nothing is fetched here.

# Which variables out of each group's set feed the monthly PCA: a list = just
# those columns, None or [] = group dropped (same rule as LatentStateEstimator).
bucket = {group: list(sel)
          for group, sel in (("growth", _cfg.data.growth_vars),
                             ("inflation", _cfg.data.inflation_vars))
          if sel}

if not bucket:
    raise ValueError("growth_vars and inflation_vars are both empty — at least one "
                     "group must name the columns used for the conditioning series.")

# e.g. "inflation: cpi" — used to label the figures with what actually produced
# the conditioning series
bucket_lbl = ";  ".join(f"{g} using {', '.join(cols)}" for g, cols in bucket.items())

print(f"[1/4] estimating macro state (method={_cfg.data.latent_method!r})...")
print(f"      bucket: {' + '.join(bucket)}")
for group, cols in bucket.items():
    print(f"        {group:<10} {', '.join(cols)}")

cond_series = LatentStateEstimator(
    method=_cfg.data.latent_method,
    growth_vars=_cfg.data.growth_vars,
    inflation_vars=_cfg.data.inflation_vars,
).fit()
print("[1/4] done.")

tickers = _cfg.data.tickers   # asset tickers only; conditioning series is separate
csv_path = _cfg.data.csv_path

print(f"[2/4] downloading price history for {tickers} from yfinance...")
df = yf.download(tickers, start = _cfg.data.start_date, auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()
print(f"[2/4] done ({len(df)} raw rows).")

print("[3/4] merging conditioning series with stock log-returns...")
df[cond_event] = cond_series.reindex(df.index)

df_out = pd.DataFrame({cond_event: df[cond_event]})
for t in tickers:
    df_out[t] = log_ret[t]

df_out = df_out.dropna(subset=tickers)
print("[3/4] done.")

print(f"[4/4] writing {csv_path}...")
df_out.to_csv(csv_path, index_label="Date")
print("[4/4] done.")

print(f"total rows: {len(df_out)}")

# ── Covariance/correlation check: real data vs. conditioning-event bucket —
# lets us see whether the selected event bucket actually shifts cross-asset
# structure before spending time on a diffusion run.
#
# This runs the REAL DataProcessor over the CSV just written, so the windows,
# EMA standardization, train/test split and windowed Z_start/Z_end event mask
# are identical to what diffusion_model_analysis/cov.py uses — the panels here
# are directly comparable with that script's "Real" panels, and they respond to
# event_causal / event_lag_gap / seq_len the same way.
dp = DataProcessor(
    csv_path        = csv_path,
    tickers         = _cfg.data.tickers,
    weekday_col     = _cfg.data.weekday_col,
    seq_len         = _cfg.data.seq_len,
    test_days       = _cfg.data.test_days,
    start_date      = _cfg.data.start_date,
    end_date        = _cfg.data.end_date,
    train_end_date  = _cfg.data.train_end_date,
    window_shift    = _cfg.data.window_shift,
    winsorize_lower = _cfg.data.winsorize_lower,
    winsorize_upper = _cfg.data.winsorize_upper,
    ema_span        = _cfg.data.ema_span,
    event_causal    = _cfg.data.event_causal,
    event_lag_gap   = _cfg.data.event_lag_gap,
)
dp.process_all()

event_type  = _cfg.hfunction.event_type
h_threshold = dp.get_event_threshold_from_percentile(
    _cfg.hfunction.event_threshold, event_type)
print(f"Event threshold: top {_cfg.hfunction.event_threshold:.1%} -> "
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
    elif event_type == "start_upper":
        return Z_start >= h_threshold
    raise NotImplementedError(f"event_type={event_type!r}")


X_train, X_test = dp.X_train, dp.X_test
Zs_tr, Ze_tr, vidx_tr = dp.get_z_windows_train_aligned()
Zs_te, Ze_te, vidx_te = dp.get_z_windows_test()
X_train_events = X_train[vidx_tr][event_mask(Zs_tr, Ze_tr)]
X_test_events  = X_test[vidx_te][event_mask(Zs_te, Ze_te)]

# Last-day returns of each window, shape (N, A) — same slice cov.py takes
panels = [
    ("Real Train (all)",           X_train[:, -1, :].numpy()),
    ("Real Train (event windows)", X_train_events[:, -1, :].numpy()),
    ("Real Test (all)",            X_test[:, -1, :].numpy()),
    ("Real Test (event windows)",  X_test_events[:, -1, :].numpy()),
]

print(f"\n{'='*60}")
print(f"Bucket check: {cond_event} {event_type}, top {_cfg.hfunction.event_threshold:.1%}")
print(f"  train events: {len(X_train_events)} / {len(X_train)}")
print(f"  test  events: {len(X_test_events)} / {len(X_test)}")

print(f"\n── Correlation Matrices ──")
for lbl, arr in panels:
    C = np.corrcoef(arr.T)
    print(f"\n{lbl}  (n={len(arr)}):")
    print(pd.DataFrame(C, index=tickers, columns=tickers).round(3).to_string())

# Heatmap image of the correlation matrices above, styled like
# diffusion_model_analysis/cov.py's plot_matrices.
tick_lbl  = [t.upper() for t in tickers]
n_assets  = len(tickers)
font_size = max(8, min(13, 40 // n_assets))

n_cols = 2
n_rows = (len(panels) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows))
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(panels):]:
    ax.axis("off")
for ax, (lbl, arr) in zip(axes, panels):
    C  = np.corrcoef(arr.T)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n_assets)); ax.set_xticklabels(tick_lbl, fontsize=10)
    ax.set_yticks(range(n_assets)); ax.set_yticklabels(tick_lbl, fontsize=10)
    ax.set_title(f"{lbl}\n(n={len(arr)})", fontsize=10, fontweight="bold", pad=8)
    for r in range(n_assets):
        for c in range(n_assets):
            v = C[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=font_size,
                    fontweight="bold", color="white" if abs(v) > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    f"Correlation Matrices — Last-Day Returns\n"
    f"(event: {cond_event} {event_type} ≥ {h_threshold} std "
    f"[top {_cfg.hfunction.event_threshold:.1%}], causal={_cfg.data.event_causal})\n"
    f"method: {_cfg.data.latent_method},  conditioning bucket: {bucket_lbl}",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()

corr_plot_path = os.path.join(os.path.dirname(csv_path), "bucket_corr_matrices.png")
fig.savefig(corr_plot_path, dpi=150, bbox_inches="tight")
print(f"Saved {corr_plot_path}")


# Quick visual sanity check (stationarity/drift) of the conditioning series just
# written into column 0 — labelled with the exact bucket that produced it, since
# the series is only estimated from the growth/inflation variables selected in
# the config.
#
# The EWMA z-score, z = (x - EWMA_mean)/EWMA_vol, is layered on top of the raw
# series (own right-hand axis, since the two live on very different scales),
# computed the same way DataProcessor._compute_ema_stats() does it — same
# ema_span, min_periods=20, and .shift(1) so each day's stats use data through
# the day before only (the EWMA mean/vol are inputs to z, not plotted). If the
# raw series wanders but z mean-reverts around 0 with a stable spread, the
# downstream standardization is what makes it usable.
_span     = _cfg.data.ema_span
cond_raw  = df_out[cond_event]
cond_mu   = cond_raw.ewm(span=_span, min_periods=20).mean().shift(1)
cond_sig  = cond_raw.ewm(span=_span, min_periods=20).std().shift(1)
cond_z    = (cond_raw - cond_mu) / cond_sig

plot_path = os.path.join(os.path.dirname(csv_path), "conditioning_series.png")
fig, ax = plt.subplots(figsize=(14, 5))
l_raw, = ax.plot(cond_raw.index, cond_raw, linewidth=0.7, color="steelblue", label="raw")
ax.set_xlabel("Date")
ax.set_ylabel(cond_event)
ax.grid(True, alpha=0.3)

ax_z = ax.twinx()
l_z, = ax_z.plot(cond_z.index, cond_z, linewidth=0.7, color="seagreen", alpha=0.85,
                 label=f"EWMA z-score (span={_span}, causal)")
ax_z.axhline(0, color="seagreen", linewidth=0.8, alpha=0.5)
ax_z.set_ylabel("EWMA z-score", color="seagreen")
ax_z.tick_params(axis="y", labelcolor="seagreen")

ax.legend(handles=[l_raw, l_z], fontsize=8, loc="upper left")
ax.set_title(f"Conditioning Series ({cond_event}, method={_cfg.data.latent_method!r})\n"
             f"bucket — {bucket_lbl}", fontsize=11)
fig.tight_layout()
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved conditioning series plot to {plot_path}")

_z = cond_z.dropna()
print(f"EWMA z (span={_span}): mean={_z.mean():.3f}  std={_z.std():.3f}  "
      f"min={_z.min():.2f}  max={_z.max():.2f}  |z|>2: {(_z.abs() > 2).mean():.1%}")
