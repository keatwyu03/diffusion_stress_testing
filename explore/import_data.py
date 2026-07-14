from fredapi import Fred
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import os
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_default_config

_cfg        = get_default_config()
cond_event  = _cfg.data.tickers[_cfg.hfunction.event_asset_idx]  # e.g. "unemp"
h_threshold = _cfg.hfunction.event_threshold

fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')

data = {
    'T10YFF': fred.get_series('T10YFF'),
    't1yffm': fred.get_series('t1yffm'),
    'vix': fred.get_series('VIXCLS'),
    'sp500': fred.get_series('SP500'),
}

cond_series = data[cond_event]

tickers = _cfg.data.tickers[1:]  # everything after the macro variable
df = yf.download(tickers, start = _cfg.data.start_date, auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()

df[cond_event] = cond_series.reindex(df.index)

df_out = pd.DataFrame({cond_event: df[cond_event]})
for t in tickers:
    df_out[t] = log_ret[t]

df_out = df_out.dropna(subset=tickers)
df_out.to_csv("explore/macro_data_new.csv", index_label="Date")


df_ct = yf.download(tickers, start = _cfg.data.ct_start_date, end = _cfg.data.ct_end_date ,auto_adjust=True)["Close"]
log_ret_ct = np.log(df_ct / df_ct.shift(1)).dropna()

df_ct[cond_event] = cond_series.reindex(df_ct.index).interpolate(method='time')

df_out_ct = pd.DataFrame({cond_event: df_ct[cond_event]})
for t in tickers:
    df_out_ct[t] = log_ret_ct[t]

df_out_ct = df_out_ct.dropna()
df_out_ct.to_csv("explore/cross_test_data.csv", index_label="Date")

print(f"total rows: {len(df_out)}")

from data.data_processor import DataProcessor

dp = DataProcessor(
    csv_path        = "explore/macro_data_new.csv",
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
dp.r_dw = dp.df[_cfg.data.tickers[1:]]  # standardize/sequence on stock cols only
dp.standardize()
dp.winsorize()
dp.make_sequences()
dp.train_test_split()

Z_start, Z_end, valid_idx = dp.get_z_windows()

# event_threshold is specified as "top X% of |Z_end - Z_start|" (e.g. 0.10 = top 10%),
# converted here to the equivalent raw numeric cutoff — see main.py for details.
event_top_fraction = h_threshold
h_threshold = dp.get_event_threshold_from_percentile(event_top_fraction, _cfg.hfunction.event_type)
print(f"Event threshold: top {event_top_fraction:.1%} -> {h_threshold:.4f} std")

n_train_windows = len(dp.X_train)
n_valid         = len(valid_idx)
abs_change      = (Z_end - Z_start).abs().numpy()
n_events        = int((abs_change >= h_threshold).sum())

print(f"Training windows:              {n_train_windows}")
print(f"Valid macro windows:           {n_valid}  ({100*n_valid/n_train_windows:.1f}%)")
print(f"Events (|ΔZ| >= {h_threshold}): {n_events} / {n_valid}  ({100*n_events/n_valid:.1f}%)")

# Test window event count
cfg = _cfg
w = cfg.data.macro_window_tolerance
macro_col = cfg.data.tickers[0]
macro_raw = dp.df[macro_col]
z_mean = macro_raw.iloc[:-cfg.data.test_days].dropna().mean()
z_std  = macro_raw.iloc[:-cfg.data.test_days].dropna().std()
macro_std_vals = ((macro_raw - z_mean) / z_std).values
n_total = len(macro_std_vals)
n_train = n_total - cfg.data.test_days

test_abs_changes, test_end_dates = [], []
test_events, test_valid = 0, 0
for i in range(n_train, n_total - dp.seq_len + 1):
    start_slice = macro_std_vals[i : i + w + 1]
    start_vals  = start_slice[~np.isnan(start_slice)]
    end_idx     = i + dp.seq_len - 1
    end_slice   = macro_std_vals[max(0, end_idx - w) : end_idx + 1]
    end_vals    = end_slice[~np.isnan(end_slice)]
    if len(start_vals) == 0 or len(end_vals) == 0:
        continue
    chg = abs(float(end_vals[-1]) - float(start_vals[0]))
    test_abs_changes.append(chg)
    test_end_dates.append(dp.df.index[end_idx])
    test_valid += 1
    if chg >= h_threshold:
        test_events += 1

n_test_windows = len(dp.X_test)
print(f"\nTest windows:                  {n_test_windows}")
print(f"Valid macro windows:           {test_valid}  ({100*test_valid/max(n_test_windows,1):.1f}%)")
print(f"Events (|ΔZ| >= {h_threshold}): {test_events} / {max(test_valid,1)}  ({100*test_events/max(test_valid,1):.1f}%)")

test_abs_changes = np.array(test_abs_changes)
test_event_mask  = test_abs_changes >= h_threshold
train_end_dates  = [dp.df.index[int(i) + dp.seq_len - 1] for i in valid_idx.numpy()]
train_event_mask = abs_change >= h_threshold

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, dates, changes, event_mask, title, n_total_win in [
    (ax1, train_end_dates, abs_change,       train_event_mask, "Train", n_train_windows),
    (ax2, test_end_dates,  test_abs_changes,  test_event_mask,  "Test",  n_test_windows),
]:
    n_ev = int(event_mask.sum())
    n_v  = len(changes)
    ax.scatter(dates, changes, s=10, color="steelblue", label="valid window")
    ax.scatter(
        [d for d, e in zip(dates, event_mask) if e],
        changes[event_mask],
        s=15, color="red", label=f"event (|ΔZ| ≥ {h_threshold})"
    )
    ax.axhline(h_threshold, color="red", linestyle="--")
    ax.legend()
    ax.set_title(f"|ΔZ| — {title} windows")
    ax.annotate(
        f"Valid: {n_v} / {n_total_win}  ({100*n_v/max(n_total_win,1):.1f}%)\n"
        f"Events: {n_ev} / {n_v}  ({100*n_ev/max(n_v,1):.1f}%)",
        xy=(0.5, -0.12), xycoords='axes fraction', ha='center'
    )

plt.tight_layout()
plt.show()