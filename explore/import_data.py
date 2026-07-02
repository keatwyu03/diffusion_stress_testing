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
sp500 = yf.download('^GSPC', start = '2008-01-01')['Close'].squeeze()


data = {
    't1yffm': fred.get_series('t1yffm'),
    'vix': fred.get_series('VIXCLS'),
    'spread': fred.get_series('T10Y2Y'),
    'sp500': fred.get_series('SP500'),
}

cond_series = data[cond_event]  

tickers = ["AAPL", "ORCL", "MSFT", "IBM"]
df = yf.download(tickers, start = "2008-01-01", auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()

df[cond_event] = cond_series.reindex(df.index)

df_out = pd.DataFrame({
    cond_event:            df[cond_event],
    "AAPL":                log_ret["AAPL"],
    "ORCL":                log_ret["ORCL"],
    "MSFT":                log_ret["MSFT"],
    "IBM":                 log_ret["IBM"],
})

df_out = df_out.dropna(subset=["AAPL", "ORCL", "MSFT", "IBM"])
df_out.to_csv("explore/macro_data_new.csv", index_label="Date")


df_ct = yf.download(tickers, start = _cfg.data.ct_start_date, end = _cfg.data.ct_end_date ,auto_adjust=True)["Close"]
log_ret_ct = np.log(df_ct / df_ct.shift(1)).dropna()

df_ct[cond_event] = cond_series.reindex(df_ct.index).interpolate(method='time')

df_out_ct = pd.DataFrame({
    cond_event:            df_ct[cond_event],
    "AAPL":                log_ret_ct["AAPL"],
    "ORCL":                log_ret_ct["ORCL"],
    "MSFT":                log_ret_ct["MSFT"],
    "IBM":                 log_ret_ct["IBM"],
})

df_out_ct = df_out_ct.dropna()
df_out_ct.to_csv("explore/cross_test_data.csv", index_label="Date")

print(f"total rows: {len(df_out)}")

from data.data_processor import DataProcessor

dp = DataProcessor(
    csv_path        = "explore/macro_data_new.csv",
    tickers         = _cfg.data.tickers,
    seq_len         = _cfg.data.seq_len,
    test_days       = _cfg.data.test_days,
    train_end_date  = _cfg.data.train_end_date,
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

n_train_windows = len(dp.X_train)
n_valid         = len(valid_idx)
abs_change      = (Z_end - Z_start).abs().numpy()
n_events        = int((abs_change >= h_threshold).sum())

print(f"Training windows:              {n_train_windows}")
print(f"Valid macro windows:           {n_valid}  ({100*n_valid/n_train_windows:.1f}%)")
print(f"Events (|ΔZ| >= {h_threshold}): {n_events} / {n_valid}  ({100*n_events/n_valid:.1f}%)")

window_end_dates = [dp.df.index[int(i) + dp.seq_len - 1] for i in valid_idx.numpy()]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(dp.df[cond_event].dropna().index, dp.df[cond_event].dropna().values)
ax1.set_title(f"Condition Event Series: {cond_event}")

event_mask = abs_change >= h_threshold

ax2.scatter(window_end_dates, abs_change, s=10, color="steelblue", label="valid window")
ax2.scatter(
    [d for d, e in zip(window_end_dates, event_mask) if e],
    abs_change[event_mask],
    s=15, color="red", label=f"event (|ΔZ| ≥ {h_threshold})"
)
ax2.axhline(h_threshold, color="red", linestyle="--")
ax2.legend()
ax2.annotate(
    f"Valid windows: {n_valid} / {n_train_windows}  ({100*n_valid/n_train_windows:.1f}%)\n"
    f"Events: {n_events} / {n_valid}  ({100*n_events/n_valid:.1f}%)",
    xy=(0.5, -0.12), xycoords='axes fraction', ha='center')
ax2.set_title(f"|ΔZ| per valid window (window end date)")

plt.show()