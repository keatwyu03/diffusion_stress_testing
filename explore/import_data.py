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
    'gs5': fred.get_series('GS5'),
    'vix': fred.get_series('VIXCLS'),
    'spread': fred.get_series('T10Y2Y'),
    'sp500': fred.get_series('SP500'),
}

cond_series = data[cond_event]  

tickers = ["AAPL", "ORCL", "MSFT", "IBM"]
df = yf.download(tickers, start = "2008-01-01", auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()

df[cond_event] = cond_series.reindex(df.index).interpolate(method='time')

df_out = pd.DataFrame({
    cond_event:            df[cond_event],
    "AAPL":                log_ret["AAPL"],
    "ORCL":                log_ret["ORCL"],
    "MSFT":                log_ret["MSFT"],
    "IBM":                 log_ret["IBM"],
})

df_out = df_out.dropna()
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

seq_len    = _cfg.data.seq_len
event_win  = _cfg.hfunction.event_window
test_days  = _cfg.data.test_days

vals = df_out[cond_event].values
n_total  = len(vals)
n_train  = n_total - test_days

# standardize using train set mean/std only
train_mean = vals[:n_train].mean()
train_std  = vals[:n_train].std()
vals_std   = (vals - train_mean) / train_std

split_date = df_out.index[n_train]
print(f"Split date: {split_date.date()}  (train={n_train}, test={test_days})")
print(f"Train unemp level: mean={train_mean:.4f}  std={train_std:.4f}")

# count events: abs(std_level[end] - std_level[start]) >= threshold over event_window
change_vals = []
time = []

def change_events(vs):
    for t in range(event_win, len(vs)):
        change_vals.append(abs(vs[t] - vs[t - event_win]))
        time.append(df_out.index[t])

change_events(vals_std)


def count_events(vs):
    count = 0
    for t in range(event_win, len(vs)):
        if abs(vs[t] - vs[t - event_win]) >= h_threshold:
            count += 1
    return count

train_events = count_events(vals_std[:n_train])
test_events  = count_events(vals_std[n_train:])

print(f"TRAIN events: {train_events} / {n_train - event_win}  ({100*train_events/(n_train - event_win):.1f}%)")
print(f"TEST  events: {test_events} / {max(test_days - event_win, 1)}  ({100*test_events/max(test_days - event_win, 1):.1f}%)")


fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(df[cond_event].index, df[cond_event].values)
ax2.plot(time, change_vals)
ax2.axhline(h_threshold, color = "red", linestyle = "--")
ax2.annotate(
    f"TRAIN events: {train_events} / {n_train - event_win}  ({100*train_events/(n_train - event_win):.1f}%)\n"
    f"TEST  events: {test_events} / {max(test_days - event_win, 1)}  ({100*test_events/max(test_days - event_win, 1):.1f}%)",
    xy=(0.5, -0.12), xycoords='axes fraction', ha='center')

ax1.set_title(f"Condition Event Series of {cond_event}")
ax2.set_title(f"Daily Change of conditioned event series of {cond_event}")

plt.show()