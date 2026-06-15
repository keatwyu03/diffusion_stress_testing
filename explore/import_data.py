from fredapi import Fred
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_default_config

_cfg        = get_default_config()
cond_event  = _cfg.data.tickers[_cfg.hfunction.event_asset_idx]  # e.g. "unemp"
h_threshold = _cfg.hfunction.event_threshold

fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')
sp500 = yf.download('^GSPC', start = '1950-01-01')['Close'].squeeze()


data = {
    'unemp': fred.get_series('UNRATE'),
    'cpi': fred.get_series('CPIAUCSL'),
    'gdp': fred.get_series('GDP'),
    'spread': fred.get_series('T10Y2Y'),
    'sp500': fred.get_series('SP500'),
    'vix': fred.get_series('VIXCLS'),
    'baa': fred.get_series('BAA'),
    'aaa': fred.get_series('AAA'),
}


baa_log = np.log(data['baa'] / data['baa'].shift(1))

cond_series = data[cond_event]  # unemployment level

tickers = ["AAPL", "ORCL", "MSFT", "IBM"]
df = yf.download(tickers, start = "2010-01-01", auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()

df[cond_event] = cond_series.reindex(df.index).interpolate(method='time')
df['baa']      = baa_log.reindex(df.index).interpolate(method='time')

df_out = pd.DataFrame({
    cond_event:            df[cond_event],
    "baa":                 df["baa"],
    "AAPL":                log_ret["AAPL"],
    "ORCL":                log_ret["ORCL"],
    "MSFT":                log_ret["MSFT"],
    "IBM":                 log_ret["IBM"],
})

df_out = df_out.dropna()
df_out.to_csv("explore/macro_data_new.csv", index_label="Date")

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