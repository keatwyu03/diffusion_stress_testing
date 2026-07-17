from fredapi import Fred
import pandas as pd
import numpy as np
import yfinance as yf
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_default_config
from latent_state_estimation.macro_main import LatentStateEstimator

_cfg        = get_default_config()
cond_event  = _cfg.data.tickers[_cfg.hfunction.event_asset_idx]  # e.g. "unemp"

fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')

data = {
    'T10YFF': fred.get_series('T10YFF'),
    't1yffm': fred.get_series('t1yffm'),
    'vix': fred.get_series('VIXCLS'),
    'sp500': fred.get_series('SP500'),
}

if _cfg.data.latent_method is not None:
    # conditioning variable = estimated daily latent macro state
    cond_series = LatentStateEstimator(method=_cfg.data.latent_method).fit()
else:
    cond_series = data[cond_event]

tickers = _cfg.data.tickers[1:]  # everything after the macro variable
csv_path = _cfg.data.csv_path

df = yf.download(tickers, start = _cfg.data.start_date, auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()

df[cond_event] = cond_series.reindex(df.index)

df_out = pd.DataFrame({cond_event: df[cond_event]})
for t in tickers:
    df_out[t] = log_ret[t]

df_out = df_out.dropna(subset=tickers)
df_out.to_csv(csv_path, index_label="Date")


df_ct = yf.download(tickers, start = _cfg.data.ct_start_date, end = _cfg.data.ct_end_date ,auto_adjust=True)["Close"]
log_ret_ct = np.log(df_ct / df_ct.shift(1)).dropna()

df_ct[cond_event] = cond_series.reindex(df_ct.index).interpolate(method='time')

df_out_ct = pd.DataFrame({cond_event: df_ct[cond_event]})
for t in tickers:
    df_out_ct[t] = log_ret_ct[t]

df_out_ct = df_out_ct.dropna()
df_out_ct.to_csv("explore/cross_test_data.csv", index_label="Date")

print(f"total rows: {len(df_out)}")
