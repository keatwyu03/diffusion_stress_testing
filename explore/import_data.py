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

print("[1/5] fetching FRED macro series (T10YFF, t1yffm, vix, sp500)...")
fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')

data = {
    'T10YFF': fred.get_series('T10YFF'),
    't1yffm': fred.get_series('t1yffm'),
    'vix': fred.get_series('VIXCLS'),
    'sp500': fred.get_series('SP500'),
}
print("[1/5] done.")

if _cfg.data.latent_method is not None:
    # conditioning variable = estimated daily latent macro state
    print(f"[2/5] estimating latent macro state (method={_cfg.data.latent_method!r})...")
    cond_series = LatentStateEstimator(method=_cfg.data.latent_method).fit()
    print("[2/5] done.")
else:
    print("[2/5] latent_method=None, using raw FRED series — skipping estimation.")
    cond_series = data[cond_event]

tickers = _cfg.data.tickers[1:]  # everything after the macro variable
csv_path = _cfg.data.csv_path

print(f"[3/5] downloading price history for {tickers} from yfinance...")
df = yf.download(tickers, start = _cfg.data.start_date, auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()
print(f"[3/5] done ({len(df)} raw rows).")

print("[4/5] merging conditioning series with stock log-returns...")
df[cond_event] = cond_series.reindex(df.index)

df_out = pd.DataFrame({cond_event: df[cond_event]})
for t in tickers:
    df_out[t] = log_ret[t]

df_out = df_out.dropna(subset=tickers)
print("[4/5] done.")

print(f"[5/5] writing {csv_path}...")
df_out.to_csv(csv_path, index_label="Date")
print("[5/5] done.")

print(f"total rows: {len(df_out)}")
