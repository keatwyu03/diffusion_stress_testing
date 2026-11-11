from fredapi import Fred
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_default_config

_cfg        = get_default_config()
# DOWNLOADING IS DISABLED. The macro panels are locked to the Table 4 snapshot
# (latent_state_estimation/frozen_table4_a6c4f3f/, git a6c4f3f); each conditioning set is built ONCE from it by
# explore/build_conditioning_sets.py. Nothing below this line runs -- no FRED / Yahoo Finance request is ever made.
raise SystemExit("latent_state_estimation/macro_importer.py: downloading is disabled -- the macro data is locked to "
                 "latent_state_estimation/frozen_table4_a6c4f3f/ (see explore/build_conditioning_sets.py). "
                 "Nothing was downloaded or written.")
fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')

# Data window = config.data.start_date .. config.data.end_date. This importer has NO date range of its own:
# every FRED series and futures contract is requested for the config window only, starting one month before
# start_date (the first daily log-diff needs a prior price), and everything is cut to the window before writing.
_start = pd.Timestamp(_cfg.data.start_date) if _cfg.data.start_date else None
_end = pd.Timestamp(_cfg.data.end_date) if _cfg.data.end_date else None
_fetch_start = (_start.to_period("M") - 1).to_timestamp() if _start is not None else None


def _fred(series_id):
    return fred.get_series(series_id, observation_start=_fetch_start, observation_end=_end)


growth_macro_data = {
    'indpro': _fred('INDPRO'),                       # earliest: 1919-01-01
    'payems': _fred('PAYEMS'),                       # earliest: 1939-01-01
    'pi_transfer': _fred('w875rx1'),                 # earliest: 1959-01-01
    'real_manf_trade': _fred('CMRMTSPL'),            # earliest: 1967-01-01
    'personal_consump' : _fred('DPCERA3M086SBEA'),   # earliest: 1959-01-01
    'capacity_util' : _fred("CUMFNS"),               # earliest: 1948-01-01
}

inflation_macro_data = {
    'cpi': _fred('CPIAUCSL'),                # earliest: 1947-01-01
    'oil_price': _fred('MCOILWTICO'),        # earliest: 1986-01-01
    'ppi': _fred('PPIACO'),                  # earliest: 1913-01-01
    'hour_earnings' : _fred('AHETPI'),       # earliest: 1964-01-01

}

vol_macro_data = {
    'epu': _fred('USEPUINDXM'),              # Economic Policy Uncertainty Index, earliest: 1985-01-01
    'umcsent': _fred('UMCSENT'),             # U. Michigan Consumer Sentiment Index, earliest: 1952-11-01
}


def _get_futures_close(ticker, retries=3):
    """Daily close price series for a single futures ticker via yfinance, over the config data window
    (from _fetch_start, one month before start_date, through end_date inclusive; yfinance's end is exclusive)."""
    kw = {"start": _fetch_start.strftime("%Y-%m-%d")} if _fetch_start is not None else {"period": "max"}
    if _end is not None:
        kw["end"] = (_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(retries):
        data = yf.Ticker(ticker).history(**kw)['Close']
        if not data.empty:
            return data.tz_localize(None)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no data for {ticker} after {retries} attempts")


growth_daily_futures = {
    'copper': _get_futures_close('HG=F'),          # COMEX copper futures, earliest: 2000-08-30
    'energy': _get_futures_close('CL=F'),          # WTI crude oil futures, earliest: 2000-08-23
    'fed_funds': _get_futures_close('ZQ=F'),       # 30-Day Fed Funds futures, earliest: 2000-09-01
    'yield_2yr': _get_futures_close('ZT=F'),       # 2-year Treasury note futures, earliest: 2000-06-02
    'yield_5yr': _get_futures_close('ZF=F'),       # 5-year Treasury note futures, earliest: 2000-09-21
    'yield_10yr': _get_futures_close('ZN=F'),      # 10-year Treasury note futures, earliest: 2000-09-21
}


inflation_daily_futures = {
    'wti_crude': _get_futures_close('CL=F'),       # WTI crude oil futures, earliest: 2000-08-23
    'natural_gas': _get_futures_close('NG=F'),     # Henry Hub natural gas futures, earliest: 2000-08-30
    'corn': _get_futures_close('ZC=F'),            # corn futures, earliest: 2000-07-17
    'wheat': _get_futures_close('ZW=F'),           # wheat futures, earliest: 2000-07-17
    'gold': _get_futures_close('GC=F'),            # gold futures, earliest: 2000-08-30
}


vol_daily_series = {
    'vix': _fred('VIXCLS'),                    # CBOE VIX, earliest: 1990-01-02
    'credit_spread': _fred('BAA10Y'),          # Moody's Baa - 10yr Treasury, earliest: 1986-01-02
    'aaa_spread': _fred('AAA10Y'),             # Moody's Aaa - 10yr Treasury, earliest: 1983-01-03
}

# Monthly anchors: log-differenced (the only transform used downstream --
# latent_state_estimation's PCA/tracking-regression/state-space all consume
# these *_macro.csv files directly, so this is the single place the
# level -> monthly-change conversion happens). capacity_util is a
# utilization RATE (already bounded ~0-100), so a plain first difference is
# used for it instead of a log-diff.
DIFF_COLS = ("capacity_util",)


def _monthly_change(df):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in DIFF_COLS:
            out[col] = df[col].diff()
        else:
            out[col] = np.log(df[col]).diff()
    return out


# TEMPORARILY DISABLED for an isolation test (raw levels vs. log-diff, to
# check whether log-diffing explains a drop in state_space's anchor R^2) --
# revert to _monthly_change(...) afterward.
df_growth_macro_data = pd.DataFrame(growth_macro_data)
df_inf_macro_data = pd.DataFrame(inflation_macro_data)
df_vol_macro_data = pd.DataFrame(vol_macro_data)

# Daily proxies: original transforms restored (log-diff for prices, duration-
# adjusted log-diff for yield futures, ex_ir diff for fed_funds, corn/wheat
# blend for grains) — these had deliberate design reasons, unlike the monthly
# anchors above.
growth_daily_cols = {}
inf_daily_cols = {}
vol_daily_cols = {}

growth_daily_cols["copper"] = np.log(growth_daily_futures["copper"]).diff()
growth_daily_cols["energy"] = np.log(growth_daily_futures["energy"]).diff()

ex_ir = 100 - growth_daily_futures["fed_funds"]
growth_daily_cols["fed_funds"] = ex_ir.diff()

def _par_bond_duration(yield_series, n_years):
    y = yield_series / 100
    return (1 - (1 + y/2)**(-2*n_years)) / y


for item, yr in [("yield_2yr", 2), ("yield_5yr", 5), ("yield_10yr", 10)]:
    duration = _par_bond_duration(_fred(f"DGS{yr}"), yr)
    price = growth_daily_futures[item]
    log_diff = -np.log(price).diff() / duration
    growth_daily_cols[item] = log_diff


wti = inflation_daily_futures["wti_crude"].where(
    inflation_daily_futures["wti_crude"] > 0
)

inf_daily_cols["wti_crude"] = np.log(wti).diff()
inf_daily_cols["natural_gas"] = np.log(inflation_daily_futures["natural_gas"]).diff()

corn = np.log(inflation_daily_futures["corn"]).diff()
wheat = np.log(inflation_daily_futures["wheat"]).diff()
inf_daily_cols["grains"] = 0.5 * corn + 0.5 * wheat

inf_daily_cols["gold"] = np.log(inflation_daily_futures["gold"]).diff()

vol_daily_cols["vix"] = np.log(vol_daily_series["vix"]).diff()
vol_daily_cols["credit_spread"] = vol_daily_series["credit_spread"].diff()
vol_daily_cols["aaa_spread"] = vol_daily_series["aaa_spread"].diff()

df_growth_daily_data = pd.DataFrame(growth_daily_cols)
df_inf_daily_data = pd.DataFrame(inf_daily_cols)
df_vol_daily_data = pd.DataFrame(vol_daily_cols)

df_growth_daily_data = df_growth_daily_data.loc[df_growth_daily_data.apply(pd.Series.first_valid_index).max():].dropna()
df_inf_daily_data = df_inf_daily_data.loc[df_inf_daily_data.apply(pd.Series.first_valid_index).max():].dropna()
df_vol_daily_data = df_vol_daily_data.loc[df_vol_daily_data.apply(pd.Series.first_valid_index).max():].dropna()

# Data window (config.data.start_date .. end_date): nothing outside it is written.
# Daily series: start_date .. end_date. Monthly series: start_date's month .. the last month BEFORE end_date's
# month (2026-08-20 -> 2026-07-01, the last month present in the Aug 23 snapshot).
_first_month = _start.to_period("M").to_timestamp() if _start is not None else None
_last_month = (_end.to_period("M") - 1).to_timestamp() if _end is not None else None
df_growth_daily_data = df_growth_daily_data.loc[_start:_end]
df_inf_daily_data = df_inf_daily_data.loc[_start:_end]
df_vol_daily_data = df_vol_daily_data.loc[_start:_end]
df_growth_macro_data = df_growth_macro_data.loc[_first_month:_last_month]
df_inf_macro_data = df_inf_macro_data.loc[_first_month:_last_month]
df_vol_macro_data = df_vol_macro_data.loc[_first_month:_last_month]
print(f"[macro_importer] data window {_cfg.data.start_date}..{_cfg.data.end_date}: daily {_start}..{_end}, "
      f"monthly {_first_month}..{_last_month}")

_out_dir = os.path.dirname(os.path.abspath(__file__))
df_growth_daily_data.to_csv(os.path.join(_out_dir, "growth_daily.csv"))
df_inf_daily_data.to_csv(os.path.join(_out_dir, "inflation_daily.csv"))
df_vol_daily_data.to_csv(os.path.join(_out_dir, "vol_daily.csv"))
df_growth_macro_data.to_csv(os.path.join(_out_dir, "growth_macro.csv"))
df_inf_macro_data.to_csv(os.path.join(_out_dir, "inflation_macro.csv"))
df_vol_macro_data.to_csv(os.path.join(_out_dir, "vol_macro.csv"))



