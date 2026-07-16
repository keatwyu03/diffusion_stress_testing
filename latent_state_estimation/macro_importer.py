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
fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')

growth_macro_data = {
    'indpro': fred.get_series('INDPRO'),                       # earliest: 1919-01-01
    'payems': fred.get_series('PAYEMS'),                       # earliest: 1939-01-01
    'pi_transfer': fred.get_series('w875rx1'),                 # earliest: 1959-01-01
    'real_manf_trade': fred.get_series('CMRMTSPL'),            # earliest: 1967-01-01
    'personal_consump' : fred.get_series('DPCERA3M086SBEA'),   # earliest: 1959-01-01
    'capacity_util' : fred.get_series("CUMFNS"),               # earliest: 1948-01-01
}

inflation_macro_data = {
    'cpi': fred.get_series('CPIAUCSL'),                # earliest: 1947-01-01
    'price_index': fred.get_series('PCEPI'),           # earliest: 1959-01-01
    'oil_price': fred.get_series('MCOILWTICO'),        # earliest: 1986-01-01
    'ppi': fred.get_series('PPIACO'),                  # earliest: 1913-01-01
    'hour_earnings' : fred.get_series('AHETPI'),       # earliest: 1964-01-01

}


def _get_futures_close(ticker, period="max", retries=3):
    """Daily close price series for a single futures ticker via yfinance."""
    for attempt in range(retries):
        data = yf.Ticker(ticker).history(period=period)['Close']
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

df_growth_macro_data = pd.DataFrame(growth_macro_data)
df_inf_macro_data = pd.DataFrame(inflation_macro_data)

_capacity_util = df_growth_macro_data["capacity_util"].diff()
df_growth_macro_data = np.log(df_growth_macro_data.drop(columns="capacity_util")).diff()
df_growth_macro_data["capacity_util"] = _capacity_util

df_inf_macro_data = np.log(df_inf_macro_data).diff()

growth_daily_cols = {}
inf_daily_cols = {}

growth_daily_cols["copper"] = np.log(growth_daily_futures["copper"]).diff()
growth_daily_cols["energy"] = np.log(growth_daily_futures["energy"]).diff()

ex_ir = 100 - growth_daily_futures["fed_funds"]
growth_daily_cols["fed_funds"] = ex_ir.diff()

def _par_bond_duration(yield_series, n_years):
    y = yield_series / 100
    return (1 - (1 + y/2)**(-2*n_years)) / y


for item, yr in [("yield_2yr", 2), ("yield_5yr", 5), ("yield_10yr", 10)]:
    duration = _par_bond_duration(fred.get_series(f"DGS{yr}"), yr)
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

df_growth_daily_data = pd.DataFrame(growth_daily_cols)
df_inf_daily_data = pd.DataFrame(inf_daily_cols)

df_growth_daily_data = df_growth_daily_data.loc[df_growth_daily_data.apply(pd.Series.first_valid_index).max():].dropna()
df_inf_daily_data = df_inf_daily_data.loc[df_inf_daily_data.apply(pd.Series.first_valid_index).max():].dropna()

_out_dir = os.path.dirname(os.path.abspath(__file__))
df_growth_daily_data.to_csv(os.path.join(_out_dir, "growth_daily.csv"))
df_inf_daily_data.to_csv(os.path.join(_out_dir, "inflation_daily.csv"))
df_growth_macro_data.to_csv(os.path.join(_out_dir, "growth_macro.csv"))
df_inf_macro_data.to_csv(os.path.join(_out_dir, "inflation_macro.csv"))



