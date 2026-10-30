"""
Shared trading calendar, window manifest, and train/test date boundary for
Experiment 2 -- imported by every other experiment_2 module so all 5 methods
(S0-S4) and all pipeline stages (state construction, standardization,
stationarity, diffusion) agree on exactly the same dates. This is the
proactive version of the lesson learned the hard way in the Doob-h
generalization (don't let each method silently use its own differently-sized
window pool).

Reuses main_exp1.py's generic, parameter-free building blocks directly
(TICKERS, build_windows, train_test_split_idx) -- does NOT reuse its
model-key-resolution layer (get_condition_array/conditioning_signature/
train_model/sample_model), which the approved plan found unsafe to reuse
as black boxes for S0-S4's own key scheme.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP1 = os.path.join(os.path.dirname(_HERE), "experiment_1")
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _EXP1)
sys.path.insert(0, _ROOT)

import main_exp1 as _m1  # noqa: E402
from config import get_default_config  # noqa: E402

TICKERS = _m1.TICKERS
METHODS = ["s0", "s1", "s2", "s3", "s4"]
METHOD_DIRNAMES = {
    "s0": "S0_state_space",
    "s1": "S1_linear",
    "s2": "S2_forward_fill",
    "s3": "S3_cubic_spline",
    "s4": "S4_tracking_regression",
}
METHOD_LABELS = {
    "s0": "S0 State-Space Model",
    "s1": "S1 Linear Interpolation",
    "s2": "S2 Forward-Fill",
    "s3": "S3 Cubic Spline",
    "s4": "S4 Tracking Regression",
}
# Per approved plan: S1/S3 are inherently retrospective (need a future
# monthly anchor); S0/S2/S4 are causal in filtering but still have
# full-sample-fit preprocessing (PCA/scaler, SSM daily z-scoring, tracking
# regression OLS betas) -- disclosed per-method in daily_macro_states.py's
# saved causality record, not collapsed into this one flag alone.
RETROSPECTIVE = {"s0": False, "s1": True, "s2": False, "s3": True, "s4": False}


def load_asset_panel(cfg=None):
    """Trading-calendar-complete asset return panel (date index + 15 tickers),
    same convention as main_exp1.load_raw_data but without picking a single
    macro column -- Experiment 2 has 5 competing macro columns, attached
    separately by daily_macro_states.py."""
    cfg = cfg or get_default_config()
    csv_path = os.path.join(_ROOT, "explore", "macro_data_new.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    missing = [t for t in TICKERS if t not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required tickers {missing}")
    df = df[TICKERS]
    if cfg.data.start_date:
        df = df[df.index >= pd.to_datetime(cfg.data.start_date)]
    if cfg.data.end_date:
        df = df[df.index <= pd.to_datetime(cfg.data.end_date)]
    df = df.dropna(subset=TICKERS)
    return df


def get_window_manifest(cfg=None):
    """Builds the ONE shared set of seq_len windows + train/test boundary
    every method/stage must agree on. Uses main_exp1.build_windows with
    required_cond_cols=[] (no conditioning-column validity filter here --
    each method's own validity gets intersected in afterward, same pattern
    as the Doob-h common-pool fix) and main_exp1.train_test_split_idx
    unmodified (both confirmed fully generic in the approved plan).

    Returns a dict: asset panel df, seq_len, starts (window start positions),
    window_start_dates, window_end_dates, split_idx, test_start_idx,
    train_end_date (last train window's end date), and
    test_start_date (first test window's start date) -- the single daily
    cutoff every train-only normalization step (state z-scoring, neural
    standardizer fitting) must use.
    """
    cfg = cfg or get_default_config()
    df = load_asset_panel(cfg)
    seq_len = cfg.data.seq_len

    # NOT main_exp1.build_windows: confirmed (smoke-tested) its cond_vals dict
    # build is hardcoded to Experiment 1's 8 specific column names regardless
    # of required_cond_cols, exactly the risk the approved plan flagged --
    # so window construction is inlined here instead (shift-1, length seq_len,
    # no conditioning-column dependency at all). Per-method validity is
    # intersected later, once all 5 states exist (same reasoning as the
    # Doob-h common-pool fix: computing it before all methods exist would
    # bake in an arbitrary ordering).
    dates = df.index
    N = len(df)
    starts = np.arange(0, N - seq_len + 1, dtype=np.int64)
    w_start = dates[starts]
    w_end = dates[starts + seq_len - 1]
    split_idx, test_start_idx = _m1.train_test_split_idx(starts, w_end, seq_len, cfg)

    return {
        "df": df,
        "seq_len": seq_len,
        "starts": starts,
        "window_start_dates": w_start,
        "window_end_dates": w_end,
        "split_idx": split_idx,
        "test_start_idx": test_start_idx,
        "train_end_date": pd.Timestamp(w_end[split_idx - 1]),
        "test_start_date": pd.Timestamp(w_start[test_start_idx]) if test_start_idx < len(w_start) else None,
    }


def month_end_anchor(monthly_series: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Re-anchors a month-start-labeled monthly series onto the LAST actual
    trading day within that same calendar month -- matching
    state_space.py's own placement exactly (it converts each label to a
    Period("M") and writes the value at that month's last daily-calendar
    position, state_space.py:123-131). Naively forward-filling from the
    CSV's day-01 label would let S1-S3 "know" each monthly value ~3-4 weeks
    before S0 ever reveals it. Returns a pd.Series indexed by actual trading
    dates (a subset of trading_dates), values = monthly_series' values,
    dropna'd -- ready to .reindex(trading_dates) and interpolate/ffill.
    """
    trading_dates = pd.DatetimeIndex(trading_dates)
    trading_periods = trading_dates.to_period("M")
    # last trading date within each calendar month present in the trading calendar
    month_end_lookup = pd.Series(trading_dates, index=trading_periods).groupby(level=0).max()

    monthly_series = monthly_series.dropna()
    monthly_periods = monthly_series.index.to_period("M")
    anchor_dates = monthly_periods.map(month_end_lookup)  # NaT where that month has no trading days at all
    valid = anchor_dates.notna()
    out = pd.Series(monthly_series.to_numpy()[valid], index=pd.DatetimeIndex(anchor_dates[valid]))
    return out.sort_index()


def train_only_zscore(daily_series: pd.Series, train_end_date: pd.Timestamp):
    """z-score a daily series using ONLY observations on or before
    train_end_date -- the one shared cutoff from get_window_manifest().
    Returns (z_scored_series, mu, sigma) so mu/sigma can be saved/reused."""
    train_part = daily_series[daily_series.index <= train_end_date]
    mu = float(train_part.mean())
    sigma = float(train_part.std())
    if sigma == 0 or np.isnan(sigma):
        raise ValueError(f"Degenerate train-period std ({sigma}) for a conditioning series -- cannot z-score.")
    return (daily_series - mu) / sigma, mu, sigma
