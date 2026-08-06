import os
import sys

import pandas as pd

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from tracking_regression import TrackingRegression, monthly_first_pc
from state_space import StateSpace


class LatentStateEstimator:
    """Estimates the single daily latent macro state from the growth/inflation
    panels in this directory. fit() returns the latent state as a daily
    pd.Series named "latent", for callers (diagnosis.py, import_data.py, ...)
    to use as the conditioning macro variable — written as the first column of
    the CSV, ahead of the asset columns in config.data.tickers.

    Two ALTERNATIVE methods that share only the per-group monthly PCA factors
    (growth PC1 and inflation PC1 — always separate PCAs, never joint):

    method="tracking_regression":
        PCA factor per group -> tracking regression per group -> daily
        tracking portfolios -> standardized average = daily latent.
        Groups with no selected variables are skipped; if only one group is
        active, "average" degenerates to that one group's tracking portfolio.

    method="state_space":
        PCA factor per group -> mixed-frequency Kalman filter: the RAW daily
        market variables (both panels appended, z-scored) drive one scalar
        latent regime state, and each active group's monthly factor anchors
        it at month-ends through its own estimated loading. No tracking
        regression is involved anywhere in this method. Always runs, for any
        non-empty selection — n (number of monthly factors) shrinks to match
        how many groups are active, but k (daily indicators) is untouched.

    variable selection (growth_vars / inflation_vars): picks which columns of
    that group's *_macro.csv feed monthly_first_pc's PCA — a list = only those
    columns, None or [] = drop the group entirely. At least one group must be
    selected. A dropped group contributes nothing to the estimate: neither its
    monthly factor nor its daily indicators. Within an ACTIVE group the
    selection narrows only the monthly/PCA side — that group's daily indicators
    are still every column of its *_daily.csv, since the point is tracking the
    selected monthly series with the full bucket of daily market data.
    """

    VARIABLES = ("growth", "inflation")

    def __init__(
        self,
        method: str = "state_space",
        data_dir: str = _dir,
        growth_vars=None,
        inflation_vars=None,
    ):
        self.method = method
        self.data_dir = data_dir
        self.var_selection = {"growth": growth_vars, "inflation": inflation_vars}
        self.trackers = {}       # name -> fitted TrackingRegression (method 1 only)
        self.state_space = None  # fitted StateSpace (method 2 only)
        self.latent = None

    def fit(self) -> pd.Series:
        # A group is active only if it has a non-empty variable selection —
        # None and [] both drop it. A dropped group contributes nothing: neither
        # its monthly factor (y) nor its daily indicators (x).
        active = [n for n in self.VARIABLES if self.var_selection[n]]
        if not active:
            raise ValueError(
                "growth_vars and inflation_vars are both empty — at least one "
                "group must name the columns to use."
            )

        macro, daily = {}, {}
        for name in active:
            daily[name] = pd.read_csv(os.path.join(self.data_dir, f"{name}_daily.csv"),
                                      index_col=0, parse_dates=True)
            macro[name] = pd.read_csv(os.path.join(self.data_dir, f"{name}_macro.csv"),
                                      index_col=0, parse_dates=True)
            macro[name] = macro[name][self.var_selection[name]]

        if self.method == "tracking_regression":
            for name in active:
                tr = TrackingRegression(macro[name], daily[name])
                tr.fit()
                self.trackers[name] = tr
            uts = pd.concat({n: tr.ut for n, tr in self.trackers.items()}, axis=1).dropna()
            z = (uts - uts.mean()) / uts.std()
            self.latent = z.mean(axis=1).rename("latent")

        elif self.method == "state_space":
            # y: one monthly PCA factor per ACTIVE group only — n shrinks if a
            # group was dropped via variable selection
            factors = pd.concat(
                {name: monthly_first_pc(macro[name])[0] for name in active},
                axis=1,
            )
            # x: raw daily market variables from the ACTIVE panels only (no
            # tracking regression), z-scored per column so the optimizer sees
            # comparable coefficient scales (yield changes are ~100x smaller
            # than commodity returns)
            x = pd.concat(daily, axis=1).dropna()
            x = (x - x.mean()) / x.std()
            self.state_space = StateSpace(y=factors, x=x).fit()
            self.latent = self.state_space.filtered_states()

        else:
            raise ValueError(f"unknown latent_method: {self.method!r}")
        return self.latent
