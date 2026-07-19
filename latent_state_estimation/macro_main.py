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
    to use as the conditioning macro variable (config.data.tickers[0]).

    Two ALTERNATIVE methods that share only the per-group monthly PCA factors
    (growth PC1 and inflation PC1 — always separate PCAs, never joint):

    method="tracking_regression":
        PCA factor per group -> tracking regression per group -> daily
        tracking portfolios -> standardized average = daily latent.

    method="state_space":
        PCA factor per group -> mixed-frequency Kalman filter: the RAW daily
        market variables (both panels appended, z-scored) drive one scalar
        latent regime state, and both monthly factors anchor it at month-ends
        through their own estimated loadings. No tracking regression is
        involved anywhere in this method.
    """

    VARIABLES = ("growth", "inflation")

    def __init__(self, method: str = "state_space", data_dir: str = _dir):
        self.method = method
        self.data_dir = data_dir
        self.trackers = {}       # name -> fitted TrackingRegression (method 1 only)
        self.state_space = None  # fitted StateSpace (method 2 only)
        self.latent = None

    def fit(self) -> pd.Series:
        macro, daily = {}, {}
        for name in self.VARIABLES:
            macro[name] = pd.read_csv(os.path.join(self.data_dir, f"{name}_macro.csv"),
                                      index_col=0, parse_dates=True)
            daily[name] = pd.read_csv(os.path.join(self.data_dir, f"{name}_daily.csv"),
                                      index_col=0, parse_dates=True)

        if self.method == "tracking_regression":
            for name in self.VARIABLES:
                tr = TrackingRegression(macro[name], daily[name])
                tr.fit()
                self.trackers[name] = tr
            uts = pd.concat({n: tr.ut for n, tr in self.trackers.items()}, axis=1).dropna()
            z = (uts - uts.mean()) / uts.std()
            self.latent = z.mean(axis=1).rename("latent")

        elif self.method == "state_space":
            # y: the two per-group monthly PCA factors, both observing the
            # single latent state through their own estimated loadings
            factors = pd.concat(
                {name: monthly_first_pc(macro[name])[0] for name in self.VARIABLES},
                axis=1,
            )
            # x: raw daily market variables from both panels (no tracking
            # regression), z-scored per column so the optimizer sees comparable
            # coefficient scales (yield changes are ~100x smaller than
            # commodity returns)
            x = pd.concat(daily, axis=1).dropna()
            x = (x - x.mean()) / x.std()
            self.state_space = StateSpace(y=factors, x=x).fit()
            self.latent = self.state_space.filtered_states()

        else:
            raise ValueError(f"unknown latent_method: {self.method!r}")
        return self.latent
