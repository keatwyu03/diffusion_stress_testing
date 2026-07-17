import os
import sys

import pandas as pd

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from tracking_regression import TrackingRegression
from state_space import StateSpace


class LatentStateEstimator:
    """Estimates the single daily latent macro state from the growth/inflation
    panels in this directory. fit() returns the latent state as a daily
    pd.Series named "latent", for callers (diagnosis.py, import_data.py, ...)
    to use as the conditioning macro variable (config.data.tickers[0]).

    method:
        "state_space"         — joint Kalman filter: both daily tracking
                                 portfolios drive one latent state, observed
                                 through both monthly factors
        "tracking_regression" — standardized average of the daily tracking
                                 portfolios (no Kalman filter)
    """

    VARIABLES = ("growth", "inflation")

    def __init__(self, method: str = "state_space", data_dir: str = _dir):
        self.method = method
        self.data_dir = data_dir
        self.trackers = {}       # name -> fitted TrackingRegression
        self.state_space = None  # fitted StateSpace (state_space method only)
        self.latent = None

    def fit(self) -> pd.Series:
        for name in self.VARIABLES:
            macro = pd.read_csv(os.path.join(self.data_dir, f"{name}_macro.csv"),
                                index_col=0, parse_dates=True)
            daily = pd.read_csv(os.path.join(self.data_dir, f"{name}_daily.csv"),
                                index_col=0, parse_dates=True)
            tr = TrackingRegression(macro, daily)
            tr.fit()
            self.trackers[name] = tr

        uts = pd.concat({n: tr.ut for n, tr in self.trackers.items()}, axis=1).dropna()

        if self.method == "state_space":
            factors = pd.concat({n: tr.factor for n, tr in self.trackers.items()}, axis=1)
            self.state_space = StateSpace(y=factors, x=uts).fit()
            self.latent = self.state_space.filtered_states()
        elif self.method == "tracking_regression":
            z = (uts - uts.mean()) / uts.std()
            self.latent = z.mean(axis=1).rename("latent")
        else:
            raise ValueError(f"unknown latent_method: {self.method!r}")
        return self.latent
