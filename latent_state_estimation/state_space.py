import numpy as np
import pandas as pd 
from scipy.optimize import minimize


class StateSpace():
    def __init__(self, y, x):
        x = x.dropna()
        self.dates = x.index
        self.x = x.to_numpy(float)
        self.T = len(self.x)

        months = self.dates.to_period("M")
        self.is_month_start = ~months.duplicated()

        self.xlag = np.r_[self.x[0], self.x[:-1]]

        y = y.dropna()
        y_by_month = dict(zip(y.index.to_period("M"), y.to_numpy(float)))
        is_month_end = np.r_[self.is_month_start[1:], True]

        self.y = np.full(self.T, np.nan)
        for t in np.where(is_month_end)[0]:
            self.y[t] = y_by_month.get(months[t], np.nan)
        
        self.y[months == months[0]] = np.nan
        self.y[months == months[-1]] = np.nan

        self.params = None

    
    def filter(self, params): 
        b0, b1, b2, a0, a1, log_var_y = params
        var_y = np.exp(log_var_y)

        a = np.zeros(2)
        P = np.eye(2) * 1e4
        RQR = np.ones((2,2))

        att = np.zeros((self.T, 2))   # filtered state [s, c] per day

        loglikelihood = 0.0
        for t in range(self.T):
            if self.is_month_start[t]: 
                gamma = 0.0
            else:
                gamma = 1.0
            
            Tt = np.array([[b1, 0.0], [b1, gamma]])
            const = (b0 + b2 * self.xlag[t]) * np.ones(2)

            a = const + Tt @ a
            P = Tt @ P @ Tt.T + RQR

            if not np.isnan(self.y[t]):
                Z = np.array([0.0, a1])
                v = self.y[t] - (a0 + Z @ a)
                F = Z @ P @ Z + var_y
                K = P @ Z / F
                a = a + K * v
                P = P - np.outer(K, Z @ P)
                loglikelihood -= 0.5 * (np.log(2 * np.pi) + np.log(F) + v**2 / F)

            att[t] = a

        return loglikelihood, att

    def fit(self):
        start = np.array([0.0, 0.9, 1.0, 0.0, 1.0, 0.0])
        obj = lambda p : -self.filter(p)[0]
        res = minimize(obj, start, method="Nelder-Mead", options={"maxiter": 5000})
        self.params = res.x
        self.res = res
        return self

    def filtered_states(self):
        _, att = self.filter(self.params)
        return pd.Series(att[:, 0], index=self.dates, name="s")