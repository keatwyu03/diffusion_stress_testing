import numpy as np
import pandas as pd
from scipy.optimize import minimize


class StateSpace():
    """One daily latent state [s, c] in vector form:
    x (T, k) daily indicators drive the state, y (T, n) monthly factors are
    observed at month ends through the intramonth cumulator c.
    y, x can be Series (n = k = 1, original behavior) or DataFrames."""

    def __init__(self, y, x):
        x = pd.DataFrame(x).dropna()
        self.dates = x.index
        self.x = x.to_numpy(float)              # (T, k)
        self.T, self.k = self.x.shape

        months = self.dates.to_period("M")
        self.is_month_start = ~months.duplicated()

        self.xlag = np.r_[self.x[:1], self.x[:-1]]

        y = pd.DataFrame(y)
        self.obs_names = [str(c) for c in y.columns]
        self.n = y.shape[1]
        is_month_end = np.r_[self.is_month_start[1:], True]

        # place each monthly factor at its month-end day, NaN elsewhere
        self.y = np.full((self.T, self.n), np.nan)
        for j, col in enumerate(y.columns):
            yj = y[col].dropna()
            y_by_month = dict(zip(yj.index.to_period("M"), yj.to_numpy(float)))
            for t in np.where(is_month_end)[0]:
                self.y[t, j] = y_by_month.get(months[t], np.nan)

        self.y[months == months[0]] = np.nan
        self.y[months == months[-1]] = np.nan

        self.params = None

    @property
    def param_names(self):
        return (["b0", "b1"]
                + [f"b2_{c}" for c in self.obs_names]
                + [f"a0_{c}" for c in self.obs_names]
                + [f"a1_{c}" for c in self.obs_names]
                + [f"log_var_y_{c}" for c in self.obs_names])

    def _unpack(self, params):
        k, n = self.k, self.n
        b0, b1 = params[0], params[1]
        b2 = np.asarray(params[2 : 2 + k])
        a0 = np.asarray(params[2 + k : 2 + k + n])
        a1 = np.asarray(params[2 + k + n : 2 + k + 2 * n])
        var_y = np.exp(np.asarray(params[2 + k + 2 * n : 2 + k + 3 * n]))
        return b0, b1, b2, a0, a1, var_y

    def filter(self, params):
        b0, b1, b2, a0, a1, var_y = self._unpack(params)

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
            const = (b0 + b2 @ self.xlag[t]) * np.ones(2)

            a = const + Tt @ a
            P = Tt @ P @ Tt.T + RQR

            obs = ~np.isnan(self.y[t])          # factors observed this day
            if obs.any():
                m = int(obs.sum())
                Z = np.column_stack([np.zeros(m), a1[obs]])     # (m, 2)
                v = self.y[t, obs] - (a0[obs] + Z @ a)
                F = Z @ P @ Z.T + np.diag(var_y[obs])
                Finv_v = np.linalg.solve(F, v)
                K = np.linalg.solve(F, Z @ P).T                 # P Zᵀ F⁻¹
                a = a + K @ v
                P = P - K @ Z @ P
                _, logdetF = np.linalg.slogdet(F)
                loglikelihood -= 0.5 * (m * np.log(2 * np.pi) + logdetF + v @ Finv_v)

            att[t] = a

        return loglikelihood, att

    def fit(self):
        start = np.r_[0.0, 0.9,
                      np.ones(self.k),                  # b2
                      np.zeros(self.n), np.ones(self.n),  # a0, a1
                      np.zeros(self.n)]                 # log_var_y
        obj = lambda p : -self.filter(p)[0]
        res = minimize(obj, start, method="Nelder-Mead",
                       options={"maxiter": 5000, "maxfev": 10000})
        self.params = res.x
        self.res = res
        return self

    def filtered_states(self):
        _, att = self.filter(self.params)
        return pd.Series(att[:, 0], index=self.dates, name="latent")
