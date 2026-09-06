import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from numba import njit


@njit(cache=True)
def _filter_numba(params, x, xlag, y, is_month_start, day_in_month, use_average, k, n, T):
    """Numba-compiled core of StateSpace.filter(). Same recursion as the
    pure-Python version — see filter() below — just compiled so the
    per-day Python-loop overhead (dominant cost across ~6600 days x
    thousands of objective evals during MLE) drops out.

    The intramonth cumulator c can be either a running SUM (use_average=False,
    original behavior) or a running MEAN (use_average=True) of the same daily
    increment (b1*s_{t-1} + b0 + b2@x_{t-1}):
        sum:     c_t = b1*s_{t-1} + c_{t-1}             + (b0 + b2@x_{t-1})
        average: c_t = (b1/d_t)*s_{t-1} + (1-1/d_t)*c_{t-1} + (b0 + b2@x_{t-1})/d_t
    where d_t = day_in_month[t] is the number of trading days elapsed since
    (and including) the most recent month-start reset -- known exactly from
    the calendar, not estimated. At d_t=1 (month-start) the average form
    reduces to c_t = b1*s_{t-1} + (b0 + b2@x_{t-1}), i.e. c_{t-1} is fully
    discarded, matching the sum form's reset-to-zero-then-add-one-term.

    Also records the one-step-ahead prediction error (innovation)
    v = y[t] - (a0 + a1*s_t) for every observed monthly anchor, so RMSE of
    the monthly anchors can be reported after fitting (see
    StateSpace.anchor_rmse()). NaN where that column has no observation at
    t (i.e. every day except its month-end)."""
    b0 = params[0]
    b1 = params[1]
    b2 = params[2 : 2 + k]
    a0 = params[2 + k : 2 + k + n]
    a1 = params[2 + k + n : 2 + k + 2 * n]
    var_y = np.exp(params[2 + k + 2 * n : 2 + k + 3 * n])

    a = np.zeros(2)
    P = np.eye(2) * 1e4
    RQR = np.ones((2, 2))

    att = np.zeros((T, 2))
    resid = np.full((T, n), np.nan)
    loglikelihood = 0.0

    for t in range(T):
        gamma = 0.0 if is_month_start[t] else 1.0
        const_val = b0 + b2 @ xlag[t]

        if use_average:
            w = 1.0 / day_in_month[t]
            Tt = np.array([[b1, 0.0], [b1 * w, 1.0 - w]])
            const = np.array([const_val, const_val * w])
        else:
            Tt = np.array([[b1, 0.0], [b1, gamma]])
            const = np.array([const_val, const_val])

        a = const + Tt @ a
        P = Tt @ P @ Tt.T + RQR

        obs = ~np.isnan(y[t])
        m = int(obs.sum())
        if m > 0:
            a1_obs = a1[obs]
            Z = np.zeros((m, 2))
            Z[:, 1] = a1_obs
            v = y[t][obs] - (a0[obs] + Z @ a)
            F = Z @ P @ Z.T + np.diag(var_y[obs])
            Finv_v = np.linalg.solve(F, v)
            K = np.linalg.solve(F, Z @ P).T
            a = a + K @ v
            P = P - K @ Z @ P
            sign, logdetF = np.linalg.slogdet(F)
            loglikelihood -= 0.5 * (m * np.log(2 * np.pi) + logdetF + v @ Finv_v)

            obs_idx = np.where(obs)[0]
            for i in range(m):
                resid[t, obs_idx[i]] = v[i]

        att[t] = a

    return loglikelihood, att, resid


class StateSpace():
    """One daily latent state [s, c] in vector form:
    x (T, k) daily indicators drive the state, y (T, n) monthly factors are
    observed at month ends through the intramonth cumulator c.
    y, x can be Series (n = k = 1, original behavior) or DataFrames."""

    def __init__(self, y, x, accumulator: str = "sum"):
        """accumulator: "sum" (original -- c is a running SUM of daily
        increments over the month) or "average" (c is a running MEAN of the
        same daily increments, using the exact trading-day count elapsed
        since the last month-start reset -- see _filter_numba)."""
        if accumulator not in ("sum", "average"):
            raise ValueError(f"accumulator must be 'sum' or 'average', got {accumulator!r}")
        self.accumulator = accumulator

        x = pd.DataFrame(x).dropna()
        self.dates = x.index
        self.x = x.to_numpy(float)              # (T, k)
        self.T, self.k = self.x.shape

        months = self.dates.to_period("M")
        self.is_month_start = np.asarray(~months.duplicated())

        # day_in_month[t] = number of trading days elapsed since (and
        # including) the most recent month-start reset, i.e. 1 on a
        # month-start day, 2 the next day, etc. -- known exactly from the
        # calendar, used only when accumulator="average".
        self.day_in_month = np.empty(self.T, dtype=np.float64)
        _counter = 0
        for _t in range(self.T):
            _counter = 1 if self.is_month_start[_t] else _counter + 1
            self.day_in_month[_t] = _counter

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
        params = np.asarray(params, dtype=np.float64)
        return _filter_numba(
            params, self.x, self.xlag, self.y, self.is_month_start,
            self.day_in_month, self.accumulator == "average",
            self.k, self.n, self.T,
        )

    def fit(self):
        start = np.r_[0.0, 0.9,
                      np.ones(self.k),                  # b2
                      np.zeros(self.n), np.ones(self.n),  # a0, a1
                      np.zeros(self.n)]                 # log_var_y
        obj = lambda p : -self.filter(p)[0]

        # Nelder-Mead: derivative-free, so it never takes the large exploratory
        # steps that can drive log_var_y toward -inf and make F singular (this
        # happened with L-BFGS-B's finite-difference-gradient steps). filter()
        # is numba-compiled now, so NM's larger iteration budget is still fast
        # in wall-clock time — the old slowness was the per-eval Python loop,
        # not really the optimizer choice.
        maxiter = 20000
        pbar = tqdm(total=maxiter, desc="StateSpace MLE (Nelder-Mead)", unit="it")

        def callback(xk):
            pbar.update(1)
            pbar.set_postfix(negloglik=f"{obj(xk):.2f}")

        res = minimize(obj, start, method="Nelder-Mead",
                       options={"maxiter": maxiter, "maxfev": 40000},
                       callback=callback)
        pbar.close()

        self.params = res.x
        self.res = res
        return self

    def filtered_states(self):
        _, att, _ = self.filter(self.params)
        return pd.Series(att[:, 0], index=self.dates, name="latent")

    def anchor_rmse(self) -> pd.Series:
        """RMSE of the one-step-ahead prediction error for each monthly
        anchor series, at the fitted params: sqrt(nanmean(v**2)) over all
        month-end observations of that column, v = y_actual - y_predicted
        (the Kalman filter's innovation, computed in _filter_numba and
        returned as `resid`)."""
        _, _, resid = self.filter(self.params)
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=0))
        return pd.Series(rmse, index=self.obs_names, name="anchor_rmse")

    def anchor_r2(self) -> pd.Series:
        """R^2 of the anchor fit for each monthly anchor series j:
            R^2_j = 1 - sum_m (z_mj - zhat_mj)^2 / sum_m (z_mj - zbar_j)^2
        sum_m (z_mj - zhat_mj)^2 is resid**2 summed over observed month-end
        rows (same `v` as anchor_rmse); zbar_j is the mean of the OBSERVED
        z_mj for that column (over the same rows resid is defined on)."""
        _, _, resid = self.filter(self.params)
        ss_res = np.nansum(resid ** 2, axis=0)

        observed = ~np.isnan(resid)
        z = np.where(observed, self.y, np.nan)
        z_bar = np.nanmean(z, axis=0)
        ss_tot = np.nansum((z - z_bar) ** 2, axis=0)

        r2 = 1.0 - ss_res / ss_tot
        return pd.Series(r2, index=self.obs_names, name="anchor_r2")
