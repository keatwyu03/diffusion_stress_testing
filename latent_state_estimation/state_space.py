import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from numba import njit


@njit(cache=True)
def _filter_numba(params, x, xlag, y, is_month_start, k, n, T):
    """Numba-compiled core of StateSpace.filter(). Same recursion as the
    pure-Python version — see filter() below — just compiled so the
    per-day Python-loop overhead (dominant cost across ~6600 days x
    thousands of objective evals during MLE) drops out."""
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
    loglikelihood = 0.0

    for t in range(T):
        gamma = 0.0 if is_month_start[t] else 1.0

        Tt = np.array([[b1, 0.0], [b1, gamma]])
        const_val = b0 + b2 @ xlag[t]
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

        att[t] = a

    return loglikelihood, att


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
        self.is_month_start = np.asarray(~months.duplicated())

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
        _, att = self.filter(self.params)
        return pd.Series(att[:, 0], index=self.dates, name="latent")
