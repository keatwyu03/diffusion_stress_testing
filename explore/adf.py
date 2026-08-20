import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import mackinnonp

# ─────────────────────────────────────────────────────────────────────────────
# Augmented Dickey-Fuller test, constant + trend variant:
#
#   dy_t = alpha + beta*t + gamma*y_{t-1} + sum_{i=1}^p phi_i * dy_{t-i} + eps_t
#
# H0: gamma = 0  (y has a unit root -> non-stationary)
# H1: gamma < 0  (y is stationary around a deterministic trend)
#
# Lag order p is chosen by argmin_p AIC(p) over a fixed candidate set, each
# candidate refit on the SAME (longest, p=p_max) usable sample so AIC values
# are comparable across p.
#
# Test statistic t_gamma = gamma_hat / SE(gamma_hat) is NOT asymptotically
# t-distributed under H0 (y_{t-1} is itself a random walk under H0, so the
# usual t-table doesn't apply) -- p-values come from MacKinnon's ADF
# response-surface tables via statsmodels.
# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
RESID_CSV = os.path.join(HERE, "ff_macro_data.csv")


def _build_design(y, p, max_lag):
    """(dy_t, [1, t, y_{t-1}, dy_{t-1}, ..., dy_{t-p}]) rows, aligned so every
    candidate lag order p in [0, max_lag] uses the same sample size/start
    index (t = max_lag+1 .. n-1) -- required for AIC to be comparable across p."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    dy = np.diff(y)                      # dy[t] = y[t+1] - y[t], length n-1

    start = max_lag                      # first usable row index into dy/y such that
                                          # dy[start-1-i] exists for i=0..max_lag-1
    rows_dy = dy[start:]                 # dy_t for t = start+1 .. n-1 (in dy's own indexing)
    n_obs = len(rows_dy)

    trend = np.arange(start + 1, start + 1 + n_obs, dtype=np.float64)  # t index
    const = np.ones(n_obs)
    y_lag1 = y[start:start + n_obs]      # y_{t-1}

    cols = [const, trend, y_lag1]
    for i in range(1, p + 1):
        cols.append(dy[start - i:start - i + n_obs])  # dy_{t-i}

    X = np.column_stack(cols)
    return rows_dy, X


def _ols(y, X):
    """Beta, residual variance (n-k denom), robust-free OLS SE via (X'X)^-1 * sigma2,
    and log-likelihood for AIC (Gaussian errors)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    ssr = resid @ resid
    sigma2 = ssr / (n - k)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    sigma2_mle = ssr / n
    loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)
    aic = -2 * loglik + 2 * k

    return beta, se, aic


def select_lag_aic(y, max_lag):
    """p* = argmin_p AIC(p) over p = 0..max_lag, all refit on the common
    max_lag-truncated sample. Returns (p_star, {p: aic})."""
    aics = {}
    for p in range(max_lag + 1):
        dy_t, X = _build_design(y, p, max_lag)
        _, _, aic = _ols(dy_t, X)
        aics[p] = aic
    p_star = min(aics, key=aics.get)
    return p_star, aics


def adf_test(y, max_lag=None, regression="ct"):
    """Augmented Dickey-Fuller test with AIC-selected lag order.

    y: 1D array-like series to test.
    max_lag: largest lag order to search over for AIC selection; defaults to
        the Schwert (1989) rule of thumb ceil(12*(n/100)^0.25).
    regression: only "ct" (constant + trend) is implemented.

    Returns a dict with p (selected lag), gamma, se_gamma, stat (gamma/se),
    pvalue (MacKinnon), n_obs, aic_by_lag.
    """
    if regression != "ct":
        raise NotImplementedError("only regression='ct' (constant + trend) is implemented")

    y = np.asarray(y, dtype=np.float64)
    y = y[~np.isnan(y)]
    n = len(y)

    if max_lag is None:
        max_lag = int(np.ceil(12 * (n / 100) ** 0.25))

    p_star, aics = select_lag_aic(y, max_lag)

    dy_t, X = _build_design(y, p_star, max_lag)
    beta, se, _ = _ols(dy_t, X)

    gamma_hat = beta[2]
    se_gamma = se[2]
    stat = gamma_hat / se_gamma
    pvalue = mackinnonp(stat, regression="ct", N=1)

    return {
        "p": p_star,
        "gamma": gamma_hat,
        "se_gamma": se_gamma,
        "stat": stat,
        "pvalue": pvalue,
        "n_obs": len(dy_t),
        "max_lag": max_lag,
        "aic_by_lag": aics,
    }


if __name__ == "__main__":
    z = pd.read_csv(RESID_CSV, parse_dates=["Date"])
    tickers = [c for c in z.columns if c not in ("Date", "m_t")]

    print(f"{'ticker':<8}{'p':>4}{'gamma':>12}{'se(gamma)':>12}{'t_gamma':>12}{'p-value':>12}{'n_obs':>10}")
    for t in tickers:
        res = adf_test(z[t].values)
        print(f"{t:<8}{res['p']:>4}{res['gamma']:>12.5f}{res['se_gamma']:>12.5f}"
              f"{res['stat']:>12.4f}{res['pvalue']:>12.5f}{res['n_obs']:>10}")
