"""
Portfolio evaluation of the CAUSAL macro-embedding runs (S0-S4, FULL macro set).
Uses existing outputs only -- no retraining, no change to the experiments.

  A  rare-event      : causal Doob-h results (each S_k keeps its own top-7.5% causal event definition)
                       generated = 2000 pooled samples per method, real = that method's event windows
  B  conditional-only: causal conditional-only results; all S0-S4 on the SAME 1200 historical windows
                       periods: "full" (all windows) and "common_nonrare" (windows that are not a causal
                       rare event under ANY of S0-S4's definitions)

Strategies (identical constraints for all methods): equal weight 1/15; global minimum variance (GMV);
Markowitz mean-variance  max_w  w'mu - (lambda/2) w'Sigma w,  lambda = 5.
Constraints: long-only, fully invested (w >= 0, sum w = 1). Sigma = sample covariance + 1e-8 I (the
project's minvar eps), mu = sample mean.

Weights are EX-ANTE: estimated from the 100 trading days of REAL raw daily log returns strictly before the
window's first day (100 = the project's existing portfolio formation length; a 10-day window is too short to
estimate a 15x15 covariance). The same weights are applied to the real window and to every generated sample
paired with it, for every S method. Generated rare-event samples are paired with the real event window whose
entry statistics / macro inputs reconstructed them (rare_event_artifacts.npz gen_window_index).

Portfolio path per window: daily rebalanced to the fixed weights, simple return sum_a w_a (exp(r_a)-1);
metrics generalize exp_evaluation's group-B definitions to arbitrary weights:
  variance of daily portfolio log returns (v2 definition), max drawdown of the wealth path,
  cumulative return (wealth_T - 1) -> W1 + |mean diff| for variance and drawdown, pinball loss on cumulative
  return at q in {0.01, 0.05, 0.95, 0.99} (pooled for A, per window for B -- as each existing evaluator does).

Usage:
    python portfolio_eval_causal_embed.py --experiment A
    python portfolio_eval_causal_embed.py --experiment B [--methods s0,s1]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, os.path.join(_HERE, "macro_inclusion")]
from build_rare_event_tables import render, fmt  # noqa: E402
import build_rare_event_tables as _tbl  # noqa: E402

TICKERS = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "TXN", "QCOM", "AMAT", "ADBE", "KO", "PG", "CL", "GIS", "KMB"]
METHODS = {"s0": ("exp2_s0", "S0_state_space", "S0 State-Space"), "s1": ("exp2_s1", "S1_linear", "S1 Linear Interp."),
           "s2": ("exp2_s2", "S2_forward_fill", "S2 Forward-Fill"), "s3": ("exp2_s3", "S3_cubic_spline", "S3 Cubic Spline"),
           "s4": ("exp2_s4", "S4_tracking_regression", "S4 Tracking Regr."),
           "s1p": ("exp2_s1p", "S1p_linear_extrapolation", "S1' Linear Extrap."),
           "s3p": ("exp2_s3p", "S3p_cubic_spline_extrapolation", "S3' Cubic Spline Extrap.")}
RETROSPECTIVE = {"s0": False, "s1": True, "s2": False, "s3": True, "s4": False, "s1p": False, "s3p": False}
STRATEGIES = ["Equal weight", "Global min variance", "Markowitz MV"]
LOOKBACK, RIDGE, LAMBDA = 100, 1e-8, 5.0
QUANTILES = (0.01, 0.05, 0.95, 0.99)
FULL = os.path.join(_HERE, "experiment_1 (Full)")
RARE_SRC = "/scr/kadenwu/Conditional_diffusion_v2_rare_emb_c/baseline_experiments/main_results/doobh_results_exp2_causal_ownstd"
RARE_SRC_NEW = "/scr/kadenwu/Conditional_diffusion_v2_rare_emb_c_new/baseline_experiments/main_results/doobh_results_exp2_causal_ownstd"


def rare_src(m):
    return RARE_SRC_NEW if m in ("s1p", "s3p") else RARE_SRC
COND_SRC = os.path.join(FULL, "causal", "macro_embed", "conditional_only")
OUT = os.path.join(FULL, "causal", "macro_embed", "portfolio_evaluation")
DISCLOSURE = ("S1 (linear) and S3 (cubic spline) are RETROSPECTIVE: their daily state between two monthly releases uses the NEXT "
              "release. S0/S2/S4 use only past releases but full-sample-fitted parameters (PCA/scaler; SSM MLE; tracking betas). "
              "All NN standardizers are fit on the full horizon and map same-day v_t to mu/sigma.")


# ---------------- weights ----------------
RAW = pd.read_csv(os.path.join(_ROOT, "explore", "macro_data_new.csv"), parse_dates=["Date"]).set_index("Date")[TICKERS]
CAL = RAW.index


def _solve(obj, grad, n):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    r = minimize(obj, np.ones(n) / n, jac=grad, bounds=[(0.0, 1.0)] * n, constraints=cons, method="SLSQP",
                 options={"ftol": 1e-15, "maxiter": 1000})
    w = np.clip(r.x, 0.0, None)
    return w / w.sum(), bool(r.success)


_CACHE = {}


def weights_for(start_date) -> dict:
    """Ex-ante weights from the LOOKBACK real trading days strictly before start_date."""
    start_date = pd.Timestamp(start_date)
    if start_date in _CACHE:
        return _CACHE[start_date]
    i = CAL.get_loc(start_date)
    hist = RAW.iloc[i - LOOKBACK:i].to_numpy(np.float64)
    if i < LOOKBACK or not np.isfinite(hist).all():
        raise ValueError(f"insufficient clean history before {start_date.date()}")
    cov = np.cov(hist, rowvar=False, ddof=1) + RIDGE * np.eye(hist.shape[1])
    mu = hist.mean(0)
    n = hist.shape[1]
    w_gmv, ok1 = _solve(lambda w: w @ cov @ w, lambda w: 2 * cov @ w, n)
    w_mv, ok2 = _solve(lambda w: -(w @ mu) + 0.5 * LAMBDA * w @ cov @ w, lambda w: -mu + LAMBDA * cov @ w, n)
    if not (ok1 and ok2):
        raise RuntimeError(f"optimizer failed for {start_date.date()}")
    out = {"Equal weight": np.ones(n) / n, "Global min variance": w_gmv, "Markowitz MV": w_mv,
           "_hist_end": CAL[i - 1], "_hist_start": CAL[i - LOOKBACK]}
    _CACHE[start_date] = out
    return out


# ---------------- portfolio paths ----------------
def path_stats(R, W):
    """R: (N, A, T) log returns; W: (N, A) weights -> variance, max drawdown, cumulative return per window."""
    simple = np.einsum("nat,na->nt", np.expm1(R), W)
    logr = np.log1p(simple)
    wealth = np.concatenate([np.ones((len(R), 1)), np.cumprod(1.0 + simple, axis=1)], axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    return logr.var(axis=1, ddof=1), ((peak - wealth) / peak).max(axis=1), wealth[:, -1] - 1.0


def pinball(y, qhat, q):
    d = y - qhat
    return np.where(d >= 0, q * d, (q - 1) * d)


def compare(real, gen):
    """real: dict of (n,) arrays; gen: dict of (n*K,) pooled arrays -> W1 / |mean diff| (exp_evaluation style)."""
    out = {}
    for k in ("var", "mdd"):
        out[f"{k}_w1"] = float(wasserstein_distance(real[k], gen[k]))
        out[f"{k}_meandiff"] = float(abs(real[k].mean() - gen[k].mean()))
        out[f"{k}_real_mean"], out[f"{k}_gen_mean"] = float(real[k].mean()), float(gen[k].mean())
    return out


# ---------------- experiment A ----------------
def experiment_a(methods):
    rows, checks = [], []
    for m in methods:
        ch, _, label = METHODS[m]
        a = np.load(os.path.join(rare_src(m), ch, "rare_event_artifacts.npz"))
        meta = json.load(open(os.path.join(rare_src(m), ch, "rare_event_artifacts.json")))
        assert meta["event_causal"] is True and meta["split"].startswith("train"), meta
        real, gen = a["real_returns_reconstructed"].astype(np.float64), a["gen_returns"].astype(np.float64)
        rd, gd = pd.to_datetime(a["real_window_start"]), pd.to_datetime(a["gen_window_start"])
        assert np.isfinite(real).all() and np.isfinite(gen).all()
        for s in STRATEGIES:
            Wr = np.stack([weights_for(d)[s] for d in rd]); Wg = np.stack([weights_for(d)[s] for d in gd])
            vr, mr, cr = path_stats(real, Wr); vg, mg, cg = path_stats(gen, Wg)
            row = {"method": label, "method_key": m, "retrospective": RETROSPECTIVE[m], "strategy": s,
                   "n_real_events": len(real), "n_generated": len(gen),
                   **compare({"var": vr, "mdd": mr}, {"var": vg, "mdd": mg}),
                   "cumret_real_mean": float(cr.mean()), "cumret_gen_mean": float(cg.mean())}
            for q in QUANTILES:  # pooled: one generated quantile vs every real event (as doobh_evaluation)
                row[f"pinball_{q}"] = float(pinball(cr, np.quantile(cg, q), q).mean())
            rows.append(row)
        checks.append(f"{m}: {len(real)} events, {len(gen)} generated; history ends before every window start: "
                      f"{all(weights_for(d)['_hist_end'] < d for d in rd.union(gd))}")
    return pd.DataFrame(rows), checks


# ---------------- experiment B ----------------
def event_dates_any(methods):
    """Start dates of causal rare-event windows under ANY of the evaluated methods' own definitions."""
    s = set()
    for m in methods:
        a = np.load(os.path.join(rare_src(m), METHODS[m][0], "rare_event_artifacts.npz"))
        s |= set(pd.to_datetime(a["real_window_start"]))
    return s


def experiment_b(methods):
    evals = {m: np.load(os.path.join(COND_SRC, METHODS[m][1], "diffusion_causal", "eval_data.npz")) for m in methods}
    ref = evals[methods[0]]
    for m in methods:
        assert np.array_equal(evals[m]["window_start_dates"], ref["window_start_dates"]) and np.array_equal(evals[m]["X_eval"], ref["X_eval"]), m
    dates = pd.to_datetime(ref["window_start_dates"])
    X = ref["X_eval"].astype(np.float64)
    ev_any = event_dates_any(methods)
    nonrare = np.array([d not in ev_any for d in dates])
    W = {s: np.stack([weights_for(d)[s] for d in dates]) for s in STRATEGIES}
    checks = [f"identical {len(dates)} windows for {methods}", f"common non-rare windows: {nonrare.sum()} / {len(dates)} "
              f"({(~nonrare).sum()} are a causal rare event for at least one of {methods})",
              f"history ends before every window start: {all(weights_for(d)['_hist_end'] < d for d in dates)}"]
    rows = []
    for m in methods:
        gen = np.load(os.path.join(COND_SRC, METHODS[m][1], "diffusion_causal", "generated_samples.npy")).astype(np.float64)
        n, K = gen.shape[:2]
        assert n == len(X) and np.isfinite(gen).all()
        for s in STRATEGIES:
            vr, mr, cr = path_stats(X, W[s])
            vg, mg, cg = path_stats(gen.reshape(n * K, *gen.shape[2:]), np.repeat(W[s], K, axis=0))
            vg, mg, cg = vg.reshape(n, K), mg.reshape(n, K), cg.reshape(n, K)
            for period, sel in [("full", np.ones(n, bool)), ("common_nonrare", nonrare)]:
                row = {"method": METHODS[m][2], "method_key": m, "retrospective": RETROSPECTIVE[m], "strategy": s,
                       "period": period, "n_windows": int(sel.sum()), "K": K,
                       **compare({"var": vr[sel], "mdd": mr[sel]}, {"var": vg[sel].ravel(), "mdd": mg[sel].ravel()}),
                       "cumret_real_mean": float(cr[sel].mean()), "cumret_gen_mean": float(cg[sel].mean())}
                for q in QUANTILES:  # per window: that window's K samples give the quantile (as group_b_metrics)
                    row[f"pinball_{q}"] = float(pinball(cr[sel], np.quantile(cg[sel], q, axis=1), q).mean())
                rows.append(row)
    return pd.DataFrame(rows), checks


def tables(df, out_dir, title, group_cols):
    _tbl.FOOTNOTE = ("Weights: ex-ante from the 100 real trading days before each window; long-only, fully invested; MV lambda=5. "
                     "Lower is better for every metric.\n" + DISCLOSURE[:180] + "... (see README)")
    for key, t in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        name = "_".join(str(k).replace(" ", "_").lower() for k in key)
        cols = ["var_w1", "var_meandiff", "mdd_w1", "mdd_meandiff", "pinball_0.05", "pinball_0.95"]
        cells = [[r["method"] + (" *" if r["retrospective"] else "")] + [fmt(r[c], 3) for c in cols] for _, r in t.iterrows()]
        raw = [[None] + [r[c] for c in cols] for _, r in t.iterrows()]
        render(["Method (* retrospective)", "Variance W1", "Variance\n|mean diff|", "Max DD W1", "Max DD\n|mean diff|",
                "Pinball 5%\n(cum. return)", "Pinball 95%\n(cum. return)"], cells,
               f"{title} — {' — '.join(map(str, key))}", os.path.join(out_dir, f"table_{name}.png"),
               best_cols=tuple(range(1, 7)), raw=raw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=["A", "B"])
    p.add_argument("--methods", default="s0,s1,s2,s3,s4")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    sub = "rare_event" if args.experiment == "A" else "conditional_only"
    out_dir = args.out_dir or os.path.join(OUT, sub)
    os.makedirs(out_dir, exist_ok=True)
    df, checks = experiment_a(methods) if args.experiment == "A" else experiment_b(methods)
    df.to_csv(os.path.join(out_dir, "portfolio_metrics.csv"), index=False)
    title = ("Causal Rare-Event (Doob-h) Portfolio Evaluation" if args.experiment == "A"
             else "Causal Conditional-Only Portfolio Evaluation")
    tables(df, out_dir, title, ["strategy"] if args.experiment == "A" else ["period", "strategy"])
    w = pd.DataFrame([{"window_start": d.date(), "history": f"{v['_hist_start'].date()}..{v['_hist_end'].date()}",
                       **{f"{s} max weight": float(v[s].max()) for s in STRATEGIES},
                       **{f"{s} n>1%": int((v[s] > 0.01).sum()) for s in STRATEGIES}} for d, v in sorted(_CACHE.items())])
    w.to_csv(os.path.join(out_dir, "weights_summary.csv"), index=False)
    with open(os.path.join(out_dir, "README.json"), "w") as f:
        json.dump({"experiment": args.experiment, "methods": methods, "strategies": STRATEGIES, "lookback_days": LOOKBACK,
                   "constraints": "long-only, fully invested", "markowitz_lambda": LAMBDA, "ridge": RIDGE,
                   "source": RARE_SRC if args.experiment == "A" else COND_SRC, "checks": checks,
                   "disclosure": DISCLOSURE, "real_reference": ("real event windows reconstructed with each method's own "
                   "standardizer (winsorized-consistent, as in the rare-event return-unit tables)" if args.experiment == "A"
                   else "real raw log-return windows (eval_data.npz X_eval)")}, f, indent=2)
    for c in checks:
        print("[check]", c)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
