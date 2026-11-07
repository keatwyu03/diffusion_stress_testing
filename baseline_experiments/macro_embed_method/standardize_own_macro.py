"""
Per-method NN standardization for the macro-embedding experiment (S0-S4).

For one method k, fits the PRODUCTION standardizer (explore/nn_standardize.py
run_ticker, unmodified, same hyperparameters for every method) on that
method's OWN daily macro series v^(k), one asset at a time:

    z_{t,j}^(k) = (r_{t,j} - mu_{k,j}(v_t^(k))) / sigma_{k,j}(v_t^(k))

r = raw daily log returns from explore/macro_data_new.csv. The fit sample is
the SAME set of dates for all five methods: every date (config start/end
range) where all five daily states are defined, so S0-S4 differ only in the
macro series, never in the data they are fit on. Like production, the fit is
full-horizon (no train/test split).

Writes <out-dir>/<method dir>/:
    residuals.csv     z^(k)  (Date x 15 assets; NaN outside the fit dates)
    mu.csv, sigma.csv mu_k(v_t), sigma_k(v_t) on the same dates (for the inverse r = mu + sigma*z)
    standardizer.pt   per asset: mu_net / sigma_net state dicts + the macro normalization (mean, std)
    meta.json         fit dates, source files, hyperparameters, forward/inverse check

Usage:
    python standardize_own_macro.py --method s0 [--experiment full]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import common

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "macro_inclusion"))
sys.path.insert(0, os.path.join(common._ROOT, "explore"))
import main_exp1 as _m1  # noqa: E402
from nn_standardize import run_ticker  # noqa: E402
from window_conditioning import EXPERIMENTS, experiment_paths  # noqa: E402

METHODS = ["s0", "s1", "s2", "s3", "s4"]
HYPERPARAMS = dict(hidden=64, n_layers=2, n_epochs=1000, batch_size=128, delta=1e-3, lr=1e-3, seed=0)  # run_ticker defaults


def load_states(states_dir: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({m: pd.read_csv(os.path.join(states_dir, common.METHOD_DIRNAMES[m], "daily_state.csv"),
                                        index_col=0, parse_dates=True)["daily_state"].reindex(index)
                         for m in METHODS})


def fit_dates(returns: pd.DataFrame, states: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates where every asset return and every one of the five states is finite."""
    ok = np.isfinite(returns.to_numpy()).all(axis=1) & np.isfinite(states.to_numpy()).all(axis=1)
    return returns.index[ok]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--experiment", choices=list(EXPERIMENTS), default="full")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    paths = experiment_paths(args.experiment)
    cfg = common.get_default_config()
    df, _ = _m1.load_raw_data(cfg, csv_override=paths["macro_csv"])          # raw log returns, config date range
    returns = df[_m1.TICKERS]
    states = load_states(paths["states_dir"], returns.index)
    dates = fit_dates(returns, states)
    v = states.loc[dates, args.method].to_numpy(dtype=np.float32)
    print(f"[{args.method}] fit on {len(dates)} dates {dates[0].date()}..{dates[-1].date()} "
          f"(same dates for all methods); state from {paths['states_dir']}", flush=True)

    m = torch.tensor(v, dtype=torch.float32).reshape(-1, 1)
    m_mean, m_std = m.mean(0), m.std(0)   # exactly what run_ticker normalizes by
    resid, mu_df, sig_df, nets = {}, {}, {}, {}
    worst = 0.0
    for ticker in _m1.TICKERS:
        r = torch.tensor(returns.loc[dates, ticker].to_numpy(dtype=np.float32))
        model, z, mu, sigma = run_ticker(r, m, f"{args.method}_{ticker}", verbose=False, **HYPERPARAMS)
        z, mu, sigma = z.numpy().astype(np.float64), mu.numpy().astype(np.float64), sigma.numpy().astype(np.float64)
        err = float(np.max(np.abs(mu + sigma * z - r.numpy())))
        worst = max(worst, err)
        resid[ticker], mu_df[ticker], sig_df[ticker] = z, mu, sigma
        nets[ticker] = {"mu_net": model.mu_net.state_dict(), "sigma_net": model.sigma_net.state_dict(),
                        "best_loss": float(getattr(model, "best_loss", np.nan))}
        print(f"  {ticker}: best NLL {nets[ticker]['best_loss']:.4f}, z std {z.std():.3f}, "
              f"|mu+sigma*z - r| max {err:.1e}", flush=True)

    out = os.path.join(paths["standardization_dir"], common.METHOD_DIRNAMES[args.method])
    os.makedirs(out, exist_ok=True)
    for name, d in [("residuals", resid), ("mu", mu_df), ("sigma", sig_df)]:
        pd.DataFrame(d, index=dates).reindex(returns.index).to_csv(os.path.join(out, f"{name}.csv"), index_label="Date")
    torch.save({"nets": nets, "m_mean": float(m_mean), "m_std": float(m_std), "hyperparams": HYPERPARAMS,
                "tickers": _m1.TICKERS}, os.path.join(out, "standardizer.pt"))
    meta = {"method": args.method, "experiment": args.experiment,
            "macro_series": os.path.join(paths["states_dir"], common.METHOD_DIRNAMES[args.method], "daily_state.csv"),
            "returns_source": paths["macro_csv"], "returns": "raw daily log returns",
            "fit": "full horizon (production nn_standardize.run_ticker, unmodified)", "hyperparams": HYPERPARAMS,
            "n_fit_dates": int(len(dates)), "fit_start": str(dates[0].date()), "fit_end": str(dates[-1].date()),
            "macro_normalization": {"mean": float(m_mean), "std": float(m_std)},
            "inverse_check_max_abs_err": worst}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if worst > 1e-5:
        raise RuntimeError(f"forward/inverse check failed: max |mu + sigma*z - r| = {worst}")
    print(f"[{args.method}] wrote {out} (inverse check max err {worst:.1e})", flush=True)


if __name__ == "__main__":
    main()
