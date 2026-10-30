"""
Fits the existing neural conditional mean/volatility standardizer
(explore/nn_standardize.py's MLP + NLL_Stationarity, reused unmodified) per
asset, per method (S0-S4), on TRAIN dates only, then freezes it and applies
it to produce the full (train+test) standardized residual series.

Why not reuse explore/nn_standardize.py's own run_ticker()/load_data(): confirmed
(approved-plan review + independent verification) that run_ticker() recomputes
macro-state normalization from whatever `m` it's given with no train/test
awareness (nn_standardize.py:152, `m_full = (m - m.mean(0)) / m.std(0)`) and
no mechanism to inject pre-fit stats -- calling it on train+test combined
would leak test-period information into that normalization. It also expects
torch.Tensor inputs, not raw numpy (confirmed: would crash in MLP.forward
via nn.Linear on a bare ndarray). This module is the small train/transform
wrapper the approved plan calls for instead: fit mu_net/sigma_net AND the
`m`-normalization stats on train only, then standardize() on the full
series using those frozen stats -- never refit/renormalize on test.

Daily macro state here is already a train-only z-scored, group-averaged
series (daily_macro_states.py's aggregate_groups) -- it is normalized ONE
more time with its own train-only mean/std before feeding the MLP, since
averaging two independently-unit-variance series doesn't itself produce
unit variance, and nn_standardize.py's own convention expects a roughly
unit-scale `m`.
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
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "explore"))
from nn_standardize import MLP, NLL_Stationarity  # noqa: E402

OUT_DIR = os.path.join(_HERE, "exp2_results")

# Identical across every method/asset -- only `m` (the daily state) changes.
HPARAMS = dict(hidden=64, n_layers=2, delta=1e-3, lr=1e-3, seed=0)


def load_returns(trading_dates):
    csv_path = os.path.join(_ROOT, "explore", "macro_data_new.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    return df[common.TICKERS].reindex(trading_dates)


def fit_one(r_np, m_np, train_mask, n_epochs, batch_size, hp):
    torch.manual_seed(hp["seed"])
    mu_net = MLP(p=1, hidden=hp["hidden"], n_layers=hp["n_layers"], out=1)
    sigma_net = MLP(p=1, hidden=hp["hidden"], n_layers=hp["n_layers"], out=1)
    model = NLL_Stationarity(mu_net, sigma_net, delta=hp["delta"], lr=hp["lr"])

    r_t = torch.tensor(r_np, dtype=torch.float32)
    m_t = torch.tensor(m_np, dtype=torch.float32).reshape(-1, 1)

    model.fit(r_t[train_mask], m_t[train_mask], n_epochs=n_epochs, batch_size=batch_size, verbose=False)
    resid_full = model.standardize(r_t, m_t).numpy()
    return model, resid_full


def run_method(method: str, manifest, cfg, n_epochs: int, batch_size: int, assets=None):
    trading_dates = manifest["df"].index
    train_end_date = manifest["train_end_date"]
    train_mask = np.asarray(trading_dates <= train_end_date)

    state_csv = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[method], "daily_state.csv")
    raw_state = pd.read_csv(state_csv, index_col=0, parse_dates=True)["daily_state"].reindex(trading_dates)
    m_z, m_mu, m_sigma = common.train_only_zscore(raw_state, train_end_date)

    returns = load_returns(trading_dates)
    assets = assets or common.TICKERS

    out_dir = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[method], "standardizer")
    os.makedirs(out_dir, exist_ok=True)

    train_diag, test_diag = {}, {}
    residuals = {}
    for ticker in assets:
        r_np = returns[ticker].to_numpy(dtype=np.float32)
        m_np = m_z.to_numpy(dtype=np.float32)
        valid = np.isfinite(r_np) & np.isfinite(m_np)
        fit_mask = train_mask & valid
        if fit_mask.sum() < 20:
            print(f"  [skip] {ticker}: only {fit_mask.sum()} valid train observations")
            continue

        model, resid_full = fit_one(
            np.where(valid, r_np, 0.0), np.where(valid, m_np, 0.0), fit_mask, n_epochs, batch_size, HPARAMS,
        )
        resid_full = np.where(valid, resid_full, np.nan)
        residuals[ticker] = resid_full

        train_r = resid_full[train_mask & valid]
        test_r = resid_full[(~train_mask) & valid]
        train_diag[ticker] = {"n": int(len(train_r)), "mean": float(np.mean(train_r)) if len(train_r) else None,
                               "std": float(np.std(train_r)) if len(train_r) else None}
        test_diag[ticker] = {"n": int(len(test_r)), "mean": float(np.mean(test_r)) if len(test_r) else None,
                              "std": float(np.std(test_r)) if len(test_r) else None}

        torch.save({"mu_net": model.mu_net.state_dict(), "sigma_net": model.sigma_net.state_dict()},
                   os.path.join(out_dir, f"{ticker}_net.pt"))
        print(f"  {ticker}: train resid mean/std = {train_diag[ticker]['mean']:.4f}/{train_diag[ticker]['std']:.4f}"
              f"  test resid mean/std = {test_diag[ticker]['mean']}/{test_diag[ticker]['std']}")

    resid_df = pd.DataFrame(residuals, index=trading_dates)
    resid_df.to_csv(os.path.join(out_dir, "residuals.csv"), index_label="Date")

    meta = {
        "method": method, "hparams": HPARAMS, "n_epochs_cap": n_epochs, "batch_size": batch_size,
        "m_normalization": {"mu": m_mu, "sigma": m_sigma},
        "train_end_date": str(train_end_date.date()),
        "train_diagnostics": train_diag, "test_diagnostics": test_diag,
        "note": "train_diagnostics and test_diagnostics are reported separately -- never pooled "
                "into one apparent out-of-sample number. mu_net/sigma_net fit on train dates only; "
                "test residuals use those same frozen networks and the same frozen m-normalization "
                "stats (no refit, no renormalization on test).",
    }
    with open(os.path.join(out_dir, "standardizer_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Saved -> {out_dir}/residuals.csv (+ standardizer_meta.json, {len(residuals)} nets)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default="s0,s1,s2,s3,s4")
    parser.add_argument("--assets", type=str, default=None, help="comma-separated tickers; default = all 15")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    cfg = common.get_default_config()
    manifest = common.get_window_manifest(cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    assets = [a.strip() for a in args.assets.split(",")] if args.assets else None

    for m in methods:
        print(f"\n=== Standardizing {m} ({common.METHOD_LABELS[m]}) ===")
        run_method(m, manifest, cfg, args.epochs, args.batch_size, assets)

    print("\nDone.")


if __name__ == "__main__":
    main()
