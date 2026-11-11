"""
Full-horizon NN standardization for the two NEW methods S1' (s1p) and S3' (s3p) ONLY -- the exact procedure of
fullhorizon_standardize.py (which produced the residuals behind the existing S0-S4 stationarity table), copied
line for line so that file stays untouched:

    production explore/nn_standardize.py run_ticker(), unmodified, default hyperparameters, seed 0;
    fit on the ENTIRE horizon at once; conditioning m = the method's own daily_state;
    dates = state dates intersected with explore/macro_data_new.csv; per asset, rows where both the
    return and the state are finite.

Reads   experiment_1 (Full)/extras/exp2_results/<method>/daily_state.csv
Writes  experiment_1 (Full)/extras/exp2_results/<method>/standardizer_fullhorizon/residuals.csv  (new folders only)

Usage:  python fullhorizon_standardize_new_methods.py --method s1p
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import common

sys.path.insert(0, os.path.join(common._ROOT, "explore"))
from nn_standardize import run_ticker  # noqa: E402

STATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_1 (Full)", "extras", "exp2_results")
ORDER = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "TXN", "QCOM", "AMAT", "ADBE",
         "KO", "PG", "CL", "GIS", "KMB"]   # identical to fullhorizon_standardize.ORDER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["s1p", "s3p"])
    args = p.parse_args()
    method = args.method

    raw_returns_df = pd.read_csv(
        os.path.join(common._ROOT, "explore", "macro_data_new.csv"), parse_dates=["Date"]
    ).set_index("Date")

    method_dir = common.METHOD_DIRNAMES[method]
    out_dir = os.path.join(STATES_DIR, method_dir, "standardizer_fullhorizon")
    if os.path.exists(os.path.join(out_dir, "residuals.csv")):
        raise FileExistsError(f"{out_dir}/residuals.csv exists -- refusing to overwrite")
    state = pd.read_csv(os.path.join(STATES_DIR, method_dir, "daily_state.csv"),
                        index_col=0, parse_dates=True)["daily_state"]
    common_dates = state.index.intersection(raw_returns_df.index)
    state = state.reindex(common_dates)
    returns = raw_returns_df.reindex(common_dates)

    residuals = {}
    for ticker in ORDER:
        r_np = returns[ticker].to_numpy(dtype=np.float32)
        m_np = state.to_numpy(dtype=np.float32)
        valid = np.isfinite(r_np) & np.isfinite(m_np)
        r_t = torch.tensor(r_np[valid], dtype=torch.float32)
        m_t = torch.tensor(m_np[valid], dtype=torch.float32).reshape(-1, 1)

        print(f"[{method}] fitting {ticker} (n={valid.sum()}) ...", flush=True)
        _, z, _, _ = run_ticker(r_t, m_t, f"{method}_{ticker}", verbose=False)
        resid_full = np.full(len(common_dates), np.nan, dtype=np.float32)
        resid_full[valid] = z.detach().numpy()
        residuals[ticker] = resid_full

    os.makedirs(out_dir, exist_ok=True)
    resid_df = pd.DataFrame(residuals, index=common_dates)
    resid_df.to_csv(os.path.join(out_dir, "residuals.csv"), index_label="Date")
    print(f"{method}: wrote {out_dir}/residuals.csv ({len(common_dates)} rows)", flush=True)


if __name__ == "__main__":
    main()
