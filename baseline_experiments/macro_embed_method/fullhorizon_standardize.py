"""
Full-horizon macro-conditional standardization for all 5 Experiment-2 methods
(S0-S4), using the EXISTING production standardizer
(explore/nn_standardize.py::run_ticker()) completely unmodified -- fits
mu_net/sigma_net on the ENTIRE 2000-2025 horizon at once (no train/test
split), exactly matching the methodology that produced the user's earlier
Table 4 (nn_standardized_macro.csv), just with each of S0-S4's daily_state
as the conditioning series `m` instead of the single production m_t.

This is a separate standardizer output from macro_standardization.py's
train-only-fit version (kept, not overwritten) -- written to
<method_dir>/standardizer_fullhorizon/residuals.csv so both are available
for comparison.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import common

sys.path.insert(0, os.path.join(common._ROOT, "explore"))
from nn_standardize import run_ticker  # noqa: E402

OUT_DIR = os.path.join(common._RESULTS_ROOT, "exp2_results")
ORDER = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "TXN", "QCOM", "AMAT", "ADBE",
         "KO", "PG", "CL", "GIS", "KMB"]


def main():
    raw_returns_df = pd.read_csv(
        os.path.join(common._ROOT, "explore", "macro_data_new.csv"), parse_dates=["Date"]
    ).set_index("Date")

    for method in ["s0", "s1", "s2", "s3", "s4"]:
        method_dir = common.METHOD_DIRNAMES[method]
        state = pd.read_csv(os.path.join(OUT_DIR, method_dir, "daily_state.csv"),
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

        out_dir = os.path.join(OUT_DIR, method_dir, "standardizer_fullhorizon")
        os.makedirs(out_dir, exist_ok=True)
        resid_df = pd.DataFrame(residuals, index=common_dates)
        resid_df.to_csv(os.path.join(out_dir, "residuals.csv"), index_label="Date")
        print(f"{method}: wrote {out_dir}/residuals.csv ({len(common_dates)} rows)", flush=True)


if __name__ == "__main__":
    main()
