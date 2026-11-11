"""
Full-horizon NN standardization of the 15 stocks' raw returns on ONE macro-embedding method's daily state (S1-S4),
for the conditioning set selected in config (or CDG_CONDITIONING_SET). Fitted ONCE and saved; refuses to overwrite.

Same procedure as the set's own S0 residuals in explore/conditioning_sets/<set>/data_for_diffusion.csv
(explore/build_conditioning_sets.py): production explore/nn_standardize.run_ticker, default hyperparameters,
seed = config seed, fitted per stock on every date where the method's state and the stock's return exist (the
method's own span) -- so S0 (explore) and S1-S4 (here) are standardized identically, differing only in the state.

Inputs : raw returns  config.data.macro_csv_path (explore/conditioning_sets/<set>/macro_data_new.csv)
         state        <experiment>/extras/exp2_results/<method>/daily_state.csv (daily_macro_states.py)
Output : <experiment>/extras/standardization_fullhorizon/<method>/residuals.csv, meta.json

Usage:  python standardize_fullhorizon_set.py --method s1
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import common
from window_conditioning import experiment_paths

sys.path.insert(0, os.path.join(common._ROOT, "explore"))
from nn_standardize import run_ticker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["s1", "s2", "s3", "s4"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = common.get_default_config()
    paths = experiment_paths()
    mdir = common.METHOD_DIRNAMES[a.method]
    out = os.path.join(paths["experiment_dir"], "extras", "standardization_fullhorizon", mdir)
    if os.path.exists(os.path.join(out, "residuals.csv")):
        raise SystemExit(f"{out}/residuals.csv exists -- fitted once already, refusing to overwrite")
    state_csv = os.path.join(paths["states_dir"], mdir, "daily_state.csv")
    raw = pd.read_csv(cfg.data.macro_csv_path, parse_dates=["Date"]).set_index("Date")
    state = pd.read_csv(state_csv, index_col=0, parse_dates=True)["daily_state"]
    dates = state.index.intersection(raw.index)
    state, raw = state.reindex(dates), raw.reindex(dates)
    print(f"conditioning_set = {cfg.data.conditioning_set} | {a.method} state {state_csv} | returns {cfg.data.macro_csv_path} "
          f"| seed {cfg.seed}", flush=True)

    resid = {}
    for t in common.TICKERS:
        r_np = raw[t].to_numpy(dtype=np.float32)
        m_np = state.to_numpy(dtype=np.float32)
        ok = np.isfinite(r_np) & np.isfinite(m_np)
        _, z, _, _ = run_ticker(torch.tensor(r_np[ok]), torch.tensor(m_np[ok]).reshape(-1, 1), f"{a.method}_{t}",
                                seed=cfg.seed, verbose=False)
        col = np.full(len(dates), np.nan, dtype=np.float32)
        col[ok] = z.detach().numpy()
        resid[t] = col
        print(f"  {t}: n={int(ok.sum())}", flush=True)
    os.makedirs(out, exist_ok=True)
    R = pd.DataFrame(resid, index=dates).dropna(how="all")
    R.to_csv(os.path.join(out, "residuals.csv"), index_label="Date")
    json.dump({"method": a.method, "conditioning_set": cfg.data.conditioning_set, "state_file": state_csv,
               "returns_file": cfg.data.macro_csv_path, "seed": cfg.seed, "fit": "full horizon, own span (run_ticker defaults)",
               "span": [str(R.index[0].date()), str(R.index[-1].date())], "n_days": int(len(R)),
               "built": datetime.now().isoformat(timespec="seconds")}, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"wrote {out}/residuals.csv ({len(R)} days, {R.index[0].date()}..{R.index[-1].date()})")


if __name__ == "__main__":
    main()
