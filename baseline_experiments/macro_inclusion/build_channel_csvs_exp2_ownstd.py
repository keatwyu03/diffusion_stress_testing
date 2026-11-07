"""
Macro embedding, rare-event branch: builds the 5 per-method channel CSVs with
each method's OWN NN residuals, and verifies them. Runs after the five
standardize_own_macro.py jobs; exits non-zero on any failed check, so the
dependent Slurm jobs (afterok) never start on bad inputs.

data_for_diffusion_exp2_s{k}_ownstd.csv:
    column 0     = S_k raw daily state, shared NaN mask (identical to the
                   existing data_for_diffusion_exp2_s{k}.csv -> same events)
    asset columns = z^(k), residuals standardized on S_k's own state
                   (NOT the production S0 residuals)

Checks: column 0 unchanged; residuals finite on every production-calendar
date up to config end_date; mu + sigma * z == raw log returns; S1-S4 residuals
differ from S0's; DataProcessor window -> date alignment on every channel.

Usage:
    python build_channel_csvs_exp2_ownstd.py --standardization-dir DIR --copy-to COPY1 COPY2 ...
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from config import get_default_config  # noqa: E402
from save_rare_event_artifacts import build_processor  # noqa: E402

SR = os.path.join(_ROOT, "explore", "standardization_results")
PROD_CSV = os.path.join(SR, "data_for_diffusion.csv")
METHOD_DIRNAMES = {"exp2_s0": "S0_state_space", "exp2_s1": "S1_linear", "exp2_s2": "S2_forward_fill",
                   "exp2_s3": "S3_cubic_spline", "exp2_s4": "S4_tracking_regression"}


def fail(msg):
    print(f"[FAIL] {msg}", flush=True)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--standardization-dir", required=True)
    p.add_argument("--copy-to", nargs="*", default=[], help="run copies to place the CSVs in (explore/standardization_results/)")
    args = p.parse_args()
    cfg = get_default_config()
    end = pd.Timestamp(cfg.data.end_date)

    prod = pd.read_csv(PROD_CSV, parse_dates=["Date"]).set_index("Date")
    assets = list(prod.columns[1:])
    raw = pd.read_csv(os.path.join(_ROOT, "explore", "macro_data_new.csv"), parse_dates=["Date"]).set_index("Date")
    raw = raw[assets].reindex(prod.index)
    in_range = (prod.index <= end)
    resid = {}
    for ch, mdir in METHOD_DIRNAMES.items():
        d = os.path.join(args.standardization_dir, mdir)
        meta = json.load(open(os.path.join(d, "meta.json")))
        if not os.path.exists(os.path.join(d, "standardizer.pt")):
            fail(f"{ch}: no fitted standardizer in {d}")
        if not meta["macro_series"].endswith(os.path.join(mdir, "daily_state.csv")):
            fail(f"{ch}: standardizer was fit on {meta['macro_series']}, not {mdir}")
        z, mu, sg = (pd.read_csv(os.path.join(d, f"{k}.csv"), parse_dates=["Date"]).set_index("Date")[assets].reindex(prod.index)
                     for k in ("residuals", "mu", "sigma"))
        if not np.isfinite(z[in_range].to_numpy()).all():
            fail(f"{ch}: residuals not finite on every production date <= {end.date()}")
        err = float(np.nanmax(np.abs((mu + sg * z - raw)[in_range].to_numpy())))
        if err > 1e-5:
            fail(f"{ch}: mu + sigma * z != raw log returns (max err {err})")
        resid[ch] = z
        print(f"{ch}: standardizer {d} (fit on {meta['macro_series'].split('/')[-2]}), inverse check {err:.1e}", flush=True)
    for ch in list(METHOD_DIRNAMES)[1:]:
        dmax = float(np.nanmax(np.abs((resid[ch] - resid["exp2_s0"])[in_range].to_numpy())))
        if dmax < 1e-2:
            fail(f"{ch}: residuals are (near-)identical to S0's (max diff {dmax}) -- S0 residuals reused?")
        print(f"{ch}: differs from S0 residuals (max |diff| {dmax:.3f})")

    for ch in METHOD_DIRNAMES:
        old = pd.read_csv(os.path.join(SR, f"data_for_diffusion_{ch}.csv"), parse_dates=["Date"]).set_index("Date")
        if not old.index.equals(prod.index):
            fail(f"{ch}: existing channel CSV calendar differs from production")
        out = resid[ch].copy()
        out.insert(0, ch, old.iloc[:, 0].to_numpy())  # same event series + shared NaN mask as before
        path = os.path.join(SR, f"data_for_diffusion_{ch}_ownstd.csv")
        out.to_csv(path, index_label="Date")
        dp = build_processor(path, cfg)
        T = cfg.data.seq_len
        X = dp.destandardize_windows(np.transpose(dp.X_train.numpy(), (0, 2, 1)), dp.mu_entry_train, dp.sig_entry_train)
        vals = dp.df[cfg.data.tickers].to_numpy(np.float64)
        al = float(np.max(np.abs(X - np.stack([vals[j:j + T].T for j in range(len(X))]))))
        if al > 1e-4 or not np.isfinite(X).all():
            fail(f"{ch}: DataProcessor alignment/finite check failed (max err {al})")
        print(f"{ch}: wrote {path} -- DataProcessor {len(dp.df)} rows, {len(X)} train windows, alignment err {al:.1e}", flush=True)
        for c in args.copy_to:
            dst = os.path.join(c, "explore", "standardization_results")
            shutil.copy2(path, dst)
            print(f"   -> {dst}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
