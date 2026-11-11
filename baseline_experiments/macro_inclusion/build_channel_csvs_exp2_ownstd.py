"""
Macro embedding (S0-S4), rare-event branch: builds the per-method channel files for the conditioning set selected
in config (config.data.conditioning_set) and verifies them. Runs after daily_macro_states.py and the
standardize_own_macro.py jobs; exits non-zero on any failed check.

<experiment folder>/extras/rare_event_channels/data_for_diffusion_<channel>_ownstd.csv:
    column 0      = that method's RAW daily state (<experiment>/extras/exp2_results/<method>/daily_state.csv), with
                    ONE shared NaN mask across the channels built together (identical window set for every channel)
    asset columns = z^(k), the 15 stocks' residuals standardized on that method's OWN state
                    (<experiment>/extras/standardization_own/<method>/residuals.csv)
Calendar = the conditioning set's data_for_diffusion.csv dates (config.data.csv_path) on which the residuals exist
(standardize_own_macro.py fits on the dates where all five base states S0-S4 are defined).

Checks: residuals finite on the calendar; mu + sigma * z == the set's raw log returns; S1-S4 residuals differ from
S0's; DataProcessor window -> date alignment on every channel.

Usage (set the conditioning set in config/config.py first):
    python build_channel_csvs_exp2_ownstd.py [--channels exp2_s0,exp2_s1p,exp2_s2,exp2_s3p,exp2_s4]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from config import get_default_config  # noqa: E402
from save_rare_event_artifacts import build_processor  # noqa: E402
from window_conditioning import experiment_paths  # noqa: E402

METHOD_DIRNAMES = {"exp2_s0": "S0_state_space", "exp2_s1": "S1_linear", "exp2_s2": "S2_forward_fill",
                   "exp2_s3": "S3_cubic_spline", "exp2_s4": "S4_tracking_regression",
                   "exp2_s1p": "S1p_linear_extrapolation", "exp2_s3p": "S3p_cubic_spline_extrapolation"}
DEFAULT = ["exp2_s0", "exp2_s1", "exp2_s2", "exp2_s3", "exp2_s4"]


def fail(msg):
    print(f"[FAIL] {msg}", flush=True)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", default=",".join(DEFAULT), help="channels to build (S0 is always loaded for the reuse check)")
    args = p.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    for c in channels:
        if c not in METHOD_DIRNAMES:
            fail(f"unknown channel {c!r}; expected one of {list(METHOD_DIRNAMES)}")
    cfg = get_default_config()
    paths = experiment_paths()
    out_dir = paths["embed_channels_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"Conditioning set {paths['conditioning_set']!r}: calendar {cfg.data.csv_path}\n  states {paths['states_dir']}\n"
          f"  residuals {paths['standardization_dir']}\n  out {out_dir}", flush=True)

    prod = pd.read_csv(cfg.data.csv_path, parse_dates=["Date"]).set_index("Date")
    assets = list(prod.columns[1:])
    raw_all = pd.read_csv(cfg.data.macro_csv_path, parse_dates=["Date"]).set_index("Date")[assets]
    resid, states = {}, {}
    for ch in dict.fromkeys(["exp2_s0"] + channels):
        d = os.path.join(paths["standardization_dir"], METHOD_DIRNAMES[ch])
        if not os.path.exists(os.path.join(d, "standardizer.pt")):
            fail(f"{ch}: no fitted standardizer in {d} (run standardize_own_macro.py --method {ch[5:]})")
        meta = json.load(open(os.path.join(d, "meta.json")))
        if meta.get("conditioning_set") != paths["conditioning_set"]:
            fail(f"{ch}: standardizer in {d} was fit for conditioning set {meta.get('conditioning_set')!r}")
        resid[ch] = tuple(pd.read_csv(os.path.join(d, f"{k}.csv"), parse_dates=["Date"]).set_index("Date")[assets]
                          for k in ("residuals", "mu", "sigma"))
        states[ch] = pd.read_csv(os.path.join(paths["states_dir"], METHOD_DIRNAMES[ch], "daily_state.csv"),
                                 index_col=0, parse_dates=True)["daily_state"]

    # calendar: the set's dates on which every channel's residuals exist (same fit dates for all methods)
    ok = np.ones(len(prod), dtype=bool)
    for ch in channels:
        ok &= np.isfinite(resid[ch][0].reindex(prod.index).to_numpy()).all(axis=1)
    cal = prod.index[ok]
    print(f"calendar: {len(cal)} of {len(prod)} set dates have residuals for every channel "
          f"({cal[0].date()}..{cal[-1].date()})", flush=True)
    raw = raw_all.reindex(cal)

    for ch in channels:
        z, mu, sg = (x.reindex(cal) for x in resid[ch])
        err = float(np.nanmax(np.abs((mu + sg * z - raw).to_numpy())))
        if err > 1e-5:
            fail(f"{ch}: mu + sigma * z != raw log returns (max err {err})")
        print(f"{ch}: inverse check {err:.1e}", flush=True)
        if ch != "exp2_s0":
            dmax = float(np.nanmax(np.abs((z - resid["exp2_s0"][0].reindex(cal)).to_numpy())))
            if dmax < 1e-2:
                fail(f"{ch}: residuals are (near-)identical to S0's (max diff {dmax}) -- S0 residuals reused?")
            print(f"{ch}: differs from S0 residuals (max |diff| {dmax:.3f})")

    shared_nan = pd.concat([states[ch].reindex(cal) for ch in channels], axis=1).isna().any(axis=1).to_numpy()
    print(f"shared NaN mask on column 0: {shared_nan.sum()}/{len(cal)} dates (applied to every channel)", flush=True)
    T = cfg.data.seq_len
    for ch in channels:
        out = resid[ch][0].reindex(cal).copy()
        out.insert(0, ch, states[ch].reindex(cal).mask(shared_nan).to_numpy())
        path = os.path.join(out_dir, f"data_for_diffusion_{ch}_ownstd.csv")
        out.to_csv(path, index_label="Date")
        dp = build_processor(path, cfg)
        X = dp.destandardize_windows(np.transpose(dp.X_train.numpy(), (0, 2, 1)), dp.mu_entry_train, dp.sig_entry_train)
        vals = dp.df[cfg.data.tickers].to_numpy(np.float64)
        al = float(np.max(np.abs(X - np.stack([vals[j:j + T].T for j in range(len(X))]))))
        if al > 1e-4 or not np.isfinite(X).all():
            fail(f"{ch}: DataProcessor alignment/finite check failed (max err {al})")
        print(f"{ch}: wrote {path} -- DataProcessor {len(dp.df)} rows, {len(X)} train windows, alignment err {al:.1e}",
              flush=True)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
