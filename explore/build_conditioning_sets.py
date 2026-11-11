"""
Build each macro conditioning set ONCE on the locked data and save it, so the state-space model and the NN
standardizer never have to be refit for that set again.

    explore/conditioning_sets/<set>/
        m_t.csv                   fitted macro state (LatentStateEstimator, config latent_method / accumulator)
        macro_data_new.csv        m_t + raw daily log returns        (same format as explore/macro_data_new.csv)
        data_for_diffusion.csv    m_t + NN residuals, seed=config    (same format as config.data.csv_path)
        meta.json                 variables, method, window, data snapshot, seed, anchor fit, build time
        real_corr_event_vs_all_top{7.5,5}pct_lag<gap>.png   causal event-window correlation check (config lag)

Steps and code are those of explore/import_data.py [1/5]-[5/5]; inputs are the LOCKED macro panels (already
restored into latent_state_estimation/, verified here against the frozen snapshot) and the frozen returns;
the window, tickers, seed, method and accumulator come from config. Only the variable selection differs per set.
A set that already exists is never rebuilt (pass --force to rebuild it).

Usage:  python explore/build_conditioning_sets.py --set full|cpi_only|inflation_only|growth_only
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from config import get_default_config  # noqa: E402
from nn_standardize import run_ticker  # noqa: E402
from latent_state_estimation.macro_main import LatentStateEstimator  # noqa: E402
from latent_state_estimation.frozen_macro import MONTHLY, DAILY  # noqa: E402

GROWTH = ["indpro", "payems", "pi_transfer", "real_manf_trade", "personal_consump", "capacity_util"]
INFLATION = ["cpi", "oil_price", "ppi", "hour_earnings"]
SETS = {
    "full":           {"growth_vars": GROWTH, "inflation_vars": INFLATION, "vol_vars": None},
    "cpi_only":       {"growth_vars": None,   "inflation_vars": ["cpi"],   "vol_vars": None},
    "inflation_only": {"growth_vars": None,   "inflation_vars": INFLATION, "vol_vars": None},
    "growth_only":    {"growth_vars": GROWTH, "inflation_vars": None,      "vol_vars": None},
}
OUT_ROOT = os.path.join(_HERE, "conditioning_sets")
# Locked Table 4 inputs (git a6c4f3f). Nothing is ever downloaded: macro panels + stock returns come only from here.
FROZEN_MACRO_SNAPSHOT = "latent_state_estimation/frozen_table4_a6c4f3f"
FROZEN_RETURNS_CSV = "latent_state_estimation/frozen_table4_a6c4f3f/stock_log_returns.csv"
COND = "m_t"


def check_live_panels(cfg):
    """The estimator reads latent_state_estimation/*.csv: they must be the locked Table 4 panels
    (monthly byte-identical, daily = snapshot rows inside the config window)."""
    snap = os.path.join(_ROOT, FROZEN_MACRO_SNAPSHOT)
    live = os.path.join(_ROOT, "latent_state_estimation")
    lo, hi = str(cfg.data.start_date)[:10], str(cfg.data.end_date)[:10]
    for p in MONTHLY:
        if open(os.path.join(snap, f"{p}.csv"), "rb").read() != open(os.path.join(live, f"{p}.csv"), "rb").read():
            raise SystemExit(f"live {p}.csv differs from the frozen snapshot -- restore them with: python -c \"from latent_state_estimation.frozen_macro import restore_frozen_panels; restore_frozen_panels('latent_state_estimation/frozen_table4_a6c4f3f', '2000-09-25', '2026-08-20')\"")
    for p in DAILY:
        lines = open(os.path.join(snap, f"{p}.csv")).readlines()
        want = [lines[0]] + [l for l in lines[1:] if lo <= l[:10] <= hi]
        if open(os.path.join(live, f"{p}.csv")).readlines() != want:
            raise SystemExit(f"live {p}.csv is not the windowed frozen snapshot -- restore them with: python -c \"from latent_state_estimation.frozen_macro import restore_frozen_panels; restore_frozen_panels('latent_state_estimation/frozen_table4_a6c4f3f', '2000-09-25', '2026-08-20')\"")
    print("[check] live macro panels == locked Table 4 snapshot (daily cut to the config window)")


def plot_corr(cfg, csv_path, out_png, threshold):
    """Exact plotting block of import_data.py (causal event windows, colours only), saved to out_png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (used by the exec'd block)
    from data import DataProcessor  # noqa: F401
    cfg.hfunction.event_threshold = threshold
    tickers, cond_event = cfg.data.tickers, COND  # noqa: F841
    bucket = {g: list(s) for g, s in (("growth", cfg.data.growth_vars), ("inflation", cfg.data.inflation_vars),
                                      ("vol", cfg.data.vol_vars)) if s}
    bucket_lbl = ";  ".join(f"{g} using {', '.join(cols)}" for g, cols in bucket.items())  # noqa: F841
    src = open(os.path.join(_HERE, "import_data.py")).read()
    blk = src[src.index("dp = DataProcessor("):src.index("# Quick visual sanity check")]
    old = 'os.path.join(os.path.dirname(csv_path), "real_corr_event_vs_all.png")'
    assert blk.count(old) == 1, "import_data.py plot block changed -- update build_conditioning_sets.plot_corr"
    blk = blk.replace(old, repr(out_png))
    env = dict(globals(), _cfg=cfg, csv_path=csv_path, tickers=tickers, cond_event=cond_event, bucket_lbl=bucket_lbl,
               np=np, pd=pd, plt=plt, DataProcessor=DataProcessor, os=os)
    exec(blk, env)
    T, S = slice(0, 10), slice(10, 15)
    off = lambda M: float(M[~np.eye(len(M), dtype=bool)].mean())  # noqa: E731
    return {lbl: {"n": len(arr), "overall": off(np.corrcoef(arr.T)), "tech": off(np.corrcoef(arr.T)[T, T]),
                  "staples": off(np.corrcoef(arr.T)[S, S]), "cross": float(np.corrcoef(arr.T)[T, S].mean())}
            for lbl, arr in env["panels"]}


def build(name, force=False):
    cfg = get_default_config()
    for k, v in SETS[name].items():
        setattr(cfg.data, k, v)
    out_dir = os.path.join(OUT_ROOT, name)
    if os.path.exists(out_dir) and not force:
        raise SystemExit(f"{out_dir} already exists -- built once already (use --force to rebuild)")
    if not cfg.data.event_causal:
        raise SystemExit("config.data.event_causal must be True (causal-only)")
    check_live_panels(cfg)
    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    print(f"[1/5] {name}: estimating macro state ({cfg.data.latent_method}, accumulator={cfg.data.latent_accumulator}) "
          f"growth={cfg.data.growth_vars} inflation={cfg.data.inflation_vars} vol={cfg.data.vol_vars}")
    est = LatentStateEstimator(method=cfg.data.latent_method, growth_vars=cfg.data.growth_vars,
                               inflation_vars=cfg.data.inflation_vars, vol_vars=cfg.data.vol_vars,
                               accumulator=cfg.data.latent_accumulator)
    cond_series = est.fit()
    cond_series.rename(COND).to_csv(os.path.join(tmp, "m_t.csv"), index_label="Date")

    tickers = cfg.data.tickers
    win0, win1 = pd.Timestamp(cfg.data.start_date), pd.Timestamp(cfg.data.end_date)
    frozen = pd.read_csv(os.path.join(_ROOT, FROZEN_RETURNS_CSV), parse_dates=["Date"]).set_index("Date")
    log_ret = frozen[tickers].dropna().loc[win0:win1]
    df_out = pd.DataFrame({COND: cond_series.reindex(log_ret.index)})
    for t in tickers:
        df_out[t] = log_ret[t]
    df_out = df_out.dropna(subset=tickers)
    raw_csv = os.path.join(tmp, "macro_data_new.csv")
    df_out.to_csv(raw_csv, index_label="Date")
    print(f"[4/5] macro_data_new.csv: {len(df_out)} rows, m_t valid {int(df_out[COND].notna().sum())}")

    diff_df = df_out[df_out[COND].notna()].copy()
    m = torch.tensor(diff_df[[COND]].to_numpy(), dtype=torch.float32)
    out = pd.DataFrame(index=diff_df.index)
    for t in tickers:
        r = torch.tensor(diff_df[t].to_numpy(), dtype=torch.float32)
        _, z, _, _ = run_ticker(r, m, t, seed=cfg.seed, verbose=False)
        out[t] = z.detach().cpu().numpy()
        print(f"[5/5] {name}: {t} done", flush=True)
    out.insert(0, COND, diff_df[COND].to_numpy())
    out.index.name = "Date"
    out.to_csv(os.path.join(tmp, "data_for_diffusion.csv"))

    corr = {}
    for thr in (0.075, 0.05):
        tag = f"top{thr * 100:g}pct_lag{cfg.data.event_lag_gap}"
        corr[tag] = plot_corr(cfg, raw_csv, os.path.join(tmp, f"real_corr_event_vs_all_{tag}.png"), thr)

    meta = {"set": name, "growth_vars": cfg.data.growth_vars, "inflation_vars": cfg.data.inflation_vars,
            "vol_vars": cfg.data.vol_vars, "latent_method": cfg.data.latent_method,
            "latent_accumulator": cfg.data.latent_accumulator, "window": [cfg.data.start_date, cfg.data.end_date],
            "macro_snapshot": FROZEN_MACRO_SNAPSHOT, "returns": FROZEN_RETURNS_CSV,
            "tickers": list(tickers), "seed": cfg.seed, "event_causal": cfg.data.event_causal,
            "m_t_range": [str(cond_series.dropna().index[0].date()), str(cond_series.dropna().index[-1].date())],
            "m_t_days": int(diff_df.shape[0]), "rows_macro_data_new": int(len(df_out)),
            "anchor_rmse": {k: float(v) for k, v in est.anchor_rmse.items()},
            "anchor_r2": {k: float(v) for k, v in est.anchor_r2.items()},
            "corr_event_vs_all": corr, "built": datetime.now().isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=2)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)
    print(f"DONE {name} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.set, a.force)
