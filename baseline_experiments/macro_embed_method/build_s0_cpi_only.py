"""
One-off script: genuine CPI-only S0 (state-space) fit. daily_macro_states.py's
build_s0() deliberately reuses the cached, already-fitted Full-macro-set m_t
from explore/macro_data_new.csv (documented there as an intentional
cost-saving shortcut) -- that shortcut is silently wrong for any macro
variable selection other than the one m_t was originally fit under. This
script does the real fresh fit via LatentStateEstimator(method="state_space",
...), same mechanism daily_macro_states.py uses for S4's tracking-regression
fit, just with the state_space method instead, under growth_vars=None,
inflation_vars=["cpi"].

Does NOT modify config.py -- passes growth_vars/inflation_vars directly as
constructor args, same pattern build_s4()/build_interpolated() already use.
"""
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LATENT_DIR = os.path.join(_ROOT, "latent_state_estimation")
sys.path.insert(0, _LATENT_DIR)
sys.path.insert(0, _ROOT)
from macro_main import LatentStateEstimator  # noqa: E402

import common  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(_HERE), "experiment_3 (CPI Only)", "exp2_results_cpi_only", "S0_state_space")


def main():
    cfg = common.get_default_config()
    manifest = common.get_window_manifest(cfg)

    print("Fitting LatentStateEstimator(method='state_space', growth_vars=None, inflation_vars=['cpi']) ...")
    est = LatentStateEstimator(method="state_space", data_dir=_LATENT_DIR,
                                growth_vars=None, inflation_vars=["cpi"], vol_vars=None)
    latent = est.fit()
    print(f"Fit complete. latent: {len(latent)} rows, {latent.index.min()} to {latent.index.max()}")

    series = latent.reindex(manifest["df"].index)
    n_nan = int(series.isna().sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    series.rename("daily_state").to_csv(os.path.join(OUT_DIR, "daily_state.csv"), index_label="Date")
    meta = {
        "method": "s0", "source": "FRESH LatentStateEstimator(method='state_space', "
                                   "growth_vars=None, inflation_vars=['cpi']).fit() -- genuine CPI-only refit, "
                                   "NOT the cached Full-macro-set m_t",
        "uses_daily_indicators": True,
        "full_sample_fit_preprocessing": ["SSM dynamics MLE (Nelder-Mead, full sample)",
                                           "SSM daily-indicator z-scoring (full sample)",
                                           "monthly_first_pc PCA/scaler (full sample)"],
        "future_interpolation_anchors": False,
        "retrospective": False,
        "n_dates": len(series), "n_nan": n_nan,
        "date_range": [str(series.index.min().date()), str(series.index.max().date())],
        "train_end_date": str(manifest["train_end_date"].date()),
        "note": "Causal in filtering; parameters and preprocessing are full-sample fit "
                "(disclosed limitation, not re-architected this pass).",
    }
    with open(os.path.join(OUT_DIR, "state_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved -> {OUT_DIR}/daily_state.csv (+ state_meta.json), {n_nan}/{len(series)} NaN")


if __name__ == "__main__":
    main()
