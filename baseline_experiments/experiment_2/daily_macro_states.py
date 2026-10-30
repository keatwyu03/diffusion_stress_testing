"""
Builds the 5 daily macro-state series (S0-S4) for Experiment 2, all on the
SAME shared trading calendar (common.py), all starting from the SAME monthly
growth/inflation PCA factors (monthly_first_pc, shared with the production
SSM/tracking-regression code -- not recomputed independently per baseline).

S0 (state-space, proposed method): reuses the EXISTING, already-fitted m_t
column from explore/macro_data_new.csv directly -- does NOT call
StateSpace.fit() (confirmed expensive: Nelder-Mead maxiter=20000,
maxfev=40000, state_space.py:176-185). This project's established convention
already treats that column as the production SSM output; refitting it here
would just reproduce the same thing at real cost, and do it during a
"smoke test" of this experiment, exactly what the approved plan prohibits.

S4 (tracking regression): LatentStateEstimator(method="tracking_regression")
is used only to get each active group's fitted TrackingRegression (full-
sample OLS betas -- confirmed existing limitation, same category as S0's
full-sample MLE, not re-architected here). Its own self.latent aggregates
groups with FULL-SAMPLE mean/std (confirmed, macro_main.py:93-95) -- bypassed
here in favor of extracting each group's raw .ut series and re-aggregating
with TRAIN-ONLY stats (common.train_only_zscore), per the approved plan.

S1/S2/S3 (linear / forward-fill / cubic spline): built from the SAME
monthly_first_pc factors used by S0/S4, anchored at each month's last
TRADING day (common.month_end_anchor -- matches state_space.py's own
placement convention exactly, so these baselines don't "know" a monthly
value weeks before S0 does), then interpolated/held/splined onto the full
trading calendar, then z-scored (train-only) + averaged across groups --
the SAME aggregation convention S4 uses, per the plan's explicit instruction
not to invent a new combination rule.

Honest asymmetry, disclosed per method in the saved causality record: S0/S4
also see raw DAILY financial indicators (S0 through its state vector, S4
through the tracking regression's daily beta); S1/S2/S3 see only the
monthly factors. This is a comparison of daily macro-representation
pipelines, not an isolated apples-to-apples filling-algorithm comparison --
see each method's "uses_daily_indicators" flag in its saved metadata.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import common

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LATENT_DIR = os.path.join(_ROOT, "latent_state_estimation")
sys.path.insert(0, _LATENT_DIR)
from macro_main import LatentStateEstimator  # noqa: E402
from tracking_regression import monthly_first_pc  # noqa: E402

OUT_DIR = os.path.join(_HERE, "exp2_results")


def active_groups(cfg):
    sel = {"growth": cfg.data.growth_vars, "inflation": cfg.data.inflation_vars, "vol": cfg.data.vol_vars}
    return [n for n in ("growth", "inflation", "vol") if sel[n]], sel


def load_monthly_panel(name: str, cols) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_LATENT_DIR, f"{name}_macro.csv"), index_col=0, parse_dates=True)
    return df[list(cols)]


def aggregate_groups(per_group_daily: dict, train_end_date: pd.Timestamp):
    """z-score each group's daily series on TRAIN dates only, then average
    across groups -- the one shared, documented combination rule (S4's own
    convention) used by every method here, per the approved plan."""
    zscored = {}
    stats = {}
    for name, series in per_group_daily.items():
        z, mu, sigma = common.train_only_zscore(series, train_end_date)
        zscored[name] = z
        stats[name] = {"mu": mu, "sigma": sigma}
    combined = pd.concat(zscored, axis=1).mean(axis=1)
    return combined, stats


def build_s0(manifest) -> tuple:
    macro_csv = os.path.join(_ROOT, "explore", "macro_data_new.csv")
    df = pd.read_csv(macro_csv, parse_dates=["Date"]).set_index("Date")
    macro_col = df.columns[0]
    series = df[macro_col].reindex(manifest["df"].index)
    meta = {
        "method": "s0", "source": "cached m_t from explore/macro_data_new.csv (NOT refit)",
        "uses_daily_indicators": True,
        "full_sample_fit_preprocessing": ["SSM dynamics MLE (Nelder-Mead, full sample)",
                                           "SSM daily-indicator z-scoring (full sample)",
                                           "monthly_first_pc PCA/scaler (full sample)"],
        "future_interpolation_anchors": False,
        "retrospective": False,
        "note": "Causal in filtering; parameters and preprocessing are full-sample fit "
                "(disclosed limitation, not re-architected this pass).",
    }
    return series, meta


def build_s4(manifest, cfg) -> tuple:
    groups, sel = active_groups(cfg)
    est = LatentStateEstimator(method="tracking_regression", data_dir=_LATENT_DIR,
                                growth_vars=sel["growth"], inflation_vars=sel["inflation"],
                                vol_vars=sel["vol"])
    est.fit()  # full-sample OLS betas per group -- existing, disclosed limitation
    per_group = {name: tr.ut.reindex(manifest["df"].index) for name, tr in est.trackers.items()}
    combined, stats = aggregate_groups(per_group, manifest["train_end_date"])
    meta = {
        "method": "s4", "source": "LatentStateEstimator(method='tracking_regression'), "
                                   "own .trackers[*].ut re-aggregated with TRAIN-ONLY stats "
                                   "(bypassing its self.latent, which full-sample z-scores)",
        "active_groups": groups, "group_zscore_stats": stats,
        "uses_daily_indicators": True,
        "full_sample_fit_preprocessing": ["Tracking regression OLS betas (full sample)",
                                           "monthly_first_pc PCA/scaler (full sample)"],
        "future_interpolation_anchors": False,
        "retrospective": False,
        "note": "Causal in the daily .ut construction once betas are fixed; betas and PCA "
                "are full-sample fit (disclosed limitation, not re-architected this pass).",
    }
    return combined, meta


def build_interpolated(manifest, cfg, kind: str) -> tuple:
    """kind in {'s1' (linear), 's2' (ffill), 's3' (cubic spline)}."""
    groups, sel = active_groups(cfg)
    trading_dates = manifest["df"].index
    per_group = {}
    anchor_counts = {}
    for name in groups:
        monthly = monthly_first_pc(load_monthly_panel(name, sel[name]))[0]
        anchored = common.month_end_anchor(monthly, trading_dates)
        anchor_counts[name] = int(len(anchored))
        s = anchored.reindex(trading_dates)
        if kind == "s2":
            daily = s.ffill()
        elif kind == "s1":
            daily = s.interpolate(method="time", limit_area="inside")
        elif kind == "s3":
            x_anchor = anchored.index.view("int64")  # ns since epoch, exact elapsed time
            x_eval = trading_dates.view("int64")
            cs = CubicSpline(x_anchor, anchored.to_numpy(), bc_type="natural", extrapolate=False)
            daily = pd.Series(cs(x_eval), index=trading_dates)
        else:
            raise ValueError(kind)
        per_group[name] = daily

    combined, stats = aggregate_groups(per_group, manifest["train_end_date"])
    meta = {
        "method": kind,
        "source": f"monthly_first_pc per group, month-end-anchored (common.month_end_anchor), "
                  f"{'ffill' if kind=='s2' else 'time-based linear interpolation' if kind=='s1' else 'scipy CubicSpline(bc_type=natural, extrapolate=False)'}",
        "active_groups": groups, "group_zscore_stats": stats, "anchor_counts": anchor_counts,
        "uses_daily_indicators": False,
        "full_sample_fit_preprocessing": ["monthly_first_pc PCA/scaler (full sample)"],
        "future_interpolation_anchors": kind in ("s1", "s3"),
        "retrospective": kind in ("s1", "s3"),
        "note": ("Forward-fill is causal (only ever uses a past monthly anchor)." if kind == "s2" else
                 "Linear/cubic-spline interpolation requires a FUTURE monthly anchor between two "
                 "known points -- retrospective, not a real-time causal estimate. Report its "
                 "Group B/C results separately from the causal-ish methods, per the approved plan."),
    }
    return combined, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default="s0,s1,s2,s3,s4")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    cfg = common.get_default_config()
    manifest = common.get_window_manifest(cfg)
    print(f"Shared trading calendar: {len(manifest['df'])} days, "
          f"train_end_date={manifest['train_end_date'].date()}, "
          f"test_start_date={manifest['test_start_date'].date() if manifest['test_start_date'] is not None else None}")

    builders = {"s0": lambda: build_s0(manifest), "s4": lambda: build_s4(manifest, cfg),
                "s1": lambda: build_interpolated(manifest, cfg, "s1"),
                "s2": lambda: build_interpolated(manifest, cfg, "s2"),
                "s3": lambda: build_interpolated(manifest, cfg, "s3")}

    for m in methods:
        print(f"\n=== Building {m} ({common.METHOD_LABELS[m]}) ===")
        series, meta = builders[m]()
        n_nan = int(series.isna().sum())
        print(f"  {n_nan}/{len(series)} NaN, retrospective={meta['retrospective']}, "
              f"uses_daily_indicators={meta['uses_daily_indicators']}")

        out_dir = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[m])
        os.makedirs(out_dir, exist_ok=True)
        series.rename("daily_state").to_csv(os.path.join(out_dir, "daily_state.csv"), index_label="Date")
        meta["n_dates"] = len(series)
        meta["n_nan"] = n_nan
        meta["date_range"] = [str(series.index.min().date()), str(series.index.max().date())]
        meta["train_end_date"] = str(manifest["train_end_date"].date())
        with open(os.path.join(out_dir, "state_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved -> {out_dir}/daily_state.csv (+ state_meta.json)")

    print("\nDone.")


if __name__ == "__main__":
    main()
