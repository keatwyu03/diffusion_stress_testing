"""
Group A -- stationarity evaluation of the macro-conditionally standardized
residual returns (NOT the raw daily macro states), per the approved plan.
Operates on the plain daily residual series (each calendar date once), never
on overlapping shift-1 diffusion windows -- same dates/blocks for every
method.

Reuses the repo's existing ADF implementation (explore/stationary_diagnosis/
adf.py, regression="ct") unmodified. No local KPSS implementation exists
anywhere in the repo (confirmed by repo-wide grep) -- uses
statsmodels.tsa.stattools.kpss directly, regression="ct" to match ADF's
deterministic-term specification, settings documented below. Per the
approved plan, methods are NEVER ranked by raw p-value magnitude -- only the
joint 2x2 classification is used; the rolling diagnostics (periodogram,
variance, Wasserstein) are reported as descriptive second-order/distributional
stability measures, not formal stationarity tests.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import kpss

import common

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "explore", "stationary_diagnosis"))
from adf import adf_test  # noqa: E402

OUT_DIR = os.path.join(_HERE, "exp2_results")
ALPHA = 0.05
ADF_KPSS_REGRESSION = "ct"   # matched deterministic-term spec, both tests
ROLLING_WINDOW = 252         # trading days
WELCH_NPERSEG = 63           # ~1 quarter; same for every method
BLOCK_SIZE = 252             # non-overlapping Wasserstein-drift blocks


def joint_classification(adf_pvalue: float, kpss_pvalue: float, alpha: float = ALPHA) -> str:
    """ADF H0: unit root (non-stationary). KPSS H0: stationary. Do not
    interpret failure to reject KPSS as proof of stationarity -- this is a
    trend-stationarity diagnostic, not a formal proof either way."""
    adf_rejects = adf_pvalue < alpha       # ADF says: likely stationary
    kpss_rejects = kpss_pvalue < alpha     # KPSS says: likely NOT stationary
    if adf_rejects and not kpss_rejects:
        return "both_support_stationary"
    if not adf_rejects and kpss_rejects:
        return "both_support_nonstationary"
    if adf_rejects and kpss_rejects:
        return "conflicting_both_reject"
    return "conflicting_neither_rejects"


def adf_kpss_for_series(y: np.ndarray) -> dict:
    y = y[np.isfinite(y)]
    adf_res = adf_test(y, regression=ADF_KPSS_REGRESSION)
    # statsmodels kpss p-values are TABLE-BOUNDED (commonly clipped to
    # [0.01, 0.10] for regression="c"/[0.01,0.1] generally) -- documented here,
    # not silently treated as exact.
    kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(y, regression=ADF_KPSS_REGRESSION, nlags="auto")
    return {
        "adf_stat": adf_res["stat"], "adf_pvalue": adf_res["pvalue"], "adf_lag": adf_res["p"],
        "kpss_stat": float(kpss_stat), "kpss_pvalue": float(kpss_pvalue), "kpss_lags": int(kpss_lags),
        "kpss_pvalue_bounded": bool(kpss_pvalue in (0.01, 0.1) or kpss_pvalue <= 0.01 or kpss_pvalue >= 0.1),
        "classification": joint_classification(adf_res["pvalue"], float(kpss_pvalue)),
    }


def rolling_spectral_stability(y: pd.Series, window=ROLLING_WINDOW, nperseg=WELCH_NPERSEG):
    """Welch-estimated (averages modified periodograms -- an established
    smoothed-periodogram method, not a bespoke smoothing scheme) PSD per
    rolling window. Spectral-stability statistic (defined here, before any
    results are inspected): mean L2 distance between consecutive windows'
    log-PSDs, normalized by the series' own average log-PSD norm -- purely
    descriptive, NOT a formal stationarity test."""
    y = y.dropna()
    freqs = None
    psd_matrix = []
    dates = []
    for start in range(0, len(y) - window + 1, window // 4):  # 75%-overlap stride for the heatmap
        block = y.iloc[start:start + window].to_numpy()
        f, pxx = signal.welch(block, nperseg=min(nperseg, len(block)))
        if freqs is None:
            freqs = f
        psd_matrix.append(np.log(pxx + 1e-12))
        dates.append(y.index[start + window - 1])
    psd_matrix = np.array(psd_matrix)  # (n_windows, n_freqs)
    if len(psd_matrix) < 2:
        return {"freqs": freqs, "dates": dates, "psd_matrix": psd_matrix, "stability_stat": None}
    diffs = np.linalg.norm(np.diff(psd_matrix, axis=0), axis=1)
    norm = np.mean(np.linalg.norm(psd_matrix, axis=1)) + 1e-12
    stability_stat = float(np.mean(diffs) / norm)
    return {"freqs": freqs, "dates": [str(d.date()) for d in dates], "psd_matrix": psd_matrix,
            "stability_stat": stability_stat}


def rolling_variance_stability(y: pd.Series, window=ROLLING_WINDOW):
    y = y.dropna()
    roll_var = y.rolling(window).var().dropna()
    if len(roll_var) < 2 or roll_var.mean() == 0:
        return {"cov": None, "roll_var": roll_var}
    cov = float(roll_var.std() / abs(roll_var.mean()))
    return {"cov": cov, "roll_var": roll_var}


def rolling_wasserstein_drift(y: pd.Series, train_end_date: pd.Timestamp, block=BLOCK_SIZE):
    """Non-overlapping consecutive blocks; W1 between each pair of adjacent
    blocks' empirical distributions, averaged. Also reports a scale-adjusted
    version (raw W1 / ONE fixed train-period std for this asset/method) to
    guard against the trivial effect where simply compressing a residual
    series' scale also compresses its Wasserstein distances -- never
    re-standardize each block independently, which would erase the drift
    being measured."""
    y = y.dropna()
    train_std = float(y[y.index <= train_end_date].std())
    blocks = [y.iloc[i:i + block].to_numpy() for i in range(0, len(y) - block + 1, block)]
    if len(blocks) < 2:
        return {"mean_w1": None, "mean_w1_scaled": None, "n_blocks": len(blocks),
                "train_residual_std": train_std, "residual_mean": float(y.mean()), "residual_std": float(y.std())}
    dists = [wasserstein_distance(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)]
    mean_w1 = float(np.mean(dists))
    scaled = mean_w1 / train_std if train_std > 0 else None
    return {"mean_w1": mean_w1, "mean_w1_scaled": scaled, "n_blocks": len(blocks),
            "train_residual_std": train_std, "residual_mean": float(y.mean()), "residual_std": float(y.std())}


def evaluate_method(method: str, manifest, assets=None):
    resid_path = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[method], "standardizer", "residuals.csv")
    resid_df = pd.read_csv(resid_path, index_col=0, parse_dates=True)
    assets = assets or list(resid_df.columns)
    train_end_date = manifest["train_end_date"]

    per_asset = {}
    for ticker in assets:
        if ticker not in resid_df.columns:
            continue
        y = resid_df[ticker]
        n_valid = int(y.notna().sum())
        if n_valid < 2 * BLOCK_SIZE:
            per_asset[ticker] = {"insufficient_data": True, "n_valid": n_valid,
                                  "required": 2 * BLOCK_SIZE}
            print(f"    [insufficient data] {ticker}: {n_valid} valid obs, need >= {2*BLOCK_SIZE}")
            continue
        ak = adf_kpss_for_series(y.dropna().to_numpy())
        spec = rolling_spectral_stability(y)
        var_stab = rolling_variance_stability(y)
        w1 = rolling_wasserstein_drift(y, train_end_date)
        per_asset[ticker] = {
            "insufficient_data": False, "n_valid": n_valid,
            "adf_kpss": ak,
            "spectral_stability_stat": spec["stability_stat"],
            "variance_cov": var_stab["cov"],
            **w1,
        }

    valid_assets = {t: v for t, v in per_asset.items() if not v.get("insufficient_data")}
    frac_joint_stationary = (
        float(np.mean([v["adf_kpss"]["classification"] == "both_support_stationary" for v in valid_assets.values()]))
        if valid_assets else None
    )
    summary = {
        "method": method,
        "n_assets_evaluated": len(valid_assets), "n_assets_insufficient_data": len(per_asset) - len(valid_assets),
        "fraction_joint_adf_kpss_stationary": frac_joint_stationary,
        "mean_spectral_stability_stat": float(np.mean([v["spectral_stability_stat"] for v in valid_assets.values()
                                                        if v["spectral_stability_stat"] is not None])) if valid_assets else None,
        "mean_variance_cov": float(np.mean([v["variance_cov"] for v in valid_assets.values()
                                             if v["variance_cov"] is not None])) if valid_assets else None,
        "mean_wasserstein_drift_scaled": float(np.mean([v["mean_w1_scaled"] for v in valid_assets.values()
                                                         if v["mean_w1_scaled"] is not None])) if valid_assets else None,
    }
    return summary, per_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default="s0,s1,s2,s3,s4")
    parser.add_argument("--assets", type=str, default=None)
    args = parser.parse_args()

    cfg = common.get_default_config()
    manifest = common.get_window_manifest(cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    assets = [a.strip() for a in args.assets.split(",")] if args.assets else None

    summaries = []
    for m in methods:
        print(f"\n=== Stationarity: {m} ({common.METHOD_LABELS[m]}) ===")
        summary, per_asset = evaluate_method(m, manifest, assets)
        summaries.append(summary)
        print(f"  joint-stationary fraction: {summary['fraction_joint_adf_kpss_stationary']}, "
              f"mean spectral-stability: {summary['mean_spectral_stability_stat']}, "
              f"mean variance CoV: {summary['mean_variance_cov']}, "
              f"mean scaled W1 drift: {summary['mean_wasserstein_drift_scaled']}")

        out_dir = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[m])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "stationarity_per_asset.json"), "w") as f:
            json.dump(per_asset, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        with open(os.path.join(out_dir, "stationarity_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    table = pd.DataFrame(summaries)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "stationarity_comparison_table.csv"), index=False)
    print(f"\nSaved comparison table -> {OUT_DIR}/stationarity_comparison_table.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
