"""
INCREMENTAL stationarity results for the two new methods S1' (s1p) and S3' (s3p). The existing S0-S4 results
(experiment_1 (Full)/stationarity_tests/) are reused exactly as stored -- never recomputed, never modified --
and the extended seven-method tables are written to a SEPARATE folder (stationarity_tests_extended/).

Methodology = the one behind the existing table (stationarity_comparison_table.csv,
stationarity_per_asset_fullhorizon_c.json, */stationarity_summary.json: standardizer "fullhorizon", regression "c"):
  residuals : full-horizon production NN standardizer on the method's own daily state
              (fullhorizon_standardize.py procedure; S1'/S3' via fullhorizon_standardize_new_methods.py)
  ADF+KPSS  : statsmodels adfuller(regression="c", autolag="AIC") + kpss(regression="c", nlags="auto"),
              joint classification stationarity_evaluation.joint_classification, alpha 0.05
  rolling   : stationarity_evaluation.rolling_spectral_stability / rolling_variance_stability /
              rolling_wasserstein_drift(train_end_date = manifest train_end_date), imported unchanged
The script that produced the existing table is not in the repository, so this reconstruction is VERIFIED first:
the same code is applied, in memory only, to the existing S0 residuals and must reproduce the stored S0 table row
and the stored per-asset S0 ADF/KPSS values exactly; otherwise nothing is computed or written.

Usage:  python stationarity_add_new_methods.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

import common
import stationarity_evaluation as SE   # imported, not modified

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from build_rare_event_tables import fmt, render  # noqa: E402
import build_rare_event_tables as _tbl  # noqa: E402

FULL = os.path.join(os.path.dirname(_HERE), "experiment_1 (Full)")
EXIST = os.path.join(FULL, "stationarity_tests")
STATES = os.path.join(FULL, "extras", "exp2_results")
OUT = os.path.join(FULL, "stationarity_tests_extended")
NEW = ["s1p", "s3p"]
ORDER7 = ["s0", "s1", "s1p", "s2", "s3", "s3p", "s4"]
LABEL = {"s0": "S0 State-Space", "s1": "S1 Linear Interp. (retrospective)", "s1p": "S1' Linear Extrap. (causal, NEW)",
         "s2": "S2 Forward-Fill", "s3": "S3 Cubic Spline Interp. (retrospective)", "s3p": "S3' Cubic Spline Extrap. (causal, NEW)",
         "s4": "S4 Tracking Regr."}
COLS = ["fraction_joint_adf_kpss_stationary", "mean_spectral_stability_stat", "mean_variance_cov", "mean_wasserstein_drift_scaled"]


def adf_kpss_c(y):
    a = adfuller(y, regression="c", autolag="AIC")
    k = kpss(y, regression="c", nlags="auto")
    return {"adf_stat": float(a[0]), "adf_pvalue": float(a[1]), "adf_lag": int(a[2]), "kpss_stat": float(k[0]),
            "kpss_pvalue": float(k[1]), "kpss_lags": int(k[2]),
            "classification": SE.joint_classification(float(a[1]), float(k[1]))}


def evaluate(resid_df, train_end):
    per = {}
    for t in common.TICKERS:
        y = resid_df[t]
        n_valid = int(y.notna().sum())
        if n_valid < 2 * SE.BLOCK_SIZE:
            per[t] = {"insufficient_data": True, "n_valid": n_valid}
            continue
        per[t] = {"insufficient_data": False, "n_valid": n_valid, "first_date": str(y.dropna().index[0].date()),
                  "last_date": str(y.dropna().index[-1].date()), **adf_kpss_c(y.dropna().to_numpy()),
                  "spectral_stability_stat": SE.rolling_spectral_stability(y)["stability_stat"],
                  "variance_cov": SE.rolling_variance_stability(y)["cov"],
                  "mean_w1_scaled": SE.rolling_wasserstein_drift(y, train_end)["mean_w1_scaled"]}
    v = [x for x in per.values() if not x["insufficient_data"]]
    summ = {"n_assets_evaluated": len(v), "n_assets_insufficient_data": len(per) - len(v),
            "fraction_joint_adf_kpss_stationary": float(np.mean([x["classification"] == "both_support_stationary" for x in v])),
            "mean_spectral_stability_stat": float(np.mean([x["spectral_stability_stat"] for x in v])),
            "mean_variance_cov": float(np.mean([x["variance_cov"] for x in v])),
            "mean_wasserstein_drift_scaled": float(np.mean([x["mean_w1_scaled"] for x in v]))}
    return summ, per


def load_resid(m):
    return pd.read_csv(os.path.join(STATES, common.METHOD_DIRNAMES[m], "standardizer_fullhorizon", "residuals.csv"),
                       index_col=0, parse_dates=True)


def main():
    cfg = common.get_default_config()
    train_end = common.get_window_manifest(cfg)["train_end_date"]
    table = pd.read_csv(os.path.join(EXIST, "stationarity_comparison_table.csv")).set_index("method")
    stored_c = json.load(open(os.path.join(EXIST, "stationarity_per_asset_fullhorizon_c.json")))

    # ---- 1) verify the reconstructed methodology against the stored S0 results (in memory; nothing written) ----
    s0_summ, s0_per = evaluate(load_resid("s0"), train_end)
    agg_ok = all(np.isclose(s0_summ[c], table.loc["s0", c], rtol=1e-9, atol=1e-12) for c in COLS)
    asset_ok = all(np.isclose(s0_per[t]["adf_stat"], stored_c["s0"][t]["adf_stat"]) and
                   np.isclose(s0_per[t]["kpss_stat"], stored_c["s0"][t]["kpss_stat"]) and
                   s0_per[t]["classification"] == stored_c["s0"][t]["classification"] for t in common.TICKERS)
    print(f"[verify] S0 table row reproduced: {agg_ok} ({ {c: round(s0_summ[c], 6) for c in COLS} } vs "
          f"{ {c: round(table.loc['s0', c], 6) for c in COLS} })")
    print(f"[verify] S0 per-asset ADF/KPSS reproduced: {asset_ok}")
    if not (agg_ok and asset_ok):
        sys.exit("[STOP] reconstruction does not reproduce the existing S0 results -- no new results written.")

    # ---- 2) compute ONLY S1' and S3' ----
    os.makedirs(OUT, exist_ok=True)
    new_rows, new_per = {}, {}
    for m in NEW:
        summ, per = evaluate(load_resid(m), train_end)
        new_rows[m], new_per[m] = summ, per
        d = os.path.join(OUT, common.METHOD_DIRNAMES[m]); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "stationarity_summary.json"), "w") as f:
            json.dump({"method": m, **summ, "regression_spec": "c", "standardizer": "fullhorizon"}, f, indent=2)
        with open(os.path.join(d, "stationarity_per_asset_fullhorizon_c.json"), "w") as f:
            json.dump(per, f, indent=2)
        print(f"[{m}] {summ}")

    # ---- 3) extended tables: stored S0-S4 rows verbatim + new rows ----
    ext = pd.concat([table.reset_index(), pd.DataFrame([{"method": m, **new_rows[m]} for m in NEW])], ignore_index=True)
    ext["label"] = ext["method"].map(LABEL)
    ext["retrospective"] = ext["method"].map(common.RETROSPECTIVE)
    span = {}
    for m in ORDER7:   # evaluation span per method (read from the residual files, not recomputed results)
        r = load_resid(m).dropna(how="all")
        span[m] = f"{r.index[0].date()}..{r.index[-1].date()} ({len(r)} days)"
    ext["evaluation_span"] = ext["method"].map(span)
    ext = ext.set_index("method").loc[ORDER7].reset_index()
    ext.to_csv(os.path.join(OUT, "stationarity_comparison_table_7methods.csv"), index=False)
    grid = pd.DataFrame({m: {t: (stored_c[m][t]["classification"] if m in stored_c else new_per[m][t]["classification"])
                             for t in common.TICKERS} for m in ORDER7})
    grid.to_csv(os.path.join(OUT, "per_asset_joint_adf_kpss_classification_7methods.csv"), index_label="asset")

    _tbl.FOOTNOTE = ("S0-S4 rows: existing results, reused verbatim. S1'/S3': same methodology (full-horizon NN standardizer, ADF+KPSS 'c', "
                     "alpha 0.05, 252-day rolling diagnostics).\nEach method is evaluated on its OWN available span (existing convention) -- "
                     "S0/S4 from 2000, S1/S2/S3/S1'/S3' from 1991-92. Rolling diagnostics: lower = more stable.")
    def draw(methods, name, title):
        t = ext.set_index("method").loc[methods].reset_index()
        raw = [[None, -r.fraction_joint_adf_kpss_stationary, r.mean_spectral_stability_stat, r.mean_variance_cov,
                r.mean_wasserstein_drift_scaled, None] for r in t.itertuples()]
        render(["Method", "Joint ADF+KPSS\nstationary (share)", "Spectral\nstability", "Variance\nCoV", "Wasserstein\ndrift (scaled)",
                "Evaluation span"],
               [[r.label, f"{r.fraction_joint_adf_kpss_stationary:.3f}", fmt(r.mean_spectral_stability_stat, 4),
                 fmt(r.mean_variance_cov, 4), fmt(r.mean_wasserstein_drift_scaled, 4), r.evaluation_span] for r in t.itertuples()],
               title, os.path.join(OUT, f"{name}.png"), best_cols=(1, 2, 3, 4), raw=raw)
    draw(ORDER7, "stationarity_comparison_table_7methods", "Stationarity of NN-standardized residuals — 7 macro representations")
    draw(["s0", "s1", "s1p"], "compare_S0_vs_S1_S1prime", "S0 vs S1 (retrospective) vs S1' (causal)")
    draw(["s0", "s3", "s3p"], "compare_S0_vs_S3_S3prime", "S0 vs S3 (retrospective) vs S3' (causal)")
    draw(["s0", "s2", "s4"], "compare_S0_vs_S2_S4", "S0 vs S2 vs S4")
    with open(os.path.join(OUT, "README.json"), "w") as f:
        json.dump({"what": "S0-S4 reused from stationarity_tests/ (unchanged); S1'/S3' computed here with the same methodology",
                   "verification": {"S0_table_row_reproduced": agg_ok, "S0_per_asset_adf_kpss_reproduced": asset_ok},
                   "evaluation_span": span, "train_end_date_for_w1_scale": str(train_end.date()),
                   "limitations": ["methods are evaluated on different spans (existing convention); S0/S4 start 2000",
                                   "raw-return (no macro) stationarity is not part of the existing results",
                                   "single full-horizon period (existing convention); no sub-period breakdown",
                                   "S1/S3 retrospective; S1'/S3' causal in state construction but use full-sample monthly PCA",
                                   "KPSS p-values are table-bounded to [0.01, 0.10]"]}, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
