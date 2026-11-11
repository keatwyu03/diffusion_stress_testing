"""
S0-S4 stationarity analysis for the conditioning set selected in config (config.data.conditioning_set; a Slurm job may
override it with CDG_CONDITIONING_SET). Writes <experiment folder>/stationarity_analysis/.

Residuals (nothing is re-standardized and no NN is fitted here):
  S0     explore/conditioning_sets/<set>/data_for_diffusion.csv (config.data.csv_path) -- the set's existing residuals
         z = (r - mu(m_t)) / sigma(m_t), built once by build_conditioning_sets.py (run_ticker, seed 0, own span)
  S1-S4  <experiment>/extras/standardization_fullhorizon/<method>/residuals.csv -- standardize_fullhorizon_set.py, the
         same run_ticker procedure (seed 0, own span) on that method's own daily state, fitted once
Each method is evaluated on its own span.

Tests (same as Table 4): statsmodels adfuller(regression="c", autolag="AIC") and kpss(regression="c", nlags="auto");
"Stationary" iff ADF p < 0.05 and KPSS p >= 0.05 (KPSS p-values are table-bounded to [0.01, 0.10]).
Rolling diagnostics (lower = more stable): stationarity_evaluation.rolling_spectral_stability /
rolling_variance_stability / rolling_wasserstein_drift (scaled by the train-period std; train end = the shared
window manifest), imported unchanged.

Outputs: joint_adf_kpss_<method>.tex (Table-4 format), joint_adf_kpss_all_methods.csv (per stock),
summary_all_methods.csv + summary_all_methods.png, README.json (settings / provenance).

Usage (CDG_CONDITIONING_SET=<set> or the config's conditioning_set; after standardize_fullhorizon_set.py --method s1..s4):
    python stationarity_conditioning_set.py [--methods s0,s1,s2,s3,s4]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

import common
import stationarity_evaluation as SE   # imported, not modified
from window_conditioning import experiment_paths

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import build_rare_event_tables as style  # noqa: E402  (table look)

warnings.filterwarnings("ignore")


def kp_fmt(p):
    return "$>$0.10" if p >= 0.1 else ("$<$0.01" if p <= 0.01 else f"{p:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="s0,s1,s2,s3,s4")
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    cfg = common.get_default_config()
    paths = experiment_paths()
    out = os.path.join(paths["experiment_dir"], "stationarity_analysis")
    os.makedirs(out, exist_ok=True)
    train_end = common.get_window_manifest(cfg)["train_end_date"]
    fh_dir = os.path.join(paths["experiment_dir"], "extras", "standardization_fullhorizon")
    print(f"conditioning_set = {cfg.data.conditioning_set} | S0 residuals {cfg.data.csv_path} | S1-S4 residuals {fh_dir} | out {out}")
    sources = {}

    per, summ, spans = [], [], {}
    for m in methods:
        if m == "s0":   # the set's existing explore residuals
            src = cfg.data.csv_path
        else:
            d = os.path.join(fh_dir, common.METHOD_DIRNAMES[m])
            meta = json.load(open(os.path.join(d, "meta.json")))
            if meta.get("conditioning_set") != cfg.data.conditioning_set or meta.get("seed") != cfg.seed:
                raise SystemExit(f"{d} was fit for set {meta.get('conditioning_set')!r} / seed {meta.get('seed')}, "
                                 f"not {cfg.data.conditioning_set!r} / {cfg.seed}")
            src = os.path.join(d, "residuals.csv")
        sources[m] = src
        r = pd.read_csv(src, index_col=0, parse_dates=True)
        rows = []
        for t in common.TICKERS:
            y = r[t].dropna()
            a_ = adfuller(y.to_numpy(), regression="c", autolag="AIC")
            k_ = kpss(y.to_numpy(), regression="c", nlags="auto")
            rows.append({"method": m, "ticker": t, "kpss_stat": float(k_[0]), "kpss_p": float(k_[1]),
                         "adf_stat": float(a_[0]), "adf_p": float(a_[1]), "n_obs": len(y),
                         "start": str(y.index[0].date()), "end": str(y.index[-1].date()),
                         "joint": "Stationary" if (a_[1] < 0.05 and k_[1] >= 0.05) else "Not stationary",
                         "spectral_stability": SE.rolling_spectral_stability(y)["stability_stat"],
                         "variance_cov": SE.rolling_variance_stability(y)["cov"],
                         "w1_drift_scaled": SE.rolling_wasserstein_drift(y, train_end)["mean_w1_scaled"]})
        T = pd.DataFrame(rows)
        per.append(T)
        spans[m] = f"{T.start.iloc[0]}..{T.end.iloc[0]} ({int(T.n_obs.iloc[0])} days)"
        tex = ["\\begin{tabular}{lccccc}", "\\toprule", " & \\multicolumn{2}{c}{KPSS} & \\multicolumn{2}{c}{ADF} & \\\\",
               "Ticker & Test Statistic & $p$-value & Test Statistic & $p$-value & Joint Result \\\\", "\\midrule"]
        tex += [f"{x.ticker} & {x.kpss_stat:.4f} & {kp_fmt(x.kpss_p)} & ${x.adf_stat:.2f}$ & {x.adf_p:.4f} & {x.joint} \\\\"
                for x in T.itertuples()]
        open(os.path.join(out, f"joint_adf_kpss_{common.METHOD_DIRNAMES[m]}.tex"), "w").write(
            "\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n")
        n_st = int((T.joint == "Stationary").sum())
        summ.append({"method": m, "label": common.METHOD_LABELS[m], "retrospective": common.RETROSPECTIVE[m],
                     "stationary": f"{n_st}/{len(T)}", "fraction_stationary": n_st / len(T),
                     "mean_spectral_stability": float(np.nanmean(T.spectral_stability.astype(float))),
                     "mean_variance_cov": float(np.nanmean(T.variance_cov.astype(float))),
                     "mean_w1_drift_scaled": float(np.nanmean(T.w1_drift_scaled.astype(float))), "span": spans[m]})
        print(f"{m}: {n_st}/{len(T)} jointly stationary ({spans[m]})", flush=True)

    pd.concat(per).to_csv(os.path.join(out, "joint_adf_kpss_all_methods.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, "summary_all_methods.csv"), index=False)
    style.FOOTNOTE = ("Residuals: S0 = the set's explore data_for_diffusion.csv; S1-S4 = the same run_ticker NN (seed "
                      f"{cfg.seed}, own span) on each method's state. Each method on its own span. ADF c/AIC + KPSS c/auto; "
                      "stationary iff ADF p<0.05 and KPSS p>=0.05. Rolling diagnostics: lower = more stable.")
    raw = [[None, -x.fraction_stationary, x.mean_spectral_stability, x.mean_variance_cov, x.mean_w1_drift_scaled, None]
           for x in S.itertuples()]
    style.render(["Method", "Joint ADF+KPSS\nstationary", "Spectral\nstability", "Variance\nCoV", "Wasserstein\ndrift (scaled)",
                  "Span"],
                 [[x.label + (" (retrospective)" if x.retrospective else ""), x.stationary, style.fmt(x.mean_spectral_stability, 4),
                   style.fmt(x.mean_variance_cov, 4), style.fmt(x.mean_w1_drift_scaled, 4), x.span] for x in S.itertuples()],
                 f"S0-S4 Stationarity — {cfg.data.conditioning_set}\n{cfg.data.start_date}..{cfg.data.end_date} window, "
                 f"seed {cfg.seed}", os.path.join(out, "summary_all_methods.png"), best_cols=(1, 2, 3, 4), raw=raw)
    json.dump({"conditioning_set": cfg.data.conditioning_set, "residual_files": sources,
               "note": "S0 = existing explore residuals; S1-S4 = standardize_fullhorizon_set.py (fitted once)", "methods": methods, "spans": spans, "seed": cfg.seed,
               "window": [cfg.data.start_date, cfg.data.end_date], "train_end_for_w1_scale": str(train_end.date()),
               "tests": "adfuller(c, AIC) + kpss(c, auto); Stationary iff ADF p<0.05 and KPSS p>=0.05"},
              open(os.path.join(out, "README.json"), "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
