"""
Result tables (PNG + CSV) for the conditioning set selected in config -- M0-M4 (macro inclusion) and S0-S4
(macro embedding), conditional-only and causal rare-event. Reads only the folders the current config points to
(window_conditioning.experiment_paths(), config.get_run_tag) and writes each table next to its results:

  conditional-only  <exp>/causal/macro_inclusion/conditional_only/table_conditional_only.png (+ .csv)
                        from evaluation/model_comparison.csv (exp_evaluation.py)
                    <exp>/causal/macro_embed/conditional_only/table_conditional_only.png (+ .csv)
                        from group_bc_comparison_{causal,retrospective}.csv (main_exp2.py --mode evaluate)
  rare-event        <exp>/causal/<group>/rare_event/<run_tag>/table_distributional.png, table_portfolio.png (+ .csv)
                        every <channel>/ with rare_event_artifacts.npz; all metrics in daily log-return units
                        (inclusion: raw-return targets; embedding: gen r = mu_k + sigma_k * z on each method's own
                        residuals). Each channel's recorded settings must match the current config.

Usage (set the conditioning set / event settings in config/config.py first):
    python build_result_tables.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "macro_inclusion"), os.path.join(_HERE, "macro_embed_method"),
                os.path.dirname(_HERE)]
import build_rare_event_tables as style  # noqa: E402  (table look: render / fmt / FOOTNOTE)
from config import get_default_config, get_run_tag  # noqa: E402
from doobh_evaluation_exp2 import pooled_energy_score  # noqa: E402
from exp_evaluation import max_drawdown, portfolio_variance  # noqa: E402  (what the rare-event evaluators use)
from window_conditioning import experiment_paths  # noqa: E402

M_LABELS = {"ewma_mean_change": "M1 EWMA Mean", "ewma_vol_change": "M2 EWMA Volatility",
            "pca_change": "M3 Rolling PCA", "macro_change": "M4 Macro State"}
S_LABELS = {"exp2_s0": "S0 State-Space", "exp2_s1": "S1 Linear Interp. (retrospective)", "exp2_s2": "S2 Forward-Fill",
            "exp2_s3": "S3 Cubic Spline (retrospective)", "exp2_s4": "S4 Tracking Regression",
            "exp2_s1p": "S1' Linear Extrap.", "exp2_s3p": "S3' Cubic Spline Extrap."}
GROUPS = {"macro_inclusion": ("Macro Inclusion (M0-M4)", M_LABELS, "raw daily log returns"),
          "macro_embed": ("Macro Embedding (S0-S4)", S_LABELS, "each method's own NN residuals")}
EXP_LABEL = {"inflation_only": "Experiment 1 (Inflation Only)", "cpi_only": "Experiment 2 (CPI Only)"}


def settings_line(cfg):
    return (f"{EXP_LABEL.get(cfg.data.conditioning_set, cfg.data.conditioning_set)}  |  causal, event_lag_gap="
            f"{cfg.data.event_lag_gap}  |  {cfg.data.start_date}..{cfg.data.end_date}  |  seed {cfg.seed}")


def rare_metrics(real, gen):
    A = real.shape[1]
    w = [wasserstein_distance(real[:, a, -1], gen[:, a, -1]) for a in range(A)]
    vr = np.array([portfolio_variance(x) for x in real]); vg = np.array([portfolio_variance(x) for x in gen])
    mr = np.array([max_drawdown(x) for x in real]); mg = np.array([max_drawdown(x) for x in gen])
    return {"energy_score": pooled_energy_score(real, gen), "wasserstein1_marginal": float(np.mean(w)),
            "w1_best": float(np.min(w)), "w1_best_idx": int(np.argmin(w)),
            "w1_worst": float(np.max(w)), "w1_worst_idx": int(np.argmax(w)),
            "var_real_median": float(np.median(vr)), "var_gen_median": float(np.median(vg)),
            "var_w1": float(wasserstein_distance(vr, vg)), "mdd_real_median": float(np.median(mr)),
            "mdd_gen_median": float(np.median(mg)), "mdd_w1": float(wasserstein_distance(mr, mg)),
            "event_window_vol": float(real.std())}


def check_meta(cfg, d):
    """The channel's recorded data / event settings must match the current config."""
    meta = json.load(open(os.path.join(d, "generated_samples_meta.json")))
    want = {"conditioning_set": cfg.data.conditioning_set, "event_causal": cfg.data.event_causal,
            "event_lag_gap": cfg.data.event_lag_gap, "event_type": cfg.hfunction.event_type,
            "event_top_fraction": cfg.hfunction.event_threshold, "seed": cfg.seed}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise RuntimeError(f"{d}: recorded settings differ from the current config (recorded, config): {bad}")
    return meta


def rare_event_table(cfg, paths, group, tag):
    label, labels, target = GROUPS[group]
    root = os.path.join(paths["rare_inclusion" if group == "macro_inclusion" else "rare_embed"], tag)
    chans = [c for c in labels if os.path.exists(os.path.join(root, c, "rare_event_artifacts.npz"))]
    if not chans:
        print(f"[{group}] rare-event: no results in {root} -- skipped")
        return
    tickers = list(cfg.data.tickers)
    rows, native = [], []
    for ch in chans:
        d = os.path.join(root, ch)
        meta = check_meta(cfg, d)
        nat = pd.read_csv(os.path.join(d, "doobh_comparison_train.csv")).iloc[0]
        native.append(nat.rename(ch))
        a = np.load(os.path.join(d, "rare_event_artifacts.npz"))
        real = (a["real_csv_units"] if group == "macro_inclusion" else a["real_returns_reconstructed"]).astype(np.float64)
        m = rare_metrics(real, a["gen_returns"].astype(np.float64))
        if group == "macro_inclusion" and not np.isclose(m["energy_score"], nat["energy_score"], rtol=1e-6):
            raise RuntimeError(f"{ch}: return-unit recomputation disagrees with the evaluator")
        rows.append({"model": labels[ch], "channel": ch, "events": int(nat["n_real_events"]), "data_file": meta["data_file"],
                     **m, "w1_best_ticker": tickers[m["w1_best_idx"]], "w1_worst_ticker": tickers[m["w1_worst_idx"]]})
    pd.DataFrame(native).to_csv(os.path.join(root, "comparison_train_evaluator_native.csv"))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(root, "comparison_train_return_units.csv"), index=False)
    style.FOOTNOTE = (f"Rare event = top {cfg.hfunction.event_threshold:.1%} {cfg.hfunction.event_type} of each model's own "
                      f"conditioning series, causal (event window ends before the return window), in-sample (train) "
                      f"events. Diffusion targets: {target}. All metrics in daily log-return units.\nEach model is scored "
                      f"on its OWN event windows -- compare with the event-window volatility column before reading the ranking.")
    title = f"{label} — Rare Event (causal)\n{settings_line(cfg)}"
    raw = [[None, None, None, r["energy_score"], r["wasserstein1_marginal"]] for r in rows]
    style.render(["Model", "Events", "Event-window\nvolatility", "Energy Score", "Marginal W1\n(avg, 15 assets)",
                  "Marginal W1\n(best asset)", "Marginal W1\n(worst asset)"],
                 [[r["model"], r["events"], style.fmt(r["event_window_vol"], 3), style.fmt(r["energy_score"]),
                   style.fmt(r["wasserstein1_marginal"]), f"{style.fmt(r['w1_best'], 3)} ({r['w1_best_ticker']})",
                   f"{style.fmt(r['w1_worst'], 3)} ({r['w1_worst_ticker']})"] for r in rows],
                 title + "\nDistributional Fidelity", os.path.join(root, "table_distributional.png"), best_cols=(3, 4), raw=raw)
    raw = [[None] * 4 + [r["var_w1"], None, None, r["mdd_w1"]] for r in rows]
    style.render(["Model", "Events", "Variance:\nActual (med)", "Variance:\nGenerated (med)", "Variance W1",
                  "Max DD:\nActual (med)", "Max DD:\nGenerated (med)", "Max DD W1"],
                 [[r["model"], r["events"], style.fmt(r["var_real_median"], 3), style.fmt(r["var_gen_median"], 3),
                   style.fmt(r["var_w1"], 3), style.fmt(r["mdd_real_median"], 3), style.fmt(r["mdd_gen_median"], 3),
                   style.fmt(r["mdd_w1"], 3)] for r in rows],
                 title + "\nEqual-Weight Portfolio (10-day window)", os.path.join(root, "table_portfolio.png"),
                 best_cols=(4, 7), raw=raw)
    print(f"[{group}] rare-event: {len(rows)} models -> {root}/table_distributional.png, table_portfolio.png")


def conditional_only_table(cfg, paths, group):
    label = GROUPS[group][0]
    d = paths["macro_inclusion" if group == "macro_inclusion" else "macro_embed"]
    if group == "macro_inclusion":
        src = [os.path.join(d, "evaluation", "model_comparison.csv")]
    else:
        src = [os.path.join(d, f"group_bc_comparison_{k}.csv") for k in ("causal", "retrospective")]
    src = [p for p in src if os.path.exists(p)]
    if not src:
        print(f"[{group}] conditional-only: no evaluation output in {d} -- skipped")
        return
    t = pd.concat([pd.read_csv(p) for p in src], ignore_index=True)
    t.to_csv(os.path.join(d, "table_conditional_only.csv"), index=False)
    q = [c for c in t.columns if c.startswith("pinball_")]
    style.FOOTNOTE = ("Conditional-only: each model is conditioned on the causal window of its own variable; in-sample "
                      "evaluation windows shared by all models. Diffusion targets: " + GROUPS[group][2] + ". Lower is better "
                      "(lowest highlighted).")
    cols = ["energy_score", "wasserstein1_marginal", "portfolio_variance_w1", "portfolio_mdd_w1"] + q
    names = ["Energy Score", "Marginal W1", "Portfolio\nVariance W1", "Portfolio\nMax DD W1"] + \
            [f"Pinball\n{float(c.split('_')[1]):.0%}" for c in q]
    raw = [[None] + [r[c] for c in cols] for _, r in t.iterrows()]
    style.render(["Model"] + names, [[r["model"]] + [style.fmt(r[c]) for c in cols] for _, r in t.iterrows()],
                 f"{label} — Conditional-Only (causal)\n{settings_line(cfg)}", os.path.join(d, "table_conditional_only.png"),
                 best_cols=tuple(range(1, len(cols) + 1)), raw=raw)
    print(f"[{group}] conditional-only: {len(t)} models -> {d}/table_conditional_only.png")


def main():
    cfg = get_default_config()
    if not cfg.data.event_causal:
        raise SystemExit("config.data.event_causal must be True (causal-only)")
    paths = experiment_paths()
    tag = get_run_tag(cfg)
    print(f"{settings_line(cfg)}\nexperiment folder: {paths['experiment_dir']}\nrun_tag: {tag}")
    for group in GROUPS:
        conditional_only_table(cfg, paths, group)
        rare_event_table(cfg, paths, group, tag)


if __name__ == "__main__":
    main()
