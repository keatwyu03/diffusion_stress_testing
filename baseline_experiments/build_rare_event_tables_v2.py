"""
Files the corrected (v2, 2026-11-07) FULL rare-event runs into
experiment_1 (Full)/<causal|noncausal>/<macro_inclusion|macro_embed>/rare_event/
and renders the tables in RETURN units for both groups:

  inclusion : targets are raw log returns -> the evaluator's own output (doobh_comparison_train.csv) is used as is.
  embedding : targets are each method's own NN residuals, so the evaluator's output is in different units per
              method; the same metric functions (pooled energy score, marginal W1, portfolio variance / max drawdown)
              are applied to the saved return-unit artifacts instead (gen r = mu_k + sigma_k * z, real windows
              reconstructed the same way). The evaluator's native CSV is kept alongside.

No new metrics. Each model is still scored against its OWN event windows (see the event-volatility column).

Usage:  python build_rare_event_tables_v2.py
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "macro_inclusion"), os.path.dirname(_HERE)]
from build_rare_event_tables import fmt, render  # noqa: E402  (same table style)
import build_rare_event_tables as old  # noqa: E402
from doobh_evaluation_exp2 import pooled_energy_score  # noqa: E402
from exp_evaluation import max_drawdown, portfolio_variance  # noqa: E402  (what the rare-event evaluators use)

EXP_DIR = os.path.join(_HERE, "experiment_1 (Full)")
COPY = "/scr/kadenwu/Conditional_diffusion_v2_rare_{}/baseline_experiments/main_results/{}"
SETS = {("macro_inclusion", "noncausal"): ("inc_nc", "doobh_results"),
        ("macro_inclusion", "causal"): ("inc_c", "doobh_results_causal"),
        ("macro_embed", "noncausal"): ("emb_nc", "doobh_results_exp2_noncausal_ownstd"),
        ("macro_embed", "causal"): ("emb_c", "doobh_results_exp2_causal_ownstd")}
CHANNELS = old.CHANNELS
GROUP_LABEL = old.GROUP_LABEL
TARGET = {"macro_inclusion": "raw daily log returns (all M)", "macro_embed": "each method's own NN residuals"}


def metrics(real, gen):
    A = real.shape[1]
    w = [wasserstein_distance(real[:, a, -1], gen[:, a, -1]) for a in range(A)]
    vr = np.array([portfolio_variance(x) for x in real]); vg = np.array([portfolio_variance(x) for x in gen])
    mr = np.array([max_drawdown(x) for x in real]); mg = np.array([max_drawdown(x) for x in gen])
    return {"energy_score": pooled_energy_score(real, gen), "wasserstein1_marginal": float(np.mean(w)),
            "w1_best": float(np.min(w)), "w1_best_idx": int(np.argmin(w)), "w1_worst": float(np.max(w)), "w1_worst_idx": int(np.argmax(w)),
            "var_real_median": float(np.median(vr)), "var_gen_median": float(np.median(vg)), "var_w1": float(wasserstein_distance(vr, vg)),
            "mdd_real_median": float(np.median(mr)), "mdd_gen_median": float(np.median(mg)), "mdd_w1": float(wasserstein_distance(mr, mg)),
            "event_window_vol": float(real.std())}


def build(group, mode):
    short, sub = SETS[(group, mode)]
    src = COPY.format(short, sub)
    out_dir = os.path.join(EXP_DIR, mode, group, "rare_event")
    extras = os.path.join(EXP_DIR, "extras", "rare_event", mode, group)
    os.makedirs(out_dir, exist_ok=True)
    from config import get_default_config
    tickers = list(get_default_config().data.tickers)
    native, rows = [], []
    for ch, label in CHANNELS[group]:
        d = os.path.join(src, ch)
        nat = pd.read_csv(os.path.join(d, "doobh_comparison_train.csv")).iloc[0]
        native.append(nat)
        a = np.load(os.path.join(d, "rare_event_artifacts.npz"))
        real = (a["real_csv_units"] if group == "macro_inclusion" else a["real_returns_reconstructed"]).astype(np.float64)
        m = metrics(real, a["gen_returns"].astype(np.float64))
        if group == "macro_inclusion" and not np.isclose(m["energy_score"], nat["energy_score"], rtol=1e-6):
            raise RuntimeError(f"{ch}: return-unit recomputation disagrees with the evaluator")
        rows.append({"model": label, "channel": ch, "events": int(nat["n_real_events"]), **m,
                     "w1_best_ticker": tickers[m["w1_best_idx"]], "w1_worst_ticker": tickers[m["w1_worst_idx"]]})
        dst = os.path.join(extras, ch)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*")):
            shutil.copy2(f, dst)
    pd.DataFrame(native).to_csv(os.path.join(out_dir, "comparison_train_evaluator_native.csv"), index=False)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "comparison_train_return_units.csv"), index=False)

    old.FOOTNOTE = (f"Rare event = top 7.5% upper change (Z_end - Z_start), in-sample (train) events. Diffusion targets: {TARGET[group]}. "
                    "All metrics in daily log-return units.\nEach model is scored on its OWN event windows -- compare with the "
                    "event-window volatility column before reading the ranking.")
    title = f"Experiment 1 (Full) — {GROUP_LABEL[group]} — Rare Event ({mode}) — corrected run"
    raw = [[None, None, None, r["energy_score"], r["wasserstein1_marginal"]] for r in rows]
    render(["Model", "Events", "Event-window\nvolatility", "Energy Score", "Marginal W1\n(avg, 15 assets)",
            "Marginal W1\n(best asset)", "Marginal W1\n(worst asset)"],
           [[r["model"], r["events"], fmt(r["event_window_vol"], 3), fmt(r["energy_score"]), fmt(r["wasserstein1_marginal"]),
             f"{fmt(r['w1_best'], 3)} ({r['w1_best_ticker']})", f"{fmt(r['w1_worst'], 3)} ({r['w1_worst_ticker']})"] for r in rows],
           title + "\nDistributional Fidelity", os.path.join(out_dir, "table_distributional.png"), best_cols=(3, 4), raw=raw)
    raw = [[None] * 4 + [r["var_w1"], None, None, r["mdd_w1"]] for r in rows]
    render(["Model", "Events", "Variance:\nActual (med)", "Variance:\nGenerated (med)", "Variance W1",
            "Max DD:\nActual (med)", "Max DD:\nGenerated (med)", "Max DD W1"],
           [[r["model"], r["events"], fmt(r["var_real_median"], 3), fmt(r["var_gen_median"], 3), fmt(r["var_w1"], 3),
             fmt(r["mdd_real_median"], 3), fmt(r["mdd_gen_median"], 3), fmt(r["mdd_w1"], 3)] for r in rows],
           title + "\nEqual-Weight Portfolio (10-day window)", os.path.join(out_dir, "table_portfolio.png"), best_cols=(4, 7), raw=raw)
    print(f"{group} {mode}: filed -> {out_dir}")
    return t.assign(group=group, mode=mode)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(_HERE))
    allt = pd.concat([build(g, m) for (g, m) in SETS])
    allt.to_csv(os.path.join(EXP_DIR, "rare_event_summary_return_units.csv"), index=False)
