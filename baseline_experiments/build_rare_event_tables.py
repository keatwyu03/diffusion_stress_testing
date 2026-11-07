"""
Files one finished rare-event (Doob-h) run into its experiment folder and
renders its comparison tables. Pure presentation + copying: no numbers are
recomputed, the per-channel doobh_comparison_train.csv files are read as saved.

    <experiment>/<causal|noncausal>/<macro_inclusion|macro_embed>/rare_event/
        comparison_train.csv              one row per model (all channels)
        table_distributional.png          energy score, marginal W1
        table_portfolio_variance.png      portfolio variance actual vs generated
    <experiment>/extras/rare_event/<causal|noncausal>/<group>/<channel>/
        generated samples, h-function checkpoint + losses (copied)
    <experiment>/extras/rare_event/<causal|noncausal>/<group>/score_model/

Units: the rare-event pipeline models the production data_for_diffusion.csv,
i.e. macro-conditionally standardized residuals (std ~1), not raw returns.
Max drawdown, cumulative return and the pooled pinball loss (which is on
cumulative return) apply exp() to these values as if they were log returns,
so they are left out of the tables (still in comparison_train.csv).

Usage:
    python build_rare_event_tables.py --experiment full --mode causal --group macro_inclusion
    python build_rare_event_tables.py --all
"""
import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
COPY_ROOT = "/scr/kadenwu/Conditional_diffusion_rare_{prefix}{short}"
EXPERIMENT_DIR = {"full": "experiment_1 (Full)", "cpi": "experiment_2 (CPI Only)"}
EXPERIMENT_LABEL = {"full": "Experiment 1 (Full)", "cpi": "Experiment 2 (CPI Only)"}
# (group, mode) -> (copy suffix, results subfolder inside the copy's main_results/)
SOURCES = {
    ("macro_inclusion", "noncausal"): ("inc_nc", "doobh_results"),
    ("macro_inclusion", "causal"): ("inc_c", "doobh_results_causal"),
    ("macro_embed", "noncausal"): ("emb_nc", "doobh_results_exp2_noncausal"),
    ("macro_embed", "causal"): ("emb_c", "doobh_results_exp2_causal"),
}
CHANNELS = {
    "macro_inclusion": [("ewma_mean_change", "M1 EWMA Mean"), ("ewma_vol_change", "M2 EWMA Volatility"),
                        ("pca_change", "M3 Rolling PCA"), ("macro_change", "M4 Macro State")],
    "macro_embed": [("exp2_s0", "S0 State-Space"), ("exp2_s1", "S1 Linear Interp."), ("exp2_s2", "S2 Forward-Fill"),
                    ("exp2_s3", "S3 Cubic Spline"), ("exp2_s4", "S4 Tracking Regr.")],
}
GROUP_LABEL = {"macro_inclusion": "Macro Inclusion", "macro_embed": "Macro Embedding"}
FOOTNOTE = ("Rare event = top 7.5% upper change (Z_end - Z_start), in-sample (train) events. Units: macro-standardized "
            "residuals.\nDrawdown, cumulative return and pinball loss omitted (they treat residuals as log returns).")


def fmt(x, sig=4):
    return f"{x:.{sig}g}"


def render(col_labels, cell_rows, title, out_path, best_cols=(), raw=None):
    """best_cols: column indices where the lowest value (from raw[row][col]) is highlighted."""
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * n_cols, 0.62 * (len(cell_rows) + 2.6)))
    ax.axis("off")
    table = ax.table(cellText=cell_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.auto_set_column_width(col=list(range(n_cols)))
    table.scale(1, 2.0)
    best = {c: min(range(len(cell_rows)), key=lambda r: raw[r][c]) for c in best_cols}
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(weight="bold", color="white")
            continue
        cell.set_facecolor("#f2f2f2" if r % 2 == 0 else "white")
        if c == 0:
            cell.set_text_props(ha="left")
        if c in best and best[c] == r - 1:
            cell.set_facecolor("#d4edda")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, pad=12)
    fig.text(0.5, 0.01, FOOTNOTE + "  Green = best (lowest) in column.", ha="center", va="bottom", fontsize=7.5,
             color="#555555")
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def build(experiment: str, mode: str, group: str):
    short, sub = SOURCES[(group, mode)]
    copy = COPY_ROOT.format(prefix="cpi_" if experiment == "cpi" else "", short=short)
    src = os.path.join(copy, "baseline_experiments", "main_results", sub)
    exp_dir = os.path.join(_HERE, EXPERIMENT_DIR[experiment])
    out_dir = os.path.join(exp_dir, mode, group, "rare_event")
    extras = os.path.join(exp_dir, "extras", "rare_event", mode, group)
    print(f"\n{EXPERIMENT_LABEL[experiment]} | {GROUP_LABEL[group]} | {mode}\n  from {src}")

    rows = []
    for ch, label in CHANNELS[group]:
        f = os.path.join(src, ch, "doobh_comparison_train.csv")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} missing -- this run hasn't finished every channel yet.")
        r = pd.read_csv(f).iloc[0]
        if r["split"] != "train":
            raise ValueError(f"{f}: split={r['split']!r}, expected in-sample 'train'.")
        if "event_causal" in r and bool(r["event_causal"]) != (mode == "causal"):
            raise ValueError(f"{f}: event_causal={r['event_causal']} but filing it as {mode}.")
        rows.append((label, r))

    os.makedirs(out_dir, exist_ok=True)
    table = pd.DataFrame([r for _, r in rows])
    table.insert(0, "model", [label for label, _ in rows])
    table.to_csv(os.path.join(out_dir, "comparison_train.csv"), index=False)
    print(f"  Saved -> {os.path.join(out_dir, 'comparison_train.csv')}")

    title = f"{EXPERIMENT_LABEL[experiment]} — {GROUP_LABEL[group]} — Rare Event ({mode})"
    dist_raw = [[None, None, r["energy_score"], r["wasserstein1_marginal"]] for _, r in rows]
    render(["Model", "Events", "Energy Score", "Marginal W1\n(avg, 15 assets)", "Marginal W1\n(best asset)",
            "Marginal W1\n(worst asset)"],
           [[label, int(r["n_real_events"]), fmt(r["energy_score"]), fmt(r["wasserstein1_marginal"]),
             f"{fmt(r['wasserstein1_marginal_best'], 3)} ({r['wasserstein1_marginal_best_ticker']})",
             f"{fmt(r['wasserstein1_marginal_worst'], 3)} ({r['wasserstein1_marginal_worst_ticker']})"]
            for label, r in rows],
           title + "\nDistributional Fidelity", os.path.join(out_dir, "table_distributional.png"),
           best_cols=(2, 3), raw=dist_raw)
    var_raw = [[None] * 4 + [r["portfolio_variance_w1"]] for _, r in rows]
    render(["Model", "Events", "Variance:\nActual (median)", "Variance:\nGenerated (median)", "Variance W1\n(actual vs gen)"],
           [[label, int(r["n_real_events"]), fmt(r["portfolio_variance_real_median"]),
             fmt(r["portfolio_variance_gen_median"]), fmt(r["portfolio_variance_w1"])] for label, r in rows],
           title + "\nEqual-Weight Portfolio Variance (10-day window)",
           os.path.join(out_dir, "table_portfolio_variance.png"), best_cols=(4,), raw=var_raw)

    # Raw artifacts -> extras (copied; the isolated run copy is left intact).
    for ch, _ in CHANNELS[group]:
        dst = os.path.join(extras, ch)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, ch, "*")):
            shutil.copy2(f, dst)
    score_dst = os.path.join(extras, "score_model")
    os.makedirs(score_dst, exist_ok=True)
    for f in glob.glob(os.path.join(copy, "ckpt_new", "score_*")):
        shutil.copy2(f, score_dst)
    print(f"  Copied raw artifacts -> {extras}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=list(EXPERIMENT_DIR))
    p.add_argument("--mode", choices=["causal", "noncausal"])
    p.add_argument("--group", choices=list(CHANNELS))
    p.add_argument("--all", action="store_true", help="Every experiment/mode/group whose run has finished.")
    args = p.parse_args()
    if args.all:
        for e in EXPERIMENT_DIR:
            for g in CHANNELS:
                for m in ("noncausal", "causal"):
                    try:
                        build(e, m, g)
                    except FileNotFoundError as err:
                        print(f"  [skip] {err}")
    else:
        build(args.experiment, args.mode, args.group)


if __name__ == "__main__":
    main()
