"""
Rebuilds doobh_results_exp2_{causal,noncausal}/comparison_table_distributional.png
from the already-saved per-channel doobh_comparison_train.csv files -- same
navy-header/alternating-row style as experiment_1/build_doobh_tables.py.
Pure presentation, no new numbers computed here.

Only builds rows for channels that actually exist on disk yet -- S1/S3/S4
are still running as of this build, so partial tables (S0/S2 only) are
expected and clearly labeled as such in the title.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_ROOT = os.path.join(os.path.dirname(_HERE), "main_results")
CHANNEL_LABELS = {
    "exp2_s0": "S0 State-Space Model",
    "exp2_s1": "S1 Linear Interp.",
    "exp2_s2": "S2 Forward-Fill",
    "exp2_s3": "S3 Cubic Spline",
    "exp2_s4": "S4 Tracking Regression",
}
CHANNEL_ORDER = ["exp2_s0", "exp2_s1", "exp2_s2", "exp2_s3", "exp2_s4"]


def load_rows(results_dir):
    rows = []
    missing = []
    for ch in CHANNEL_ORDER:
        csv_path = os.path.join(results_dir, ch, "doobh_comparison_train.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            rows.append((CHANNEL_LABELS[ch], df.iloc[0]))
        else:
            missing.append(CHANNEL_LABELS[ch])
    return rows, missing


def fmt(x, sig=3):
    return f"{x:.{sig}g}"


def render(col_labels, cell_rows, title, out_path):
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(2.0 + 1.9 * n_cols, 0.75 * (len(cell_rows) + 1.6)))
    ax.axis("off")
    table = ax.table(cellText=cell_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(n_cols)))
    table.scale(1, 2.2)
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#f2f2f2" if r % 2 == 0 else "white")
            if c == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.02
    ax.set_title(title, fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def build_distributional(results_dir, causal_label):
    rows, missing = load_rows(results_dir)
    if missing:
        print(f"[note] {causal_label}: not yet finished -- missing {missing}")
    col_labels = ["Method", "Events", "Energy Score", "Marginal W1\n(avg, 15 assets)",
                  "Marginal W1\n(best asset)", "Marginal W1\n(worst asset)"]
    cell_rows = []
    for label, row in rows:
        cell_rows.append([
            label, int(row["n_real_events"]), fmt(row["energy_score"]),
            fmt(row["wasserstein1_marginal"]),
            f"{fmt(row['wasserstein1_marginal_best'])} ({row['wasserstein1_marginal_best_ticker']})",
            f"{fmt(row['wasserstein1_marginal_worst'])} ({row['wasserstein1_marginal_worst_ticker']})",
        ])
    title = f"Doob-h Exp2 — Distributional ({causal_label})"
    if missing:
        title += f"  [partial: {len(rows)}/5]"
    render(col_labels, cell_rows, title,
           os.path.join(results_dir, "comparison_table_distributional.png"))


def build_portfolio(results_dir, causal_label):
    rows, missing = load_rows(results_dir)
    col_labels = ["Method", "Events", "Variance:\nActual", "Variance:\nGenerated",
                  "Max Drawdown:\nActual", "Max Drawdown:\nGenerated",
                  "Cum Return:\nActual", "Cum Return:\nGenerated"]
    cell_rows = []
    for label, row in rows:
        cell_rows.append([
            label, int(row["n_real_events"]),
            fmt(row["portfolio_variance_real_median"]), fmt(row["portfolio_variance_gen_median"]),
            fmt(row["portfolio_mdd_real_median"]), fmt(row["portfolio_mdd_gen_median"]),
            fmt(row["portfolio_cumret_real_median"]), fmt(row["portfolio_cumret_gen_median"]),
        ])
    title = f"Doob-h Exp2 — Portfolio ({causal_label})"
    if missing:
        title += f"  [partial: {len(rows)}/5]"
    render(col_labels, cell_rows, title,
           os.path.join(results_dir, "comparison_table_portfolio.png"))


if __name__ == "__main__":
    for results_root, causal_label in [
        (os.path.join(_RESULTS_ROOT, "doobh_results_exp2_noncausal"), "event_causal=False"),
        (os.path.join(_RESULTS_ROOT, "doobh_results_exp2_causal"), "event_causal=True"),
    ]:
        build_distributional(results_root, causal_label)
        build_portfolio(results_root, causal_label)
