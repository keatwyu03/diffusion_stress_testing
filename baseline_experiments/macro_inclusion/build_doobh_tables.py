"""
Rebuilds doobh_results/comparison_table_distributional.png and
comparison_table_portfolio.png from the already-saved per-channel
doobh_comparison_train.csv files. Pure presentation (same navy-header,
alternating-row style as before) -- no new numbers are computed here.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_ROOT = os.path.join(os.path.dirname(_HERE), "main_results")
OUT_DIR = os.path.join(_RESULTS_ROOT, "doobh_results")
CHANNELS = [
    ("macro_change", "Macro State Change (M4)"),
    ("ewma_mean_change", "EWMA Mean Return Change (M1)"),
    ("ewma_vol_change", "EWMA Volatility Change (M2)"),
    ("pca_change", "Rolling PCA Factor Change (M3)"),
]


def load_rows():
    rows = []
    for ch, label in CHANNELS:
        df = pd.read_csv(os.path.join(OUT_DIR, ch, "doobh_comparison_train.csv"))
        rows.append((label, df.iloc[0]))
    return rows


def render(col_labels, cell_rows, title, out_path, col_widths=None):
    n_cols = len(col_labels)
    fig_w = 2.0 + (col_widths[0] if col_widths else 1.6 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, 0.75 * (len(cell_rows) + 1.6)))
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


def fmt(x, sig=3):
    return f"{x:.{sig}g}"


def build_distributional():
    col_labels = ["Channel", "Events", "Energy Score", "Marginal W1\n(avg, 15 assets)",
                  "Marginal W1\n(best asset)", "Marginal W1\n(worst asset)"]
    cell_rows = []
    for label, row in load_rows():
        cell_rows.append([
            label, int(row["n_real_events"]), fmt(row["energy_score"]),
            fmt(row["wasserstein1_marginal"]),
            f"{fmt(row['wasserstein1_marginal_best'])} ({row['wasserstein1_marginal_best_ticker']})",
            f"{fmt(row['wasserstein1_marginal_worst'])} ({row['wasserstein1_marginal_worst_ticker']})",
        ])
    render(col_labels, cell_rows, "Doob-h — Distributional Fidelity",
           os.path.join(OUT_DIR, "comparison_table_distributional.png"))


def build_portfolio():
    col_labels = ["Channel", "Events", "Variance:\nActual", "Variance:\nGenerated",
                  "Max Drawdown:\nActual", "Max Drawdown:\nGenerated",
                  "Cum Return:\nActual", "Cum Return:\nGenerated"]
    cell_rows = []
    for label, row in load_rows():
        cell_rows.append([
            label, int(row["n_real_events"]),
            fmt(row["portfolio_variance_real_median"]), fmt(row["portfolio_variance_gen_median"]),
            fmt(row["portfolio_mdd_real_median"]), fmt(row["portfolio_mdd_gen_median"]),
            fmt(row["portfolio_cumret_real_median"]), fmt(row["portfolio_cumret_gen_median"]),
        ])
    render(col_labels, cell_rows, "Doob-h — Portfolio: Actual vs Generated (median)",
           os.path.join(OUT_DIR, "comparison_table_portfolio.png"))


if __name__ == "__main__":
    build_distributional()
    build_portfolio()
