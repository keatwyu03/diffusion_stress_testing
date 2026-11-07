"""
Renders exp2_results/*.csv (already-saved Group A/B/C summary tables) as PNG
table images, matching the visual style used for Experiment 1's comparison
tables. Pure presentation -- no new numbers are computed here.

Produces three separate images (never merges causal and retrospective Group
B/C rows into one table, per the approved Experiment 2 plan's §9):
    stationarity_comparison_table.png   (Group A, all 5 methods)
    group_bc_comparison_causal.png      (Group B/C, S0/S2/S4)
    group_bc_comparison_retrospective.png (Group B/C, S1/S3 -- future-anchor leakage disclosed)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(_HERE), "main_results", "exp2_results")


def render_table(cell_rows, col_labels, title, out_path, figsize_scale=1.0):
    fig_w = 2.0 + 1.55 * len(col_labels) * figsize_scale
    fig_h = 0.6 * (len(cell_rows) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1, 2.1)
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#dbe9f6")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def fmt(x, sig=4):
    return f"{x:.{sig}g}"


def build_stationarity_table():
    df = pd.read_csv(os.path.join(OUT_DIR, "stationarity_comparison_table.csv"))
    label = {"s0": "S0 State-Space Model", "s1": "S1 Linear Interpolation", "s2": "S2 Forward-Fill",
              "s3": "S3 Cubic Spline", "s4": "S4 Tracking Regression"}
    col_labels = ["Method", "N Assets", "Stationary Frac.", "Spectral Stab.", "Variance CoV", "W1 Drift"]
    cell_rows = []
    for _, row in df.iterrows():
        cell_rows.append([
            label[row["method"]], int(row["n_assets_evaluated"]),
            fmt(row["fraction_joint_adf_kpss_stationary"]), fmt(row["mean_spectral_stability_stat"]),
            fmt(row["mean_variance_cov"]), fmt(row["mean_wasserstein_drift_scaled"]),
        ])
    render_table(cell_rows, col_labels, "Group A — Stationarity",
                 os.path.join(OUT_DIR, "stationarity_comparison_table.png"))


def build_bc_table(csv_name, out_name, title):
    df = pd.read_csv(os.path.join(OUT_DIR, csv_name))
    col_labels = ["Model", "Energy", "Marg. W1", "Var (W1)", "MDD (W1)", "P01", "P05", "P95", "P99"]
    cell_rows = []
    for _, row in df.iterrows():
        cell_rows.append([
            row["model"], fmt(row["energy_score"]), fmt(row["wasserstein1_marginal"]),
            fmt(row["portfolio_variance_w1"]), fmt(row["portfolio_mdd_w1"]),
            fmt(row["pinball_0.01"]), fmt(row["pinball_0.05"]),
            fmt(row["pinball_0.95"]), fmt(row["pinball_0.99"]),
        ])
    render_table(cell_rows, col_labels, title, os.path.join(OUT_DIR, out_name), figsize_scale=0.85)


def main():
    build_stationarity_table()
    build_bc_table("group_bc_comparison_causal.csv", "group_bc_comparison_causal.png",
                    "Group B/C — Causal Methods")
    build_bc_table("group_bc_comparison_retrospective.csv", "group_bc_comparison_retrospective.png",
                    "Group B/C — Retrospective Methods")


if __name__ == "__main__":
    main()
