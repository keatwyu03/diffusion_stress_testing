"""
Rebuilds exp1_results_change_conditioning/evaluation/comparison_table_distributional.png
with the same columns as doobh_results/comparison_table_distributional.png
(Energy Score + Marginal W1 broken into avg/best/worst asset, not just the
average) -- same visual style, same per-asset breakdown logic. Reuses the
already-saved evaluation CSV's energy_score (no retraining/resampling) and
recomputes only the per-asset W1 breakdown from the already-saved
generated_samples.npy / shared test_data.npz (that breakdown was never
saved to the CSV, only its average).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import main_exp1 as m1  # noqa: E402

_RESULTS_ROOT = os.path.join(os.path.dirname(_HERE), "main_results")
EXP_DIR = os.path.join(_RESULTS_ROOT, "exp1_results_change_conditioning")
OUT_DIR = os.path.join(EXP_DIR, "evaluation")
MODEL_KEYS = ["m0", "m1", "m2", "m3", "m4"]
MODEL_DIRNAMES = {
    "m0": "M0_unconditional", "m1": "M1_ewma_mean", "m2": "M2_ewma_volatility",
    "m3": "M3_rolling_pca", "m4": "M4_macro_ssm",
}
MODEL_LABELS = {
    "m0": "M0 Unconditional", "m1": "M1 EWMA Mean", "m2": "M2 EWMA Volatility",
    "m3": "M3 Rolling PCA", "m4": "M4 Macro SSM",
}


def per_asset_w1(X_test: np.ndarray, gen: np.ndarray):
    n_test = min(len(X_test), len(gen))
    A = X_test.shape[1]
    ws = [wasserstein_distance(X_test[:n_test, a, -1], gen[:n_test, :, a, -1].reshape(-1)) for a in range(A)]
    best_idx, worst_idx = int(np.argmin(ws)), int(np.argmax(ws))
    return float(np.mean(ws)), float(ws[best_idx]), m1.TICKERS[best_idx], float(ws[worst_idx]), m1.TICKERS[worst_idx]


def fmt(x, sig=3):
    return f"{x:.{sig}g}"


def main():
    csv_df = pd.read_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
    X_test = np.load(os.path.join(EXP_DIR, "shared", "test_data.npz"))["X_test"]
    print(f"Real test set: {X_test.shape}")

    rows = []
    for mk in MODEL_KEYS:
        gen = np.load(os.path.join(EXP_DIR, MODEL_DIRNAMES[mk], "generated_samples.npy"))
        n_test = min(len(X_test), len(gen))
        w1_avg, w1_best, best_ticker, w1_worst, worst_ticker = per_asset_w1(X_test, gen)
        row_csv = csv_df[csv_df["model"] == MODEL_LABELS[mk]]
        energy = float(row_csv["energy_score"].iloc[0])
        rows.append({
            "model": MODEL_LABELS[mk], "n_test": n_test, "energy_score": energy,
            "w1_avg": w1_avg, "w1_best": w1_best, "best_ticker": best_ticker,
            "w1_worst": w1_worst, "worst_ticker": worst_ticker,
        })
        print(f"  {MODEL_LABELS[mk]}: energy={energy:.4f}, W1 avg={w1_avg:.4g}, "
              f"best={w1_best:.4g} ({best_ticker}), worst={w1_worst:.4g} ({worst_ticker})")

    col_labels = ["Model", "Energy Score", "Marginal W1\n(avg, 15 assets)",
                  "Marginal W1\n(best asset)", "Marginal W1\n(worst asset)"]
    cell_rows = [[r["model"], fmt(r["energy_score"]), fmt(r["w1_avg"]),
                  f"{fmt(r['w1_best'])} ({r['best_ticker']})", f"{fmt(r['w1_worst'])} ({r['worst_ticker']})"]
                 for r in rows]

    fig, ax = plt.subplots(figsize=(2.0 + 2.1 * len(col_labels), 0.65 * (len(cell_rows) + 2)))
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
    ax.set_title("Experiment 1 — Distributional Fidelity", fontsize=13, pad=12)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "comparison_table_distributional.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
