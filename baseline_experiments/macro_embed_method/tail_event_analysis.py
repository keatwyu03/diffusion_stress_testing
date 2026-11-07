"""
Restricts the already-generated S0/S2 samples to each method's own top-7.5%
"upper_change" windows -- same event definition as the production Doob-h
pipeline (config.hfunction: event_type="upper_change", event_threshold=0.075,
metric = Z_end - Z_start over the window, threshold = train-only quantile).
No retraining/resampling: this is a re-analysis of saved generated_samples.npy
restricted to a tail-event subset of the test windows, plus a marginal plot
and a comparison table for that subset.

Per-method event windows, NOT a shared calendar subset (same convention as
the original Doob-h channel comparison): S0's events are the test windows
where S0's OWN state had its biggest 10-day upward move; S2's events are
where S2's OWN (forward-filled) state had its biggest 10-day upward move.
These can be different calendar windows for each method -- that's expected,
not a bug (each channel is being judged on its own notion of a "big event").
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
_BASELINE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _BASELINE)

import common  # noqa: E402
from diffusion_adapters import build_cond_feature, per_window_cond_values, common_valid_window_mask  # noqa: E402
from exp_evaluation import energy_score  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(_HERE), "main_results", "exp2_results")
EVENT_WINDOW = 10        # == cfg.data.seq_len (matches config.hfunction.event_window convention)
EVENT_THRESHOLD = 0.075  # top 7.5%, same as config.hfunction.event_threshold
METHODS = ["s0", "s2"]
COLORS = {"s0": "#1f77b4", "s2": "#d62728"}


def raw_state_values(method, manifest):
    state_csv = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[method], "daily_state.csv")
    s = pd.read_csv(state_csv, index_col=0, parse_dates=True)["daily_state"]
    return s.reindex(manifest["df"].index).to_numpy(dtype=np.float64)


def window_z_start_end(values, seq_len, starts):
    return values[starts], values[starts + seq_len - 1]


def event_mask_and_threshold(method, manifest, mask, test_starts):
    """Threshold fit on the common-valid TRAIN pool for this method's own
    state; event mask applied to this method's test windows (test_starts)."""
    values = raw_state_values(method, manifest)
    seq_len = manifest["seq_len"]
    split_idx, test_start_idx = manifest["split_idx"], manifest["test_start_idx"]

    train_starts = manifest["starts"][:split_idx][mask[:split_idx]]
    z_start_tr, z_end_tr = window_z_start_end(values, seq_len, train_starts)
    metric_tr = z_end_tr - z_start_tr
    n_nan = int(np.isnan(metric_tr).sum())
    threshold = float(np.nanquantile(metric_tr, 1.0 - EVENT_THRESHOLD))

    z_start_te, z_end_te = window_z_start_end(values, seq_len, test_starts)
    metric_te = z_end_te - z_start_te
    event = metric_te >= threshold  # NaN comparisons are False -- undefined-state windows just don't qualify
    print(f"  [{method}] {n_nan}/{len(metric_tr)} train windows had NaN Z_start/Z_end (own-state gaps), "
          f"{int(np.isnan(metric_te).sum())}/{len(metric_te)} test windows likewise excluded from eligibility")
    return event, threshold, metric_tr[~np.isnan(metric_tr)]


def main():
    cfg = common.get_default_config()
    manifest = common.get_window_manifest(cfg)
    seq_len = manifest["seq_len"]

    all_methods = common.METHODS
    all_cond = {m: per_window_cond_values(build_cond_feature(m, manifest, seq_len), manifest) for m in all_methods}
    mask = common_valid_window_mask(all_cond)
    test_start_idx = manifest["test_start_idx"]
    test_mask = mask[test_start_idx:]
    test_starts = manifest["starts"][test_start_idx:][test_mask]

    real = np.load(os.path.join(OUT_DIR, common.METHOD_DIRNAMES["s0"], "diffusion", "test_data.npz"))["X_test"]
    gen = {m: np.load(os.path.join(OUT_DIR, common.METHOD_DIRNAMES[m], "diffusion", "generated_samples.npy"))
           for m in METHODS}
    n = min(len(real), len(test_starts), *[gen[m].shape[0] for m in METHODS])
    real, test_starts = real[:n], test_starts[:n]

    events, thresholds, rows = {}, {}, []
    for m in METHODS:
        ev, thr, metric_tr = event_mask_and_threshold(m, manifest, mask, test_starts)
        events[m] = ev
        thresholds[m] = thr
        idx = np.where(ev)[0]
        es = np.mean([energy_score(real[i], gen[m][i]) for i in idx]) if len(idx) else float("nan")
        A = real.shape[1]
        mw1 = np.mean([wasserstein_distance(real[idx, a, -1], gen[m][idx][:, :, a, -1].reshape(-1))
                        for a in range(A)]) if len(idx) else float("nan")
        print(f"{m.upper()}: train-fit threshold={thr:.4g} (top {EVENT_THRESHOLD:.1%} of {len(metric_tr)} "
              f"train windows' Z_end-Z_start), {len(idx)}/{n} test windows qualify as events, "
              f"energy={es:.4f}, marg.W1={mw1:.5f}")
        rows.append({"method": common.METHOD_LABELS[m], "n_events": len(idx), "threshold": thr,
                     "energy_score": es, "marginal_w1": mw1})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "tail_event_comparison.csv"), index=False)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'tail_event_comparison.csv')}")

    # Table image
    fig, ax = plt.subplots(figsize=(8, 0.6 * (len(rows) + 2)))
    ax.axis("off")
    col_labels = ["Method", "N Events", "Energy Score", "Marginal W1"]
    cell_rows = [[r["method"], r["n_events"], f"{r['energy_score']:.4f}", f"{r['marginal_w1']:.5f}"] for r in rows]
    tbl = ax.table(cellText=cell_rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    tbl.scale(1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#dbe9f6")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Top 7.5% Upper-Change Events — S0 vs S2", fontsize=12, pad=12)
    plt.tight_layout()
    table_path = os.path.join(OUT_DIR, "tail_event_comparison.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {table_path}")

    # Marginal plot restricted to each method's own event windows
    n_assets = len(common.TICKERS)
    ncols = 5
    nrows = int(np.ceil(n_assets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = axes.ravel()
    for a, ticker in enumerate(common.TICKERS):
        ax = axes[a]
        for m in METHODS:
            idx = np.where(events[m])[0]
            if len(idx) < 5:
                continue
            r_vals = real[idx, a, -1]
            g_vals = gen[m][idx][:, :, a, -1].reshape(-1)
            lo, hi = min(r_vals.min(), g_vals.min()), max(r_vals.max(), g_vals.max())
            bins = np.linspace(lo, hi, 25)
            ax.hist(r_vals, bins=bins, density=True, histtype="step", color=COLORS[m],
                    linewidth=2, linestyle="-", label=f"{m.upper()} real (N={len(idx)})")
            ax.hist(g_vals, bins=bins, density=True, histtype="step", color=COLORS[m],
                    linewidth=1.2, linestyle="--", label=f"{m.upper()} gen")
        ax.set_title(ticker, fontsize=10)
        ax.tick_params(labelsize=7)
        if a == 0:
            ax.legend(fontsize=6, loc="upper right")
    for a in range(n_assets, len(axes)):
        axes[a].axis("off")
    fig.suptitle("Last-Day Marginals on Each Method's Own Top-7.5% Upper-Change Events", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    marg_path = os.path.join(OUT_DIR, "tail_event_marginals_s0_s2.png")
    plt.savefig(marg_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {marg_path}")


if __name__ == "__main__":
    main()
