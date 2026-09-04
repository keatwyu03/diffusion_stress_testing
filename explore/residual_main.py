"""Orchestrates the full residual-diagnostics pipeline in one call:

1. NN standardization (nn_standardize.py)      -> writes nn_standardized_macro.csv
2. ADF unit-root test per ticker                (stationary_diagnosis/adf.py)
3. Distance-correlation raw-vs-standardized     (stationary_diagnosis/dcorr.py)
4. Standardized-residuals per-asset plot        (stationary_diagnosis/res.py)
5. 252-day rolling mean/variance per asset      (this file)
6. Blockwise autocovariance stability, raw vs. standardized (this file)

Writes to explore/standardization_results/:
    residuals.png                    -- per-asset standardized residual z_t (res.py's grid)
    rolling_mean.png                 -- per-asset 252-day rolling mean of z_t
    rolling_variance.png             -- per-asset 252-day rolling variance of z_t
    adf_dcorr_tables.png             -- ADF results table + dCor results table, one figure
    autocovariance_heatmaps/{ticker}_autocovariance_stability.png
    autocovariance_heatmaps/autocovariance_dispersion_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STAT_DIAG_DIR = os.path.join(HERE, "stationary_diagnosis")
OUT_DIR = os.path.join(HERE, "standardization_results")

sys.path.insert(0, HERE)
sys.path.insert(0, STAT_DIAG_DIR)

from nn_standardize import load_data, run_ticker, save_standardized_residuals
from adf import adf_test
import dcor
import res as res_mod

MACRO_CSV = os.path.join(HERE, "macro_data_new.csv")

os.makedirs(OUT_DIR, exist_ok=True)

STD_CSV = os.path.join(OUT_DIR, "nn_standardized_macro.csv")


# ── 1. NN standardization ───────────────────────────────────────────────────

PARAMS_CSV = os.path.join(OUT_DIR, "nn_predicted_mean_variance.csv")


def step_standardize():
    dates, m, returns = load_data(csv_path=MACRO_CSV)
    print(f"standardizing {len(returns)} tickers, {len(m)} days...")

    z_by_ticker = {}
    params_cols = {}
    for ticker, r in returns.items():
        _, z, mu, sigma = run_ticker(r, m, ticker)
        z_by_ticker[ticker] = z
        params_cols[f"{ticker}_mean"] = mu.detach().cpu().numpy()
        params_cols[f"{ticker}_variance"] = (sigma ** 2).detach().cpu().numpy()

    save_standardized_residuals(z_by_ticker, dates, m=m, csv_path=STD_CSV)

    params_df = pd.DataFrame(params_cols)
    params_df.insert(0, "Date", dates.reset_index(drop=True))
    params_df.to_csv(PARAMS_CSV, index=False)
    print(f"wrote predicted mean/variance to {PARAMS_CSV}")
    print("done.")


# ── 2. ADF ───────────────────────────────────────────────────────────────────

def step_adf(std_df, tickers):
    print("running ADF tests...")
    rows = []
    for t in tickers:
        r = adf_test(std_df[t].values)
        rows.append({
            "ticker": t, "p": r["p"], "gamma": r["gamma"], "se_gamma": r["se_gamma"],
            "t_stat": r["stat"], "p_value": r["pvalue"], "n_obs": r["n_obs"],
        })
    print("done.")
    return pd.DataFrame(rows)


# ── 3. Distance correlation ──────────────────────────────────────────────────

def step_dcorr(std_df, raw_df, tickers):
    print("running distance-correlation diagnostic...")
    raw_renamed = raw_df[["Date"] + tickers].rename(columns={t: f"{t}__r" for t in tickers})
    df = std_df.merge(raw_renamed, on="Date", how="inner")

    cols = ["m_t"] + tickers + [f"{t}__r" for t in tickers]
    mask = df[cols].notna().all(axis=1)
    df = df.loc[mask].sort_values("Date").reset_index(drop=True)

    m = df["m_t"].to_numpy(dtype=float)
    Z = df[tickers].to_numpy(dtype=float)
    R = df[[f"{t}__r" for t in tickers]].to_numpy(dtype=float)

    rows = []
    for j, t in enumerate(tickers):
        d_raw = dcor.distance_correlation(m, R[:, j], method="avl")
        d_std = dcor.distance_correlation(m, Z[:, j], method="avl")
        rows.append({
            "ticker": t,
            "dcor_raw": d_raw,
            "dcor_std": d_std,
            "abs_reduction": d_raw - d_std,
            "rel_reduction": (d_raw - d_std) / d_raw if abs(d_raw) > 1e-12 else float("nan"),
        })

    d_raw_joint = dcor.distance_correlation(m, R, method="naive")
    d_std_joint = dcor.distance_correlation(m, Z, method="naive")
    rows.append({
        "ticker": "Joint (10-asset)",
        "dcor_raw": d_raw_joint,
        "dcor_std": d_std_joint,
        "abs_reduction": d_raw_joint - d_std_joint,
        "rel_reduction": (d_raw_joint - d_std_joint) / d_raw_joint if abs(d_raw_joint) > 1e-12 else float("nan"),
    })
    print("done.")
    return pd.DataFrame(rows)


# ── 4. Standardized residuals plot (delegates to res.py) ────────────────────

def step_residuals_plot(std_df, raw_df, tickers):
    print("plotting standardized residuals...")
    z = std_df.rename(columns={t: f"{t}_z" for t in tickers})
    r = raw_df[["Date"] + tickers].rename(columns={t: f"{t}_r" for t in tickers})
    df = r.merge(z, on="Date", how="inner").dropna()

    res_mod.plot_grid(
        df, tickers, col="z", color="steelblue",
        series_label="standardized residual z (post)",
        title="NN-Standardized Residuals (per asset)",
        save_path=os.path.join(OUT_DIR, "residuals.png"),
    )
    plt.close("all")
    print("done.")


# ── 5. 252-day rolling mean/variance grid ────────────────────────────────────

def _plot_rolling_grid(std_df, tickers, window, series_fn, color, ylabel, hline,
                        title, save_path, n_cols=2):
    n_rows = (len(tickers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 3.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(tickers):]:
        ax.axis("off")

    for ax, ticker in zip(axes, tickers):
        series = series_fn(std_df[ticker])
        ax.plot(std_df["Date"], series, linewidth=0.8, color=color, label=ylabel)
        ax.axhline(hline, color=color, linewidth=0.6, alpha=0.4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_title(ticker, fontsize=10, fontweight="bold", loc="left")

        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper left")

    line1, line2 = res_mod.describe_cond_bucket()
    fig.suptitle(f"{title}\n{line1}\n{line2}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def step_rolling_mean(std_df, tickers, window=252):
    print(f"plotting {window}-day rolling mean...")
    _plot_rolling_grid(
        std_df, tickers, window, lambda z: z.rolling(window).mean(),
        color="steelblue", ylabel="rolling mean", hline=0,
        title=f"{window}-Day Rolling Mean of Standardized Residuals (per asset)",
        save_path=os.path.join(OUT_DIR, "rolling_mean.png"),
    )


def step_rolling_variance(std_df, tickers, window=252):
    print(f"plotting {window}-day rolling variance...")
    _plot_rolling_grid(
        std_df, tickers, window, lambda z: z.rolling(window).var(),
        color="firebrick", ylabel="rolling variance", hline=1,
        title=f"{window}-Day Rolling Variance of Standardized Residuals (per asset)",
        save_path=os.path.join(OUT_DIR, "rolling_variance.png"),
    )


# ── 5b. Rolling-moment stabilization: standardized vs. 3 baselines ──────────
#
# Two SEPARATE relative measures (mean, variance) of how much standardization
# changed rolling-window instability relative to a given baseline series
# (raw price, differenced price, log return). These are descriptive relative
# reductions computed from overlapping rolling windows — NOT independent-
# observation confidence intervals, and NOT formal stationarity tests.

def _rolling_mean_instability(x, window):
    """M_W(x): mean_t |(-hat mu_{t,W}(x) - hat mu(x)) / hat sigma(x)|, both
    hat mu(x) and hat sigma(x) full-sample (same baseline used for both
    series being compared — never 0 as a fixed target)."""
    x = pd.Series(x).astype(float)
    mu_full = x.mean()
    sigma_full = x.std(ddof=0)
    if sigma_full == 0 or not np.isfinite(sigma_full):
        return float("nan")
    roll_mean = x.rolling(window).mean().dropna()
    return (roll_mean - mu_full).abs().div(sigma_full).mean()


def _rolling_variance_instability(x, window):
    """V_W(x): mean_t |log(hat v_{t,W}(x) / hat v(x))|, hat v(x) the
    full-sample variance of x itself (never 1 as a fixed target)."""
    x = pd.Series(x).astype(float)
    var_full = x.var(ddof=0)
    if var_full <= 0 or not np.isfinite(var_full):
        return float("nan")
    roll_var = x.rolling(window).var().dropna()
    roll_var = roll_var[roll_var > 0]
    return np.log(roll_var / var_full).abs().mean()


def _pct_reduction(baseline_val, standardized_val):
    if baseline_val is None or not np.isfinite(baseline_val) or baseline_val == 0:
        return float("nan")
    return 100 * (1 - standardized_val / baseline_val)


def step_stabilization_table(std_df, tickers, window=252):
    """For each asset x each of 3 baselines (raw price, differenced price,
    log return), compute M_W/V_W for the baseline and for z_t (date-aligned,
    same window), then R_mu and R_v. Prints a concise table and renders it as
    an image — no CSV, per explicit instruction."""
    print(f"computing rolling-moment stabilization vs. 3 baselines (window={window})...")

    cfg_tickers = tickers
    start_date = None
    try:
        sys.path.insert(0, os.path.dirname(HERE))
        from config import get_default_config
        _cfg = get_default_config()
        start_date = _cfg.data.start_date
    except Exception:
        pass

    import yfinance as yf
    print(f"  fetching raw price history for {cfg_tickers} from yfinance...")
    px = yf.download(cfg_tickers, start=start_date, auto_adjust=True)["Close"]
    px = px[cfg_tickers].dropna(how="all")

    baselines = {
        "raw_price": px,
        "diff_price": px.diff().dropna(how="all"),
        "log_return": np.log(px / px.shift(1)).dropna(how="all"),
    }

    z_indexed = std_df.set_index("Date")

    rows = []
    for ticker in tickers:
        z_full = z_indexed[ticker].dropna()

        for baseline_name, base_df in baselines.items():
            if ticker not in base_df.columns:
                continue
            base_series = base_df[ticker].dropna()

            joined = pd.concat(
                {"r": base_series, "z": z_full}, axis=1
            ).dropna()
            if len(joined) <= window:
                continue

            r_aligned = joined["r"]
            z_aligned = joined["z"]

            M_r = _rolling_mean_instability(r_aligned, window)
            M_z = _rolling_mean_instability(z_aligned, window)
            R_mu = _pct_reduction(M_r, M_z)

            V_r = _rolling_variance_instability(r_aligned, window)
            V_z = _rolling_variance_instability(z_aligned, window)
            R_v = _pct_reduction(V_r, V_z)

            rows.append({
                "asset": ticker,
                "baseline": baseline_name,
                "mean_instability_baseline": M_r,
                "mean_instability_standardized": M_z,
                "mean_stabilization_pct": R_mu,
                "variance_instability_baseline": V_r,
                "variance_instability_standardized": V_z,
                "variance_stabilization_pct": R_v,
            })

    summary_df = pd.DataFrame(rows)

    print(f"\n{'asset':<7}{'baseline':<12}{'R_mu (%)':>12}{'R_v (%)':>12}")
    for _, row in summary_df.iterrows():
        r_mu_str = f"{row['mean_stabilization_pct']:+.1f}"
        r_v_str = f"{row['variance_stabilization_pct']:+.1f}"
        flag = ""
        if row["mean_stabilization_pct"] < 0:
            flag += " [mean LESS stable]"
        if row["variance_stabilization_pct"] < 0:
            flag += " [var LESS stable]"
        print(f"{row['asset']:<7}{row['baseline']:<12}{r_mu_str:>12}{r_v_str:>12}{flag}")

    _render_stabilization_image(summary_df, tickers, window)
    print("done.")
    return summary_df


def _render_stabilization_image(summary_df, tickers, window):
    """One table per baseline (raw_price, diff_price, log_return), placed
    SIDE BY SIDE (not stacked) — columns: asset, R_mu (%), R_v (%).
    matplotlib's ax.table(loc="center") renders at a fixed size independent
    of its parent axes' height, so stacking tables vertically in separate
    axes leaves large blank gaps; side-by-side avoids that since every axes
    gets the same (correctly sized) height."""
    baseline_order = ["raw_price", "diff_price", "log_return"]
    baseline_titles = {
        "raw_price": "Raw Price Levels",
        "diff_price": "Differenced Prices",
        "log_return": "Log Returns",
    }
    present = [b for b in baseline_order if b in summary_df["baseline"].unique()]
    n_assets = summary_df["asset"].nunique()

    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 0.35 * n_assets + 1.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, baseline in zip(axes, present):
        ax.axis("off")
        sub = summary_df[summary_df["baseline"] == baseline].set_index("asset")
        sub = sub.loc[[t for t in tickers if t in sub.index]]

        cell_text = [[t, f"{row.mean_stabilization_pct:+.1f}", f"{row.variance_stabilization_pct:+.1f}"]
                     for t, row in sub.iterrows()]
        tbl = ax.table(cellText=cell_text,
                        colLabels=["asset", "μ Deviation Reduction (%)", "σ² Deviation Reduction (%)"],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

        for i, row in enumerate(sub.itertuples(), start=1):
            if row.mean_stabilization_pct < 0:
                tbl[(i, 1)].set_facecolor("#fdd")
            if row.variance_stabilization_pct < 0:
                tbl[(i, 2)].set_facecolor("#fdd")

        ax.set_title(baseline_titles[baseline], fontsize=11, fontweight="bold", pad=14)

    fig.suptitle(
        f"Rolling-Moment Stabilization vs. Baseline Returns (window={window}d)\n"
        "Descriptive relative reduction in rolling-window instability — not a formal stationarity test.\n"
        "Positive = standardized residuals more stable than baseline; negative (shaded) = less stable.",
        fontsize=11.5, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_path = os.path.join(OUT_DIR, "rolling_moment_stabilization.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── 6. Blockwise autocovariance stability (raw vs. standardized) ────────────
#
# Splits each series into consecutive non-overlapping blocks of block_length
# trading days (final partial block dropped), computes the sample
# autocovariance at lags 1..max_lag within each block (centered on that
# block's own mean), and normalizes by the FULL-SERIES variance of that same
# series (raw variance for raw, standardized variance for standardized) — not
# the local block variance, since local-variance drift is what the rolling
# mean/variance plots above already diagnose. A column (fixed lag) that stays
# a uniform color down the rows means gamma(k) is stable over time, i.e. the
# autocovariance structure is closer to second-order stationary. A reduction
# in dispersion after standardization is evidence of MORE STABLE
# autocovariance, not proof of stationarity on its own.

def _blockwise_normalized_autocov(x, block_length=252, max_lag=10):
    """x: 1-D np.ndarray. Returns (gamma_tilde, n_blocks) where gamma_tilde
    has shape (n_blocks, max_lag) — row b, col k-1 is gamma_tilde_b(k)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_blocks = n // block_length
    var_full = x.var(ddof=0)

    gamma_tilde = np.full((n_blocks, max_lag), np.nan)
    for b in range(n_blocks):
        block = x[b * block_length : (b + 1) * block_length]
        assert len(block) == block_length
        centered = block - block.mean()
        for k in range(1, max_lag + 1):
            gamma_hat_k = (centered[k:] * centered[:-k]).sum() / (block_length - k)
            gamma_tilde[b, k - 1] = gamma_hat_k / var_full if var_full > 0 else np.nan

    return gamma_tilde, n_blocks


def _block_end_labels(dates, block_length, n_blocks):
    """One label per block: the year of its last date (block-ending year)."""
    labels = []
    for b in range(n_blocks):
        end_idx = (b + 1) * block_length - 1
        labels.append(str(pd.Timestamp(dates.iloc[end_idx]).year))
    return labels


def _dispersion(gamma_tilde):
    """D_k = SD_b(gamma_tilde_b(k)) per lag, and D_ACov = sqrt(mean(D_k^2))."""
    D_k = gamma_tilde.std(axis=0, ddof=0)
    D_acov = np.sqrt(np.mean(D_k ** 2))
    return D_k, D_acov


def step_autocov_heatmaps(std_df, raw_df, tickers, block_length=252, max_lag=10):
    """For each ticker: side-by-side heatmaps (raw vs. standardized) of the
    blockwise-normalized autocovariance gamma_tilde_b(k), rows = time blocks,
    cols = lags 1..max_lag. Also writes a summary CSV of the per-lag and
    aggregate dispersion D_ACov for both series, and prints a before/after
    comparison. Reuses res_mod.load_joined() for the exact same date-aligned
    raw/standardized panel the other diagnostics use — no refitting."""
    print(f"computing blockwise autocovariance stability "
          f"(block_length={block_length}, max_lag={max_lag})...")

    out_dir = os.path.join(OUT_DIR, "autocovariance_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    df, _tickers = res_mod.load_joined(resid_csv=STD_CSV, macro_csv=MACRO_CSV)
    assert set(tickers) <= set(_tickers), "ticker mismatch between std_df/raw_df and res_mod.load_joined()"
    dates = df["Date"]

    summary_rows = []
    lag_cols = [f"D_{k}" for k in range(1, max_lag + 1)]

    for ticker in tickers:
        raw_series = df[f"{ticker}_r"].to_numpy(dtype=float)
        std_series = df[f"{ticker}_z"].to_numpy(dtype=float)

        gamma_raw, n_blocks = _blockwise_normalized_autocov(raw_series, block_length, max_lag)
        gamma_std, n_blocks_std = _blockwise_normalized_autocov(std_series, block_length, max_lag)
        assert n_blocks == n_blocks_std, f"{ticker}: raw/standardized block counts differ"
        assert gamma_raw.shape == gamma_std.shape
        assert gamma_raw.shape[1] == max_lag
        assert np.isfinite(gamma_raw).all() and np.isfinite(gamma_std).all(), \
            f"{ticker}: non-finite values in blockwise autocovariance"

        row_labels = _block_end_labels(dates, block_length, n_blocks)

        D_k_raw, D_acov_raw = _dispersion(gamma_raw)
        D_k_std, D_acov_std = _dispersion(gamma_std)
        pct_reduction = 100 * (1 - D_acov_std / D_acov_raw) if D_acov_raw > 0 else float("nan")

        summary_rows.append({"asset": ticker, "series": "raw",
                              **dict(zip(lag_cols, D_k_raw)),
                              "D_ACov": D_acov_raw, "n_blocks": n_blocks})
        summary_rows.append({"asset": ticker, "series": "standardized",
                              **dict(zip(lag_cols, D_k_std)),
                              "D_ACov": D_acov_std, "n_blocks": n_blocks})

        print(f"  {ticker:<6} D_ACov raw={D_acov_raw:.4f}  standardized={D_acov_std:.4f}  "
              f"({pct_reduction:+.1f}% change; more stable autocovariance if positive, "
              f"NOT proof of stationarity)")

        vmax = max(np.abs(gamma_raw).max(), np.abs(gamma_std).max())
        vmax = vmax if vmax > 0 else 1e-8

        fig, (ax_raw, ax_std) = plt.subplots(1, 2, figsize=(11, 0.32 * n_blocks + 2.5), sharey=True)

        im0 = ax_raw.imshow(gamma_raw, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_raw.set_title("Raw returns (pre-standardization)", fontsize=10.5, fontweight="bold")
        ax_raw.set_xlabel("Lag k")
        ax_raw.set_ylabel(f"Block (ending year, {block_length}d each)")
        ax_raw.set_xticks(range(max_lag))
        ax_raw.set_xticklabels(range(1, max_lag + 1))
        ax_raw.set_yticks(range(n_blocks))
        ax_raw.set_yticklabels(row_labels, fontsize=7)

        im1 = ax_std.imshow(gamma_std, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_std.set_title("Standardized residuals (post-standardization)", fontsize=10.5, fontweight="bold")
        ax_std.set_xlabel("Lag k")
        ax_std.set_xticks(range(max_lag))
        ax_std.set_xticklabels(range(1, max_lag + 1))

        # cell annotations only when few enough blocks/lags to stay readable
        if n_blocks * max_lag <= 120:
            for ax, mat in ((ax_raw, gamma_raw), (ax_std, gamma_std)):
                for bi in range(n_blocks):
                    for ki in range(max_lag):
                        v = mat[bi, ki]
                        color = "white" if abs(v) > 0.6 * vmax else "black"
                        ax.text(ki, bi, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color=color)

        cbar = fig.colorbar(im0, ax=[ax_raw, ax_std], fraction=0.035, pad=0.03)
        cbar.set_label("Normalized autocovariance", fontsize=9)

        fig.suptitle(
            f"{ticker} — Blockwise Autocovariance Stability\n"
            f"D_ACov: raw={D_acov_raw:.4f}  ->  standardized={D_acov_std:.4f}  "
            f"({pct_reduction:+.1f}% change in dispersion, not a stationarity proof)",
            fontsize=12, fontweight="bold"
        )
        # per-asset validation (before closing the figure)
        assert im0.get_clim() == im1.get_clim(), f"{ticker}: color limits differ between panels"

        save_path = os.path.join(out_dir, f"{ticker}_autocovariance_stability.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    assert (summary_df.groupby("asset").size() == 2).all(), "expected exactly 2 rows (raw, standardized) per asset"
    summary_path = os.path.join(out_dir, "autocovariance_dispersion_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {len(tickers)} heatmaps to {out_dir}")

    return summary_df


# ── Combined ADF + dCor table image ─────────────────────────────────────────

def make_tables_image(adf_df, dcorr_df):
    fig, (ax_adf, ax_dcor) = plt.subplots(2, 1, figsize=(11, 0.5 * (len(adf_df) + len(dcorr_df)) + 3))

    for ax in (ax_adf, ax_dcor):
        ax.axis("off")

    adf_cols = ["ticker", "p", "gamma", "se_gamma", "t_stat", "p_value", "n_obs"]
    adf_fmt = adf_df[adf_cols].copy()
    for c in ["gamma", "se_gamma", "t_stat"]:
        adf_fmt[c] = adf_fmt[c].map(lambda v: f"{v:.4f}")
    adf_fmt["p_value"] = adf_df["p_value"].map(lambda v: f"{v:.4f}")
    tbl1 = ax_adf.table(cellText=adf_fmt.values, colLabels=adf_cols,
                         loc="center", cellLoc="center")
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(8)
    tbl1.scale(1, 1.4)
    ax_adf.set_title("ADF Test (constant + trend, AIC-selected lag) — standardized residuals z_t",
                      fontsize=11, fontweight="bold", pad=14)

    dcor_cols = ["ticker", "dcor_raw", "dcor_std", "abs_reduction", "rel_reduction"]
    dcor_fmt = dcorr_df[dcor_cols].copy()
    for c in ["dcor_raw", "dcor_std", "abs_reduction"]:
        dcor_fmt[c] = dcorr_df[c].map(lambda v: f"{v:.4f}")
    dcor_fmt["rel_reduction"] = dcorr_df["rel_reduction"].map(
        lambda v: f"{v:.1%}" if v == v else "n/a")
    tbl2 = ax_dcor.table(cellText=dcor_fmt.values, colLabels=dcor_cols,
                          loc="center", cellLoc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8)
    tbl2.scale(1, 1.4)
    ax_dcor.set_title("Distance Correlation with m_t — raw vs. standardized returns",
                       fontsize=11, fontweight="bold", pad=14)

    fig.tight_layout()
    save_path = os.path.join(OUT_DIR, "adf_dcorr_tables.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Residual diagnostics pipeline. With no flags, runs everything "
                     "(standardize + all 4 images). With one or more flags, runs only "
                     "the requested step(s), reusing the existing "
                     "nn_standardized_macro.csv unless --standardize is also passed.")
    parser.add_argument("--standardize", action="store_true",
                        help="(re)fit the NN standardization and overwrite nn_standardized_macro.csv")
    parser.add_argument("--adf-dcorr", action="store_true",
                        help="run ADF + distance-correlation and write adf_dcorr_tables.png")
    parser.add_argument("--plot-res", action="store_true",
                        help="write residuals.png (per-asset standardized residual plot)")
    parser.add_argument("--plot-means", action="store_true",
                        help="write rolling_mean.png (100-day rolling mean per asset)")
    parser.add_argument("--plot-vars", action="store_true",
                        help="write rolling_variance.png (252-day rolling variance per asset)")
    parser.add_argument("--autocov", action="store_true",
                        help="write autocovariance_heatmaps/ (blockwise autocovariance "
                             "stability, raw vs. standardized, per asset + summary CSV)")
    parser.add_argument("--block-length", type=int, default=252,
                        help="block length in trading days for --autocov (default: 252)")
    parser.add_argument("--max-lag", type=int, default=10,
                        help="max autocovariance lag for --autocov (default: 10)")
    parser.add_argument("--stabilization", action="store_true",
                        help="write rolling_moment_stabilization.png (relative reduction "
                             "in rolling-mean and rolling-variance instability vs. raw "
                             "price, differenced price, and log-return baselines)")
    parser.add_argument("--stabilization-window", type=int, default=252,
                        help="rolling window (days) for --stabilization (default: 252)")
    args = parser.parse_args()

    any_flag = any([args.standardize, args.adf_dcorr, args.plot_res,
                    args.plot_means, args.plot_vars, args.autocov, args.stabilization])
    run_all = not any_flag

    if args.standardize or run_all:
        step_standardize()

    std_df = pd.read_csv(STD_CSV, parse_dates=["Date"])
    raw_df = pd.read_csv(MACRO_CSV, parse_dates=["Date"])
    tickers = [c for c in std_df.columns if c not in ("Date", "m_t")]

    if args.adf_dcorr or run_all:
        adf_df = step_adf(std_df, tickers)
        dcorr_df = step_dcorr(std_df, raw_df, tickers)
        make_tables_image(adf_df, dcorr_df)

    if args.plot_res or run_all:
        step_residuals_plot(std_df, raw_df, tickers)

    if args.plot_means or run_all:
        step_rolling_mean(std_df, tickers)

    if args.plot_vars or run_all:
        step_rolling_variance(std_df, tickers)

    if args.autocov or run_all:
        step_autocov_heatmaps(std_df, raw_df, tickers,
                               block_length=args.block_length, max_lag=args.max_lag)

    if args.stabilization or run_all:
        step_stabilization_table(std_df, tickers, window=args.stabilization_window)

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
