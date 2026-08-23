import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EXPLORE_DIR)
sys.path.insert(0, REPO_ROOT)

RESID_CSV = os.path.join(EXPLORE_DIR, "nn_standardized_macro.csv")
MACRO_CSV = os.path.join(EXPLORE_DIR, "macro_data_new.csv")


# Hardcoded facts about what explore/import_data.py and
# latent_state_estimation/macro_importer.py actually write — not in config,
# since neither script exposes the transform as a flag. Update these strings
# if that ever changes (e.g. import_data.py starts writing log-returns).
PRICE_TRANSFORM = "log-returns of adjusted prices"
MONTHLY_MACRO_TRANSFORM = "raw levels (no transform)"


def describe_cond_bucket():
    """(two-line label) describing the CURRENT config's latent-state bucket
    (growth/inflation/vol vars + method) and the price/macro transform in use
    — dynamic where it can be (config), hardcoded where it can't (the
    transform, which isn't a config flag) — so the plot title always matches
    what actually produced m_t and the ticker columns."""
    from config import get_default_config

    cfg = get_default_config()
    active = [group for group, sel in (("growth", cfg.data.growth_vars),
                                       ("inflation", cfg.data.inflation_vars),
                                       ("vol", cfg.data.vol_vars))
              if sel]
    bucket_lbl = " + ".join(active)

    line1 = f"m_t method: {cfg.data.latent_method},  bucket: {bucket_lbl}"
    line2 = f"prices: {PRICE_TRANSFORM},  monthly macro: {MONTHLY_MACRO_TRANSFORM}"
    return line1, line2


def load_joined(resid_csv=RESID_CSV, macro_csv=MACRO_CSV):
    z = pd.read_csv(resid_csv, parse_dates=["Date"])
    r = pd.read_csv(macro_csv, parse_dates=["Date"])

    tickers = [c for c in z.columns if c not in ("Date", "m_t")]
    z = z.rename(columns={t: f"{t}_z" for t in tickers})
    r = r[["Date"] + tickers].rename(columns={t: f"{t}_r" for t in tickers})

    df = r.merge(z, on="Date", how="inner").dropna()
    return df, tickers


def plot_grid(df, tickers, col, color, series_label, title, n_cols=2, save_path=None):
    n_rows = (len(tickers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 3.2 * n_rows),
                              sharex=True)
    axes = axes.ravel()
    for ax in axes[len(tickers):]:
        ax.axis("off")

    for ax, ticker in zip(axes, tickers):
        series = df[f"{ticker}_{col}"]
        ax.plot(df["Date"], series, linewidth=0.6, color=color,
                 label=series_label)
        ax.axhline(0, color=color, linewidth=0.8, alpha=0.4)
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_title(ticker, fontsize=10, fontweight="bold", loc="left")

        mean, var = series.mean(), series.var()
        ax.text(0.99, 0.97, f"mean={mean:+.4f}\nvar={var:.4f}",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=color, alpha=0.85, linewidth=0.7))

        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper left")

    line1, line2 = describe_cond_bucket()
    fig.suptitle(f"{title}\n{line1}\n{line2}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    return fig


if __name__ == "__main__":
    df, tickers = load_joined()
    print(f"{len(df)} joined rows, {df['Date'].min().date()} to {df['Date'].max().date()}, "
          f"{len(tickers)} tickers: {tickers}")

    plot_grid(df, tickers, col="z", color="steelblue",
               series_label="standardized residual z (post)",
               title="NN-Standardized Residuals (per asset)",
               save_path=os.path.join(HERE, "nn_standardized_returns.png"))

    plot_grid(df, tickers, col="r", color="darkorange",
               series_label="raw return r (pre-standardization)",
               title="Raw Returns (per asset)",
               save_path=os.path.join(HERE, "raw_returns.png"))

    plt.show()
