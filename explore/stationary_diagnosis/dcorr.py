"""Distance-correlation diagnostic: does z_t (standardized returns) retain
dependence on m_t (latent macro state), and how much did standardization
reduce dependence relative to the raw return series r_t?

dCor(m_t, x_t) = 0 iff independent in the population, but a small SAMPLE dCor
is evidence consistent with independence, not proof of it. Reported here as a
descriptive dependence diagnostic only -- no permutation test / p-value (the
data are serially dependent, so an iid permutation null would be invalid; a
dependence-aware resampling scheme can be added separately if needed).

Per-asset:  D_j^r     = dCor(m_t, r_{t,j})            raw return series, whatever
                        the ticker columns of macro_data_new.csv currently hold
                        (raw prices today; would become log-returns if that CSV
                        is ever regenerated that way -- this script just reads
                        the column as-is)
            D_j^z     = dCor(m_t, z_{t,j})            NN-standardized residual
Joint:      D_joint^z = dCor(m_t, z_t)   where z_t = (z_{t,1}, ..., z_{t,10})

Both computed with dcor.distance_correlation (the usual/biased sample
estimator, double-centered Euclidean distance matrices, unsquared dCor) --
verified against the direct double-centering formula on a subset before use.
Univariate D_j uses the O(T log T) AVL algorithm; the joint 10-D statistic
uses the naive O(T^2) algorithm (AVL only applies to scalar series).
"""
import os

import dcor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_dir = os.path.dirname(os.path.abspath(__file__))
RAW_CSV_PATH = os.path.join(_dir, "macro_data_new.csv")
STD_CSV_PATH = os.path.join(_dir, "nn_standardized_macro.csv")
OUT_CSV = os.path.join(_dir, "dcorr_results.csv")
OUT_FIG = os.path.join(_dir, "dcorr_results.png")


def _verify_against_direct_formula(m, z_col, atol=1e-8):
    """Sanity-check dcor.distance_correlation against the explicit
    double-centering formula on a small subset, so a silent library
    convention mismatch (e.g. squared dCor) doesn't go unnoticed."""
    n = min(200, len(m))
    x = np.asarray(m[:n], dtype=float)
    y = np.asarray(z_col[:n], dtype=float)

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()
    direct = np.sqrt(dcov2 / np.sqrt(dvarx2 * dvary2))

    lib = dcor.distance_correlation(x, y, method="naive")
    assert abs(direct - lib) < atol, (
        f"dcor library result {lib} does not match direct double-centering "
        f"formula {direct} -- check estimator/convention"
    )


def main():
    raw_df = pd.read_csv(RAW_CSV_PATH, parse_dates=["Date"])
    std_df = pd.read_csv(STD_CSV_PATH, parse_dates=["Date"])
    tickers = [c for c in std_df.columns if c not in ("Date", "m_t")]

    # m_t comes from nn_standardized_macro.csv (the series z_t was actually
    # standardized against); r_{t,j} comes from macro_data_new.csv, joined on
    # Date. One common finite-observation mask across m_t, all ten r_{t,j},
    # and all ten z_{t,j}, so every row's raw/standardized pair -- and the
    # joint statistic -- are computed on exactly the same dates.
    raw_renamed = raw_df[["Date"] + tickers].rename(columns={t: f"{t}__r" for t in tickers})
    df = std_df.merge(raw_renamed, on="Date", how="inner")

    cols = ["m_t"] + tickers + [f"{t}__r" for t in tickers]
    mask = df[cols].notna().all(axis=1)
    df = df.loc[mask].sort_values("Date").reset_index(drop=True)

    m = df["m_t"].to_numpy(dtype=float)
    Z = df[tickers].to_numpy(dtype=float)                       # (T, 10) standardized
    R = df[[f"{t}__r" for t in tickers]].to_numpy(dtype=float)  # (T, 10) raw
    T = len(df)

    _verify_against_direct_formula(m, Z[:, 0])

    rows = []
    for j, t in enumerate(tickers):
        d_raw = dcor.distance_correlation(m, R[:, j], method="avl")
        d_std = dcor.distance_correlation(m, Z[:, j], method="avl")
        rows.append({
            "asset": t,
            "dcor_raw": d_raw,
            "dcor_standardized": d_std,
            "absolute_reduction": d_raw - d_std,
            "relative_reduction": (d_raw - d_std) / d_raw if abs(d_raw) > 1e-12 else float("nan"),
            "n_obs": T,
        })

    d_raw_joint = dcor.distance_correlation(m, R, method="naive")
    d_std_joint = dcor.distance_correlation(m, Z, method="naive")
    rows.append({
        "asset": "Joint 10-asset vector",
        "dcor_raw": d_raw_joint,
        "dcor_standardized": d_std_joint,
        "absolute_reduction": d_raw_joint - d_std_joint,
        "relative_reduction": (d_raw_joint - d_std_joint) / d_raw_joint if abs(d_raw_joint) > 1e-12 else float("nan"),
        "n_obs": T,
    })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_CSV, index=False)

    print(f"Distance correlation with m_t -- before (raw) vs after (standardized) -- T = {T}")
    print("=" * 78)
    print(f"  {'asset':<24}{'dcor_raw':>12}{'dcor_std':>12}{'abs_red':>12}{'rel_red':>12}")
    for r in rows:
        rel = f"{r['relative_reduction']:.1%}" if r['relative_reduction'] == r['relative_reduction'] else "n/a"
        print(f"  {r['asset']:<24}{r['dcor_raw']:>12.4f}{r['dcor_standardized']:>12.4f}"
              f"{r['absolute_reduction']:>12.4f}{rel:>12}")
    print()
    print("Descriptive diagnostic only: a small dCor is evidence consistent")
    print("with m_t-independence, not proof of it. No p-value is reported --")
    print("observations are serially dependent, so an iid permutation null")
    print("would not be valid here.")

    per_asset = results[results["asset"] != "Joint 10-asset vector"]

    fig, (ax_main, ax_joint) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [4, 1]}
    )

    x = np.arange(len(per_asset))
    for xi, (_, row) in zip(x, per_asset.iterrows()):
        ax_main.plot([xi, xi], [row["dcor_raw"], row["dcor_standardized"]],
                     color="gray", linewidth=1, zorder=1)
    ax_main.scatter(x, per_asset["dcor_raw"], color="firebrick", label="raw", zorder=3)
    ax_main.scatter(x, per_asset["dcor_standardized"], color="steelblue", label="standardized", zorder=3)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(per_asset["asset"], rotation=45)
    ax_main.set_xlabel("Asset")
    ax_main.set_ylabel("dCor(m_t, x_t,j)")
    ax_main.set_title("Per-asset distance correlation: raw vs. standardized")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(bottom=0)
    ax_main.legend()

    jx = np.array([0, 1])
    ax_joint.plot(jx, [d_raw_joint, d_std_joint], color="gray", linewidth=1, zorder=1)
    ax_joint.scatter([0], [d_raw_joint], color="firebrick", zorder=3)
    ax_joint.scatter([1], [d_std_joint], color="steelblue", zorder=3)
    ax_joint.set_xticks(jx)
    ax_joint.set_xticklabels(["raw", "std"])
    ax_joint.set_ylabel("dCor(m_t, x_t)")
    ax_joint.set_title("Joint multivariate")
    ax_joint.set_ylim(bottom=0)
    ax_joint.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Distance correlation: latent macro state vs. returns (raw vs. standardized)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nSaved {OUT_CSV}")
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
