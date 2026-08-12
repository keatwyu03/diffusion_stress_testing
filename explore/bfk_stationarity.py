from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfk_results")

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr, PackageNotInstalledError

    _RPY2_IMPORT_ERROR = None
except Exception as exc:  # rpy2 itself not installed
    ro = None
    importr = None
    PackageNotInstalledError = Exception
    _RPY2_IMPORT_ERROR = exc

_npcp = None  # lazily-imported R package handle


def _require_npcp():
    """Import and cache the R npcp package via rpy2, or raise a clear
    installation error. No silent fallback to a Python approximation or a
    different stationarity test is performed.
    """
    global _npcp
    if _npcp is not None:
        return _npcp

    if _RPY2_IMPORT_ERROR is not None:
        raise ImportError(
            "rpy2 is not installed. Install it with `pip install rpy2` "
            "(and make sure R itself is installed and on PATH) before "
            "using bfk_stationarity. See the module docstring for setup "
            "commands."
        ) from _RPY2_IMPORT_ERROR

    try:
        _npcp = importr("npcp")
    except PackageNotInstalledError as exc:
        raise ImportError(
            "R package 'npcp' is not installed. From R or a shell, run:\n"
            '  Rscript -e \'install.packages("npcp", '
            'repos="https://cloud.r-project.org")\'\n'
            "See the module docstring for setup commands."
        ) from exc
    except Exception as exc:
        raise ImportError(
            "Could not load the R package 'npcp' via rpy2. This usually "
            "means R is not installed or not discoverable by rpy2. See the "
            "module docstring for setup commands."
        ) from exc

    return _npcp


def stationarity_test(
    z,
    lag: int = 3,
    n_bootstrap: int = 1000,
    seed: int = 123,
    alpha: float = 0.05,
) -> dict:
    npcp = _require_npcp()

    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(z_arr)):
        raise ValueError(
            "z contains missing or non-finite observations. "
            "Align and clean the time series before running the test."
        )

    z_clean = z_arr

    if len(z_clean) < 10:
        raise ValueError(
            f"too few observations ({len(z_clean)}) to run the test"
        )

    ro.r["set.seed"](seed)
    r_matrix = ro.r["matrix"](ro.FloatVector(z_clean), ncol=1)

    result = npcp.stDistAutocop(
        r_matrix,
        lag=lag,
        b=ro.NULL,
        pairwise=False,
        weights="parzen",
        m=5,
        N=n_bootstrap,
    )

    statistic = float(result.rx2("statistic")[0])
    global_p_value = float(result.rx2("p.value")[0])
    component_p_values = list(result.rx2("component.p.values"))
    marginal_p_value = float(component_p_values[0])
    autocopula_p_value = float(component_p_values[1])
    bandwidth = float(result.rx2("b")[0])

    return {
        "n": len(z_clean),
        "statistic": statistic,
        "global_p_value": global_p_value,
        "marginal_p_value": marginal_p_value,
        "autocopula_p_value": autocopula_p_value,
        "bandwidth": bandwidth,
        "reject": global_p_value < alpha,
    }


def stationarity_test_assets(
    residual_df: pd.DataFrame,
    lag: int = 3,
    n_bootstrap: int = 1000,
    seed: int = 123,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run stationarity_test() once per column of residual_df, sequentially.

    stDistAutocop is single-threaded and can take minutes per asset at the
    paper's N=1000 bootstrap replications on a long series; this runs one
    asset at a time in a single process/embedded R instance.
    """
    rows = []
    for asset in tqdm(residual_df.columns, desc="BFK stationarity test", unit="asset"):
        res = stationarity_test(
            residual_df[asset].to_numpy(),
            lag=lag,
            n_bootstrap=n_bootstrap,
            seed=seed,
            alpha=alpha,
        )
        rows.append({"asset": asset, **res})

    table = pd.DataFrame(rows)

    reject_holm, holm_p, _, _ = multipletests(
        table["global_p_value"].to_numpy(), alpha=alpha, method="holm"
    )
    table["holm_p_value"] = holm_p
    table["reject_raw"] = table["global_p_value"] < alpha
    table["reject_holm"] = reject_holm

    cols = [
        "asset",
        "n",
        "statistic",
        "global_p_value",
        "holm_p_value",
        "marginal_p_value",
        "autocopula_p_value",
        "bandwidth",
        "reject_raw",
        "reject_holm",
    ]
    return table[cols]


def _load_residuals(csv_path: str = None) -> pd.DataFrame:
    """Load standardized residuals (Date + one column per ticker) written by
    stationary_diagnosis.save_standardized_residuals(), in chronological
    order, ticker columns only.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "nn_standardized_macro.csv")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return df.drop(columns=["Date"])


def run_one_asset_to_file(
    asset: str,
    lag: int = 3,
    n_bootstrap: int = 1000,
    seed: int = 123,
    alpha: float = 0.05,
    window: int = None,
    results_dir: str = None,
) -> dict:
    """Run stationarity_test() for a single named asset column and write its
    result to <results_dir>/<asset>.json. Meant to be invoked as one batch
    job per asset (see submit_bfk_jobs.sh) so assets don't have to be run
    sequentially, one after another, in a single interactive session --
    stDistAutocop is single-threaded and can take tens of minutes per asset
    at the paper's N=1000 on a long series, and it returns nothing until
    that asset's entire computation is done.

    npcp's underlying C routines (cpTestAutocop / cpTestF) are cubic in the
    series length n even for the observed statistic alone (an O(n) loop over
    candidate breakpoints, each doing an O(n) x O(n) empirical-copula scan),
    so a ~6000+ observation daily-returns history is impractically slow --
    confirmed directly by reading the npcp C source, not assumed. If window
    is given, only the most recent `window` observations are tested (order
    preserved, so this stays "chronologically ordered, no gaps"); this
    changes what's being tested -- recent-window stationarity, not the full
    history -- so the tested window length is recorded in the result. The
    paper's own Monte Carlo study only goes up to T=512.
    """
    if results_dir is None:
        results_dir = _RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    residual_df = _load_residuals()
    if asset not in residual_df.columns:
        raise ValueError(f"asset {asset!r} not found in columns {list(residual_df.columns)}")

    z = residual_df[asset].to_numpy()
    if window is not None:
        z = z[-window:]

    res = stationarity_test(z, lag=lag, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)
    row = {"asset": asset, "window": window, **res}

    out_path = os.path.join(results_dir, f"{asset}.json")
    with open(out_path, "w") as f:
        json.dump(row, f, indent=2)

    print(f"[{asset}] n={row['n']} statistic={row['statistic']:.6f} "
          f"global_p_value={row['global_p_value']:.4f} reject={row['reject']} "
          f"-> wrote {out_path}")
    return row


def combine_asset_results(assets: list, alpha: float = 0.05, results_dir: str = None) -> pd.DataFrame:
    """Load each asset's <results_dir>/<asset>.json (written by
    run_one_asset_to_file, one per batch job) and combine into the same
    results table stationarity_test_assets() would produce, applying Holm's
    correction across the assets' global p-values.
    """
    if results_dir is None:
        results_dir = _RESULTS_DIR

    rows = []
    missing = []
    for asset in assets:
        path = os.path.join(results_dir, f"{asset}.json")
        if not os.path.exists(path):
            missing.append(asset)
            continue
        with open(path) as f:
            rows.append(json.load(f))

    if missing:
        raise FileNotFoundError(
            f"missing result file(s) for: {missing} in {results_dir} "
            "-- make sure every per-asset job has finished."
        )

    table = pd.DataFrame(rows)

    reject_holm, holm_p, _, _ = multipletests(
        table["global_p_value"].to_numpy(), alpha=alpha, method="holm"
    )
    table["holm_p_value"] = holm_p
    table["reject_raw"] = table["global_p_value"] < alpha
    table["reject_holm"] = reject_holm

    cols = [
        "asset", "n", "window", "statistic", "global_p_value", "holm_p_value",
        "marginal_p_value", "autocopula_p_value", "bandwidth",
        "reject_raw", "reject_holm",
    ]
    return table[[c for c in cols if c in table.columns]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    # NOTE: --n-bootstrap defaults to 500 here (not the paper's N=1000) for
    # now, to balance speed and precision. Pass --n-bootstrap 1000 explicitly
    # for a paper-faithful final run.
    # --window defaults to 512 (the largest T in the paper's own Monte Carlo
    # study) because npcp's C routines are cubic in series length -- the
    # full ~6000+ observation history is impractically slow. Pass
    # --window 0 to disable windowing and use the full series.
    p_asset = sub.add_parser("run-asset", help="run the test for one asset and save its result")
    p_asset.add_argument("asset")
    p_asset.add_argument("--lag", type=int, default=3)
    p_asset.add_argument("--n-bootstrap", type=int, default=500)
    p_asset.add_argument("--seed", type=int, default=123)
    p_asset.add_argument("--alpha", type=float, default=0.05)
    p_asset.add_argument("--window", type=int, default=512)

    p_combine = sub.add_parser("combine", help="combine saved per-asset results with Holm correction")
    p_combine.add_argument("--alpha", type=float, default=0.05)

    p_all = sub.add_parser("run-all", help="run every asset sequentially in this process (slow)")
    p_all.add_argument("--lag", type=int, default=3)
    p_all.add_argument("--n-bootstrap", type=int, default=500)
    p_all.add_argument("--seed", type=int, default=123)
    p_all.add_argument("--alpha", type=float, default=0.05)
    p_all.add_argument("--window", type=int, default=512)

    args = parser.parse_args()

    if args.mode == "run-asset":
        window = args.window if args.window > 0 else None
        run_one_asset_to_file(args.asset, lag=args.lag, n_bootstrap=args.n_bootstrap,
                               seed=args.seed, alpha=args.alpha, window=window)

    elif args.mode == "combine":
        residual_df = _load_residuals()
        results = combine_asset_results(list(residual_df.columns), alpha=args.alpha)
        print(results.to_string(index=False))
        print()
        print("global_p_value < 0.05: reject stationarity for that asset.")
        print("global_p_value >= 0.05: insufficient evidence to reject stationarity")
        print("(does not prove the residuals are stationary).")
        print("holm_p_value / reject_holm applies Holm's correction across assets.")

    elif args.mode == "run-all":
        print("=" * 72)
        print("Running BFK stationarity test on standardized residuals (per asset)")
        print("=" * 72)
        residual_df = _load_residuals()
        if args.window > 0:
            residual_df = residual_df.tail(args.window).reset_index(drop=True)
        results = stationarity_test_assets(residual_df, lag=args.lag, n_bootstrap=args.n_bootstrap,
                                            seed=args.seed, alpha=args.alpha)
        print()
        print(results.to_string(index=False))
        print()
        print("global_p_value < 0.05: reject stationarity for that asset.")
        print("global_p_value >= 0.05: insufficient evidence to reject stationarity")
        print("(does not prove the residuals are stationary).")
        print("holm_p_value / reject_holm applies Holm's correction across assets.")
