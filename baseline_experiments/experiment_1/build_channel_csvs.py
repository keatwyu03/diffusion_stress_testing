"""
Build 4 per-channel variants of data_for_diffusion.csv for the generalized
Doob-h rare-event experiment (Experiment 2).

Each output CSV is BYTE-IDENTICAL to data_for_diffusion.csv in its Date index
and 15 asset columns -- only column 0 (the exogenous conditioning channel,
originally the raw macro state m_t) is replaced with one of:
    macro_change      (existing: |m_t[end] - m_t[end-window+1]|)
    ewma_mean_change   |EWMA mean[end] - EWMA mean[end-window+1]|
    ewma_vol_change     |EWMA vol[end]  - EWMA vol[end-window+1]|
    pca_change          |PCA factor[end] - PCA factor[end-window+1]|

These are the EXACT series already computed and verified in
experiment_1/main_exp1.py's compute_conditioning_variables (from RAW returns
in explore/macro_data_new.csv), reused here rather than re-derived -- this
mirrors how main.py's own existing macro channel already works: m_t itself is
never NN-residualized, only the asset returns are. A "raw" conditioning
channel paired with DataProcessor's processed asset returns is the existing
pipeline's own convention, not a new inconsistency.

data/data_processor.py picks the conditioning column POSITIONALLY
(df.columns[0]), so main.py, data_processor.py, hfunction_direct.py, and
conditional_generator.py need ZERO code changes to consume these.

Usage:
    python build_channel_csvs.py
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import main_exp1 as m1  # noqa: E402

PROD_CSV = os.path.join(_ROOT, "explore", "standardization_results", "data_for_diffusion.csv")
OUT_DIR = os.path.join(_ROOT, "explore", "standardization_results")

CHANNELS = ["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"]


def main():
    cfg = m1.get_default_config()

    df_raw, macro_col = m1.load_raw_data(cfg)
    cond_df = m1.compute_conditioning_variables(
        df_raw, macro_col, L_ewma=10, L_pca=10, alpha=59 / 61, macro_change_window=10,
    )

    prod_df = pd.read_csv(PROD_CSV, parse_dates=["Date"]).set_index("Date")
    prod_macro_col = prod_df.columns[0]
    asset_cols = list(prod_df.columns[1:])
    print(f"Production CSV: {PROD_CSV}")
    print(f"  {prod_df.shape[0]} rows, date range {prod_df.index.min()} -> {prod_df.index.max()}")
    print(f"  conditioning col (to be replaced): {prod_macro_col!r}, asset cols: {len(asset_cols)}")

    missing_dates = prod_df.index.difference(cond_df.index)
    if len(missing_dates) > 0:
        # data_for_diffusion.csv currently extends a bit past cfg.data.end_date (the
        # cutoff main_exp1.py's load_raw_data applies), so the conditioning series has
        # no value for these trailing dates. DataProcessor already tolerates NaN gaps
        # in the conditioning column by design (its own load_returns() docstring: "macro
        # series is sparse") -- those few dates simply won't form valid event windows,
        # same as macro_data_new.csv's own pre-existing scattered NaNs elsewhere.
        print(f"[warn] {len(missing_dates)} dates in {PROD_CSV} fall outside the "
              f"conditioning series' range (likely beyond cfg.data.end_date) -- channel "
              f"value will be NaN for these dates: {missing_dates.min()} -> {missing_dates.max()}")

    # macro_change is the only channel with real missing-data gaps (the raw
    # macro state is genuinely unobserved on many days -- EWMA/PCA are
    # derived purely from complete stock-return data, so they're valid
    # everywhere). main.py's own, unmodified validity check
    # (_scan_macro_windows -> get_z_windows_train_aligned) only looks at
    # THIS channel's own column, so leaving the other 3 fully populated means
    # their h-functions train on more/different calendar windows than
    # macro_change's. Forcing the SAME NaN pattern onto all 4 columns makes
    # main.py's own validity check land on the identical window set for
    # every channel, with zero code changes -- consistent training data is a
    # property of the INPUT, not something that needs special-casing in
    # main.py/data_processor.py.
    macro_series = cond_df["macro_change"].reindex(prod_df.index)
    macro_nan_mask = macro_series.isna().to_numpy()
    print(f"\nmacro_change NaN mask: {macro_nan_mask.sum()}/{len(macro_nan_mask)} dates -- "
          f"forcing this same mask onto every other channel's column too.")

    for channel in CHANNELS:
        series = cond_df[channel].reindex(prod_df.index)
        if channel != "macro_change":
            series = series.mask(macro_nan_mask)
        n_nan = int(series.isna().sum())
        print(f"\n{channel}: {n_nan}/{len(series)} NaN after reindexing + common-mask "
              f"(common pool guarantees identical valid windows across all 4 channels)")

        out_df = prod_df[asset_cols].copy()
        out_df.insert(0, channel, series.to_numpy())

        out_path = os.path.join(OUT_DIR, f"data_for_diffusion_{channel}.csv")
        out_df.to_csv(out_path, index_label="Date")
        print(f"  Wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
