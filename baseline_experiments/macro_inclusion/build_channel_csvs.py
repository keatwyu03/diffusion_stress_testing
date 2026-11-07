"""
Build 4 per-channel variants of data_for_diffusion.csv for the Doob-h
rare-event experiment (macro inclusion, M1-M4).

Each output CSV is identical to data_for_diffusion.csv in its Date index and
15 asset columns. Only column 0 is replaced, with the RAW (level) series of
one exogenous variable:
    macro_change      <- m_t           (macro state level)
    ewma_mean_change  <- EWMA mean
    ewma_vol_change   <- EWMA vol
    pca_change        <- PCA factor

Column 0 must be the raw level, not a pre-computed change: data_processor.py's
_scan_macro_windows reads Z_start/Z_end from column 0 and the event label is
Z_end - Z_start. A raw level makes that label v[end] - v[start] (one change).
A pre-computed change would make it a change of a change. Channel names
describe the resulting label (the change in that variable).

data/data_processor.py picks the conditioning column positionally
(df.columns[0]), so main.py and data_processor.py need no changes.

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

# channel name (= the label it produces) -> raw level column in compute_conditioning_variables
CHANNEL_SOURCE = {
    "macro_change": "macro_level",
    "ewma_mean_change": "ewma_mean",
    "ewma_vol_change": "ewma_vol",
    "pca_change": "pca",
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-csv", default=None,
                         help="Input with the macro state in column 0 (default explore/macro_data_new.csv); "
                              "e.g. explore/macro_data_new_cpi.csv for CPI-only.")
    parser.add_argument("--suffix", default="", help="Output filename suffix, e.g. _cpi")
    args = parser.parse_args()
    cfg = m1.get_default_config()

    df_raw, macro_col = m1.load_raw_data(cfg, csv_override=args.macro_csv)
    cond_df = m1.compute_conditioning_variables(
        df_raw, macro_col, L_ewma=10, L_pca=10, alpha=59 / 61, macro_change_window=cfg.data.seq_len,
    )

    prod_df = pd.read_csv(PROD_CSV, parse_dates=["Date"]).set_index("Date")
    asset_cols = list(prod_df.columns[1:])
    print(f"Production CSV: {PROD_CSV}")
    print(f"  {prod_df.shape[0]} rows, date range {prod_df.index.min()} -> {prod_df.index.max()}")

    missing_dates = prod_df.index.difference(cond_df.index)
    if len(missing_dates) > 0:
        print(f"[warn] {len(missing_dates)} dates fall outside the conditioning series' range "
              f"(channel value NaN): {missing_dates.min()} -> {missing_dates.max()}")

    # The macro level is the only series with real missing-data gaps. Forcing its
    # NaN mask onto all 4 columns makes _scan_macro_windows' validity check land
    # on the identical window set for every channel.
    macro_nan_mask = cond_df["macro_level"].reindex(prod_df.index).isna().to_numpy()
    print(f"\nmacro_level NaN mask: {macro_nan_mask.sum()}/{len(macro_nan_mask)} dates "
          f"(applied to every channel)")

    for channel, source in CHANNEL_SOURCE.items():
        series = cond_df[source].reindex(prod_df.index).mask(macro_nan_mask)
        print(f"\n{channel} <- raw {source}: {int(series.isna().sum())}/{len(series)} NaN")

        out_df = prod_df[asset_cols].copy()
        out_df.insert(0, channel, series.to_numpy())
        out_path = os.path.join(OUT_DIR, f"data_for_diffusion_{channel}{args.suffix}.csv")
        out_df.to_csv(out_path, index_label="Date")
        print(f"  Wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
