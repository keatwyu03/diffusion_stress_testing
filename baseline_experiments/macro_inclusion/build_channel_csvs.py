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

--raw-returns puts RAW daily log returns (the conditioning set's macro_data_new.csv) in the
15 asset columns instead of the production NN-standardized residuals, so the
rare-event diffusion targets are raw returns for every M1-M4 channel (macro
inclusion must not depend on S0-standardized residuals). Calendar, column 0
and its NaN mask are unchanged.

Usage:
    python build_channel_csvs.py
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import main_exp1 as m1  # noqa: E402

# All inputs/outputs come from the conditioning set selected in config (config.data.conditioning_set):
#   calendar  = config.data.csv_path        explore/conditioning_sets/<set>/data_for_diffusion.csv (days with valid m_t)
#   macro/raw = config.data.macro_csv_path  explore/conditioning_sets/<set>/macro_data_new.csv (m_t + raw log returns)
#   output    = config.data.channels_dir    explore/conditioning_sets/<set>/rare_event_channels/

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
                         help="Input with the macro state in column 0 (default: config.data.macro_csv_path, the "
                              "conditioning set's macro_data_new.csv).")
    parser.add_argument("--suffix", default="", help="Output filename suffix, e.g. _cpi")
    parser.add_argument("--raw-returns", action="store_true",
                         help="Asset columns = raw daily log returns instead of the production residuals.")
    parser.add_argument("--out-dir", default=None, help="Where to write the CSVs (default: config.data.channels_dir)")
    args = parser.parse_args()
    cfg = m1.get_default_config()
    prod_csv = cfg.data.csv_path
    macro_csv = args.macro_csv or cfg.data.macro_csv_path
    out_dir = args.out_dir or cfg.data.channels_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Conditioning set: {cfg.data.conditioning_set} | calendar {prod_csv} | macro/raw {macro_csv} | out {out_dir}")

    df_raw, macro_col = m1.load_raw_data(cfg, csv_override=macro_csv)
    cond_df = m1.compute_conditioning_variables(
        df_raw, macro_col, L_ewma=10, L_pca=10, alpha=59 / 61, macro_change_window=cfg.data.seq_len,
    )

    prod_df = pd.read_csv(prod_csv, parse_dates=["Date"]).set_index("Date")
    asset_cols = list(prod_df.columns[1:])
    if args.raw_returns:
        # read the CSV directly (load_raw_data cuts at config end_date; DataProcessor applies that cut itself)
        raw = pd.read_csv(macro_csv, parse_dates=["Date"]).set_index("Date")[asset_cols].reindex(prod_df.index)
        if raw.isna().any().any():
            raise ValueError("raw returns missing on some production-calendar dates")
        prod_df[asset_cols] = raw
        print(f"Asset columns = RAW daily log returns from {macro_csv}")
    print(f"Calendar CSV: {prod_csv}")
    print(f"  {prod_df.shape[0]} rows, date range {prod_df.index.min()} -> {prod_df.index.max()}")

    missing_dates = prod_df.index.difference(cond_df.index)
    if len(missing_dates) > 0:
        print(f"[warn] {len(missing_dates)} dates fall outside the conditioning series' range "
              f"(channel value NaN): {missing_dates.min()} -> {missing_dates.max()}")

    # The macro level is the only series with real missing-data gaps. Forcing its
    # NaN mask onto all 4 columns makes _scan_macro_windows' validity check land
    # on the identical window set for every channel.
    # Union with every channel source's own NaNs (EWMA/PCA warm-up at the data-window start), so all 4 channels
    # keep the identical window set when the data window starts on the first available day (2000-09-25).
    macro_nan_mask = np.zeros(len(prod_df), dtype=bool)
    for source in CHANNEL_SOURCE.values():
        macro_nan_mask |= cond_df[source].reindex(prod_df.index).isna().to_numpy()
    print(f"\nshared NaN mask (macro_level + EWMA/PCA warm-up): {macro_nan_mask.sum()}/{len(macro_nan_mask)} dates "
          f"(applied to every channel)")

    for channel, source in CHANNEL_SOURCE.items():
        series = cond_df[source].reindex(prod_df.index).mask(macro_nan_mask)
        print(f"\n{channel} <- raw {source}: {int(series.isna().sum())}/{len(series)} NaN")

        out_df = prod_df[asset_cols].copy()
        out_df.insert(0, channel, series.to_numpy())
        out_path = os.path.join(out_dir, f"data_for_diffusion_{channel}{args.suffix}.csv")
        out_df.to_csv(out_path, index_label="Date")
        print(f"  Wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
