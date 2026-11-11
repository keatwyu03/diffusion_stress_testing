"""
Build 5 per-channel variants of data_for_diffusion.csv for the Doob-h
rare-event experiment (macro embedding, S0-S4): S0 state-space, S1 linear
interpolation, S2 forward-fill, S3 cubic spline, S4 tracking regression.

Each output CSV is identical to data_for_diffusion.csv in its Date index and
15 asset columns. Only column 0 is replaced, with that method's RAW daily
state level (<states-dir>/<method>/daily_state.csv).

Column 0 must be the raw level, not a pre-computed change: data_processor.py's
_scan_macro_windows reads Z_start/Z_end from column 0 and the event label is
Z_end - Z_start, i.e. state[end] - state[start] (one change). Causal vs
non-causal attachment is handled inside data_processor.py by
config.data.event_causal / event_lag_gap -- the CSV is never shifted here.

S1/S3 are RETROSPECTIVE (interpolation needs the next monthly anchor) --
disclosed in channel_meta.json downstream, not hidden.

Usage:
    python build_channel_csvs_exp2_all.py --states-dir "<.../exp2_results>"            # Full
    python build_channel_csvs_exp2_all.py --states-dir "<.../exp2_results_cpi_only>" --suffix _cpi
"""
import argparse
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

PROD_CSV = os.path.join(_ROOT, "explore", "standardization_results", "data_for_diffusion.csv")
OUT_DIR = os.path.join(_ROOT, "explore", "standardization_results", "experiments")

METHOD_DIRNAMES = {
    "exp2_s0": "S0_state_space",
    "exp2_s1": "S1_linear",
    "exp2_s2": "S2_forward_fill",
    "exp2_s3": "S3_cubic_spline",
    "exp2_s4": "S4_tracking_regression",
}
RETROSPECTIVE = {"exp2_s0": False, "exp2_s1": True, "exp2_s2": False, "exp2_s3": True, "exp2_s4": False}


def main():
    raise SystemExit('build_channel_csvs_exp2_all.py is superseded: build_channel_csvs_exp2_ownstd.py now builds the S0-S4 channel files (event column + own residuals) for the conditioning set in config, in one step.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-dir", required=True,
                         help="Folder containing <method>/daily_state.csv for all 5 methods.")
    parser.add_argument("--suffix", default="",
                         help="Output filename suffix, e.g. _cpi -> data_for_diffusion_exp2_s0_cpi.csv")
    args = parser.parse_args()

    prod_df = pd.read_csv(PROD_CSV, parse_dates=["Date"]).set_index("Date")
    asset_cols = list(prod_df.columns[1:])
    print(f"Production CSV: {PROD_CSV} ({prod_df.shape[0]} rows)")
    print(f"States dir: {args.states_dir}   output suffix: {args.suffix!r}")

    levels = {}
    for channel, method_dir in METHOD_DIRNAMES.items():
        state_csv = os.path.join(args.states_dir, method_dir, "daily_state.csv")
        state = pd.read_csv(state_csv, index_col=0, parse_dates=True)["daily_state"]
        levels[channel] = state.reindex(prod_df.index)

    # One shared missing-day mask across all 5 methods, so _scan_macro_windows
    # keeps the identical window set for every method.
    common_nan = pd.concat(levels.values(), axis=1).isna().any(axis=1).to_numpy()
    print(f"Shared NaN mask: {common_nan.sum()}/{len(common_nan)} dates (applied to all 5 methods)")

    for channel in METHOD_DIRNAMES:
        level = levels[channel].mask(common_nan)
        retro = " [RETROSPECTIVE]" if RETROSPECTIVE[channel] else ""
        print(f"\n{channel}{retro}: {int(level.isna().sum())}/{len(level)} NaN")

        out_df = prod_df[asset_cols].copy()
        out_df.insert(0, channel, level.to_numpy())
        out_path = os.path.join(OUT_DIR, f"data_for_diffusion_{channel}{args.suffix}.csv")
        out_df.to_csv(out_path, index_label="Date")
        print(f"  Wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
