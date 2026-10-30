"""
Tail / rare-event slice of Experiment 1: restricts the same Group A/B metrics
from exp_evaluation.py to just the top `--tail-fraction` (default 7.5%, matching
HFunctionConfig.event_threshold's default) of test windows, using EXACTLY the
same event-definition mechanism as the production main.py pipeline
(data/data_processor.py's get_event_threshold_from_percentile + main.py's
event masking, config.hfunction.event_type="upper_change" by default):

    Z_start = variable's value at the window's FIRST day (non-causal,
              event_causal=False default: same seq_len days as the return
              window itself -- main.py:245-311, data_processor.py:595-630).
    Z_end   = variable's value at the window's LAST day (start + seq_len - 1).
    diff    = Z_end - Z_start
    threshold = quantile(diff, 1 - tail_fraction) over TRAIN windows only
                (event_type="upper_change"; see EVENT_TYPES below for the
                other 4 branches, all copied verbatim from
                data_processor.py:684-698 and main.py:250-268/292-310).
    mask    = diff_test >= threshold   (upper_change; other event_types use
              the matching comparison, exactly as in main.py)

Applied to each of M1-M4's OWN conditioning variable (the one it's actually
trained on: ewma_mean, ewma_vol, pca, macro_change) rather than switching M4
back to the raw macro level -- this answers "does conditioning on variable X
actually pay off specifically when X's own window-over-window change is
extreme?" for each of the 4 candidate channels, generalizing the exact
production mechanism beyond just the macro state.

For each of M1-M4's tail definition, ALL 5 models (M0-M4, unchanged
checkpoints/generated samples -- NO retraining or resampling here) are
evaluated on that same masked subset of test windows, with M0 as the
unconditional baseline.

Consumes SAVED outputs only (same contract as exp_evaluation.py):
    <experiment-dir>/shared/test_data.npz
    <experiment-dir>/<MODEL_DIR>/generated_samples.npy

Usage:
    python tail_evaluation.py --experiment-dir experiment_1/exp1_results \
        --output-dir experiment_1/exp1_tail_results
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.dirname(_HERE)  # baseline_experiments/
sys.path.insert(0, _BASELINE)
sys.path.insert(0, _HERE)

from exp_evaluation import (  # noqa: E402
    MODEL_LABELS, QUANTILES, load_shared_test_data, load_generated,
    mean_energy_score, mean_wasserstein_marginals, group_b_metrics,
)
import main_exp1 as m1  # noqa: E402

TAIL_MODEL_KEYS = ["m1", "m2", "m3", "m4"]
EVENT_TYPES = ("abs_change", "absval", "upper_change", "lower_change", "start_upper")


def recompute_pipeline(cfg, args):
    """Rebuild cond_df/starts/split exactly as main_exp1.main() does, using the
    same CLI args as the training run (lookbacks etc). Pure pandas/numpy, no
    GPU, no randomness -- deterministic given the same macro_data_new.csv."""
    df, macro_col = m1.load_raw_data(cfg)
    cond_df = m1.compute_conditioning_variables(
        df, macro_col, L_ewma=args.ewma_lookback, L_pca=args.pca_lookback, alpha=59 / 61,
        macro_change_window=args.macro_change_window,
    )
    X, cond_vals, w_start, w_end, c_dates, starts = m1.build_windows(df, cond_df, seq_len=cfg.data.seq_len)
    split_idx, test_start_idx = m1.train_test_split_idx(starts, w_end, cfg.data.seq_len, cfg)
    return cond_df, starts, split_idx, test_start_idx


def z_start_end(series_values: np.ndarray, starts: np.ndarray, seq_len: int):
    """Z_start/Z_end at the window's own first/last day -- data_processor.py's
    event_causal=False (default) branch: `z_start = macro_values[i]`,
    `z_end = macro_values[i + seq_len - 1]`."""
    return series_values[starts], series_values[starts + seq_len - 1]


def event_threshold_from_percentile(diffs_train, z_start_train, z_end_train, top_fraction, event_type):
    """Verbatim port of data_processor.py:684-698, except NaN windows (e.g. a
    macro observation missing at one of the two window endpoints -- macro_level
    is ~27% NaN, unlike the complete per-day stock-derived series) are dropped
    before quantiling, matching _scan_macro_windows's explicit
    `if isnan(z_start) or isnan(z_end): continue` skip."""
    if event_type == "abs_change":
        return float(np.nanquantile(np.abs(diffs_train), 1.0 - top_fraction))
    elif event_type == "absval":
        return float(np.nanquantile(np.abs(z_end_train), 1.0 - top_fraction))
    elif event_type == "upper_change":
        return float(np.nanquantile(diffs_train, 1.0 - top_fraction))
    elif event_type == "lower_change":
        return float(-np.nanquantile(diffs_train, top_fraction))
    elif event_type == "start_upper":
        return float(np.nanquantile(z_start_train, 1.0 - top_fraction))
    raise NotImplementedError(f"event_type={event_type!r} not in {EVENT_TYPES}")


def event_mask(diffs, z_start, z_end, threshold, event_type):
    """Verbatim port of main.py:250-268 / 292-310, with NaN (missing macro
    observation at a window endpoint) explicitly treated as 'not an event'
    rather than propagating -- mirrors valid_idx gating mask_test to False
    outside the valid set in main.py."""
    with np.errstate(invalid="ignore"):
        if event_type == "abs_change":
            return np.nan_to_num(np.abs(diffs), nan=-np.inf) >= threshold
        elif event_type == "absval":
            return np.nan_to_num(np.abs(z_end), nan=-np.inf) >= threshold
        elif event_type == "upper_change":
            return np.nan_to_num(diffs, nan=-np.inf) >= threshold
        elif event_type == "lower_change":
            return np.nan_to_num(diffs, nan=np.inf) <= -threshold
        elif event_type == "start_upper":
            return np.nan_to_num(z_start, nan=-np.inf) >= threshold
    raise NotImplementedError(f"event_type={event_type!r} not in {EVENT_TYPES}")


def evaluate_subset(X_test_sub, gen_sub, model_key):
    es = mean_energy_score(X_test_sub, gen_sub)
    w1 = mean_wasserstein_marginals(X_test_sub, gen_sub)
    b = group_b_metrics(X_test_sub, gen_sub, QUANTILES)
    row = {
        "model": MODEL_LABELS[model_key],
        "energy_score": es,
        "wasserstein1_marginal": w1,
        "portfolio_variance_w1": b["E_variance_w1"],
        "portfolio_variance_meandiff": b["E_variance_meandiff"],
        "portfolio_mdd_w1": b["E_mdd_w1"],
        "portfolio_mdd_meandiff": b["E_mdd_meandiff"],
    }
    for q in QUANTILES:
        row[f"pinball_{q}"] = b["pinball"][q]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.075,
                         help="top fraction counted as a rare event (default 0.075 = top 7.5%%, "
                              "matching HFunctionConfig.event_threshold's default)")
    parser.add_argument("--event-type", type=str, default="upper_change", choices=EVENT_TYPES,
                         help="matches config.hfunction.event_type's default ('upper_change')")
    parser.add_argument("--ewma-lookback", type=int, default=10)
    parser.add_argument("--pca-lookback", type=int, default=10)
    parser.add_argument("--macro-change-window", type=int, default=None)
    args = parser.parse_args()

    cfg = m1.get_default_config()
    if args.macro_change_window is None:
        args.macro_change_window = cfg.data.seq_len
    seq_len = cfg.data.seq_len

    experiment_dir = args.experiment_dir if os.path.isabs(args.experiment_dir) \
        else os.path.join(_BASELINE, args.experiment_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(_BASELINE, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Recomputing conditioning series + window split (lookback ewma={args.ewma_lookback}, "
          f"pca={args.pca_lookback}, macro_change_window={args.macro_change_window}, "
          f"event_type={args.event_type})...")
    cond_df, starts, split_idx, test_start_idx = recompute_pipeline(cfg, args)

    X_test = load_shared_test_data(experiment_dir)
    print(f"Loaded shared test data: {X_test.shape}")

    gens = {}
    for mk in m1.MODEL_KEYS:
        gen, _ = load_generated(experiment_dir, mk)
        if gen is None:
            print(f"[warn] {mk}: no generated_samples.npy found -- it will be skipped everywhere")
        gens[mk] = gen

    starts_train = starts[:split_idx]
    starts_test = starts[test_start_idx:]

    for tail_mk in TAIL_MODEL_KEYS:
        cond_key = m1.COND_KEY[tail_mk]
        series_values = cond_df[cond_key].to_numpy()

        z_start_train, z_end_train = z_start_end(series_values, starts_train, seq_len)
        diffs_train = z_end_train - z_start_train
        threshold = event_threshold_from_percentile(diffs_train, z_start_train, z_end_train,
                                                      args.tail_fraction, args.event_type)

        z_start_test, z_end_test = z_start_end(series_values, starts_test, seq_len)
        diffs_test = z_end_test - z_start_test
        mask = event_mask(diffs_test, z_start_test, z_end_test, threshold, args.event_type)

        n_events = int(mask.sum())
        print(f"\n{'=' * 60}\n"
              f"Tail slice defined by {tail_mk.upper()} ({cond_key}), event_type={args.event_type}, "
              f"top {args.tail_fraction * 100:.1f}% (train threshold={threshold:.4f}): "
              f"{n_events}/{len(mask)} test windows\n{'=' * 60}")
        if n_events == 0:
            print("[skip] no test windows passed the threshold")
            continue

        rows = []
        for mk in m1.MODEL_KEYS:
            if gens[mk] is None:
                continue
            n = min(len(X_test), len(gens[mk]), len(mask))
            mask_n = mask[:n]
            X_test_sub = X_test[:n][mask_n]
            gen_sub = gens[mk][:n][mask_n]
            if len(X_test_sub) == 0:
                print(f"  [skip] {mk}: 0 events within its available {n} samples")
                continue
            row = evaluate_subset(X_test_sub, gen_sub, mk)
            rows.append(row)
            print(f"  {mk}: energy_score={row['energy_score']:.4f}  "
                  f"W1_marginal={row['wasserstein1_marginal']:.4f}  "
                  f"var_W1={row['portfolio_variance_w1']:.6f}  mdd_W1={row['portfolio_mdd_w1']:.4f}")

        if not rows:
            continue
        table = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"tail_{tail_mk}_{cond_key}_comparison.csv")
        table.to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}  (n_events={n_events}, threshold={threshold:.4f})")
        print(table.to_string(index=False))

    print(f"\nDone. All tail tables written to {output_dir}")


if __name__ == "__main__":
    main()
