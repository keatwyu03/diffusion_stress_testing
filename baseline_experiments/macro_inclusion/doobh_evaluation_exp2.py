"""
Evaluates the guided Doob-h generation .pt output for one Experiment-2-method
channel (exp2_s0..exp2_s4) against REAL historical event windows -- same
mechanism as doobh_evaluation.py (main.py's/data_processor.py's own
unmodified machinery), adapted in exactly one way: each channel is evaluated
against its OWN valid-window pool (non-NaN Z_start/Z_end for THIS channel
only), not a pool shared across all 5 channels. This run is deliberately
split across multiple jobs/channel subsets so results can be inspected
before the full channel set finishes, which rules out computing a
common-across-5 pool up front; disclosed here, not hidden.

See doobh_evaluation.py for the full rationale behind the pooled-energy-score/
destandardization approach (same here, unchanged).

Usage:
    python doobh_evaluation_exp2.py --channel exp2_s0 --results-root doobh_results_exp2_noncausal
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_BASELINE = os.path.dirname(_HERE)
_EXP2_ROOT = os.path.join(_BASELINE, "main_results")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BASELINE)

from config import get_default_config  # noqa: E402
from data import DataProcessor  # noqa: E402
from exp_evaluation import portfolio_variance, max_drawdown, portfolio_cum_return, pinball_loss  # noqa: E402

QUANTILES = (0.01, 0.05, 0.95, 0.99)
CHANNELS = ["exp2_s0", "exp2_s1", "exp2_s2", "exp2_s3", "exp2_s4", "exp2_s1p", "exp2_s3p"]
RETROSPECTIVE = {"exp2_s0": False, "exp2_s1": True, "exp2_s2": False, "exp2_s3": True, "exp2_s4": False,
                 "exp2_s1p": False, "exp2_s3p": False}


from doobh_evaluation import window_z_start_end  # noqa: E402  (single shared copy of _scan_macro_windows)

CSV_SUFFIX = ""  # set from --csv-suffix in main(); e.g. "_cpi"


def pooled_energy_score(real_windows_AT: np.ndarray, gen_windows_AT: np.ndarray) -> float:
    K = gen_windows_AT.shape[0]
    gen_flat = gen_windows_AT.reshape(K, -1)
    real_flat = real_windows_AT.reshape(len(real_windows_AT), -1)
    term1_per_window = cdist(real_flat, gen_flat, metric="euclidean").mean(axis=1)
    term2 = float(pdist(gen_flat, metric="euclidean").sum() / (K * (K - 1))) if K > 1 else 0.0
    return float((term1_per_window - term2).mean())


def channel_macro_values(channel: str, reference_index: pd.DatetimeIndex) -> np.ndarray:
    from window_conditioning import experiment_paths   # channel files of the config's conditioning set
    csv_path = os.path.join(experiment_paths()["embed_channels_dir"], f"data_for_diffusion_{channel}{CSV_SUFFIX}.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    col = df.columns[0]
    return df[col].reindex(reference_index).to_numpy(dtype=np.float64)


def self_valid_mask(channel, reference_dp, config, split: str) -> np.ndarray:
    """THIS channel's own non-NaN Z_start/Z_end mask -- not intersected with
    the other 4 channels (see module docstring)."""
    seq_len = config.data.seq_len
    n_train = reference_dp.X_train.shape[0]
    n_test = reference_dp.X_test.shape[0]
    n_total = n_train + n_test
    split_offset, n_windows = (0, n_train) if split == "train" else (n_train, n_test)

    values = channel_macro_values(channel, reference_dp.df.index)
    z_start_all, z_end_all = window_z_start_end(values, seq_len, n_total, config.data.event_causal, config.data.event_lag_gap)
    z_start = z_start_all[split_offset: split_offset + n_windows]
    z_end = z_end_all[split_offset: split_offset + n_windows]
    return ~np.isnan(z_start) & ~np.isnan(z_end)


def build_event_mask(channel: str, reference_dp, config, split: str, valid: np.ndarray, fit_on: np.ndarray = None):
    seq_len = config.data.seq_len
    n_train = reference_dp.X_train.shape[0]
    n_test = reference_dp.X_test.shape[0]
    n_total = n_train + n_test
    split_offset, n_windows = (0, n_train) if split == "train" else (n_train, n_test)

    values = channel_macro_values(channel, reference_dp.df.index)
    z_start_all, z_end_all = window_z_start_end(values, seq_len, n_total, config.data.event_causal, config.data.event_lag_gap)
    z_start = z_start_all[split_offset: split_offset + n_windows]
    z_end = z_end_all[split_offset: split_offset + n_windows]

    event_type = config.hfunction.event_type

    def metric_of(zs, ze):
        if event_type == "abs_change":
            return np.abs(ze - zs)
        elif event_type == "absval":
            return np.abs(ze)
        elif event_type in ("upper_change", "lower_change"):
            return ze - zs
        elif event_type == "start_upper":
            return zs
        raise NotImplementedError(event_type)

    metric = metric_of(z_start, z_end)
    fit_pool = metric[valid] if fit_on is None else fit_on

    top_fraction = config.hfunction.event_threshold
    if event_type == "lower_change":
        threshold = -np.quantile(fit_pool, top_fraction)
        event_valid = metric <= -threshold
    else:
        threshold = np.quantile(fit_pool, 1.0 - top_fraction)
        event_valid = metric >= threshold

    mask = event_valid & valid
    return mask, float(threshold), metric[valid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=CHANNELS)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--csv-suffix", default="", help="e.g. '_cpi' to evaluate against the CPI-only channel CSV")
    args = parser.parse_args()
    global CSV_SUFFIX
    CSV_SUFFIX = args.csv_suffix

    from window_conditioning import experiment_paths
    channel_csv = os.path.join(experiment_paths()["embed_channels_dir"],
                                f"data_for_diffusion_{args.channel}{args.csv_suffix}.csv")
    results_dir = os.path.join(_EXP2_ROOT, args.results_root, args.channel)
    gen_path = os.path.join(results_dir, "generated_samples_test.pt")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"{gen_path} not found -- channel hasn't produced samples yet.")

    config = get_default_config()
    print(f"Rebuilding DataProcessor exactly as main.py's STEP 1 does, from {channel_csv} "
          f"(event_causal={config.data.event_causal}) ...")
    data_processor = DataProcessor(
        csv_path=channel_csv, tickers=config.data.tickers, weekday_col=config.data.weekday_col,
        seq_len=config.data.seq_len, test_days=config.data.test_days, start_date=config.data.start_date,
        end_date=config.data.end_date, train_end_date=config.data.train_end_date,
        window_shift=config.data.window_shift, winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper, ema_span=config.data.ema_span,
        event_causal=config.data.event_causal, use_ema_standardization=config.data.use_ema_standardization,
        event_lag_gap=config.data.event_lag_gap,
    )
    data_processor.process_all()

    valid_train = self_valid_mask(args.channel, data_processor, config, "train")
    print(f"Valid window pool (this channel only, train): {valid_train.sum()}/{len(valid_train)}")

    train_mask, train_threshold, train_fit_pool = build_event_mask(
        args.channel, data_processor, config, "train", valid_train, fit_on=None)
    print(f"Event threshold (own pool): top {config.hfunction.event_threshold:.1%} -> "
          f"{train_threshold:.4f} ({config.hfunction.event_type})")

    if args.split == "train":
        mask = train_mask
    else:
        valid_test = self_valid_mask(args.channel, data_processor, config, "test")
        mask, _, _ = build_event_mask(args.channel, data_processor, config, "test", valid_test, fit_on=train_fit_pool)

    n_events = int(mask.sum())
    print(f"Real {args.split}-set events for channel '{args.channel}' (own pool): {n_events}/{len(mask)} windows")

    X_split = data_processor.X_train if args.split == "train" else data_processor.X_test
    mu_split = data_processor.mu_entry_train if args.split == "train" else data_processor.mu_entry_test
    sig_split = data_processor.sig_entry_train if args.split == "train" else data_processor.sig_entry_test

    real_z = np.transpose(X_split.numpy()[mask], (0, 2, 1))
    real_mu = mu_split[mask]
    real_sig = sig_split[mask]
    real_raw_AT = data_processor.destandardize_windows(real_z, real_mu, real_sig)

    gen_z = torch.load(gen_path, map_location="cpu", weights_only=False)
    n_gen = gen_z.shape[0]
    gen_mu, gen_sig = data_processor.sample_entry_stats(n_draw=n_gen, split=args.split, mask=mask, seed=config.seed)
    gen_raw_AT = data_processor.destandardize_windows(gen_z, gen_mu, gen_sig)

    print(f"Real event windows (A,T): {real_raw_AT.shape}, generated samples (A,T): {gen_raw_AT.shape}")

    energy = pooled_energy_score(real_raw_AT, gen_raw_AT)

    A = real_raw_AT.shape[1]
    w1_marginals = [wasserstein_distance(real_raw_AT[:, a, -1], gen_raw_AT[:, a, -1]) for a in range(A)]
    w1_marginal = float(np.mean(w1_marginals))
    best_asset_idx, worst_asset_idx = int(np.argmin(w1_marginals)), int(np.argmax(w1_marginals))

    var_real = np.array([portfolio_variance(real_raw_AT[i]) for i in range(len(real_raw_AT))])
    var_gen = np.array([portfolio_variance(gen_raw_AT[i]) for i in range(len(gen_raw_AT))])
    mdd_real = np.array([max_drawdown(real_raw_AT[i]) for i in range(len(real_raw_AT))])
    mdd_gen = np.array([max_drawdown(gen_raw_AT[i]) for i in range(len(gen_raw_AT))])

    gen_cum_returns = np.array([portfolio_cum_return(gen_raw_AT[i]) for i in range(len(gen_raw_AT))])
    real_cum_returns = np.array([portfolio_cum_return(real_raw_AT[i]) for i in range(len(real_raw_AT))])
    pinball_avg = {q: float(np.mean([pinball_loss(y, float(np.quantile(gen_cum_returns, q)), q)
                                      for y in real_cum_returns])) for q in QUANTILES}

    row = {
        "channel": args.channel, "retrospective": RETROSPECTIVE[args.channel],
        "event_causal": config.data.event_causal, "split": args.split,
        "n_real_events": n_events, "n_generated": n_gen,
        "energy_score": energy, "wasserstein1_marginal": w1_marginal,
        "wasserstein1_marginal_best": float(w1_marginals[best_asset_idx]),
        "wasserstein1_marginal_best_ticker": config.data.tickers[best_asset_idx],
        "wasserstein1_marginal_worst": float(w1_marginals[worst_asset_idx]),
        "wasserstein1_marginal_worst_ticker": config.data.tickers[worst_asset_idx],
        "portfolio_variance_real_median": float(np.median(var_real)),
        "portfolio_variance_gen_median": float(np.median(var_gen)),
        "portfolio_variance_w1": float(wasserstein_distance(var_real, var_gen)),
        "portfolio_mdd_real_median": float(np.median(mdd_real)),
        "portfolio_mdd_gen_median": float(np.median(mdd_gen)),
        "portfolio_mdd_w1": float(wasserstein_distance(mdd_real, mdd_gen)),
        "portfolio_cumret_real_median": float(np.median(real_cum_returns)),
        "portfolio_cumret_gen_median": float(np.median(gen_cum_returns)),
    }
    for q in QUANTILES:
        row[f"pinball_{q}_pooled"] = pinball_avg[q]

    out_path = os.path.join(results_dir, f"doobh_comparison_{args.split}.csv")
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
