"""
Evaluates the Doob-h guided generation .pt output for one channel against the
REAL historical event windows, using main.py's/data_processor.py's own,
unmodified, pre-existing machinery -- no changes to those files.

Why this can't just reuse exp_evaluation.py as-is:
  - main.py's STEP 5 (ConditionalGenerator.generate(num_samples=2000)) draws
    one POOLED set of samples from the event-conditional distribution -- they
    are NOT tied to any specific real test window (unlike main_exp1.py's
    per-window generation), so there is no per-window pairing for an energy
    score or a per-window pinball loss. This script instead compares the
    pooled generated distribution against the pooled real event-window
    distribution (Group B style: portfolio variance / drawdown Wasserstein-1,
    marginal Wasserstein-1, and a POOLED pinball loss).
  - The saved .pt is in per-window EMA-standardized z-units (mean~-0.16,
    std~1.09, range +-14) -- NOT raw log-returns. main.py's own STEP 6 would
    normally invert this via data_processor.destandardize_windows(), but it
    currently calls a removed method name (invert_samples) and crashes before
    reaching that step -- a pre-existing bug, already known/accepted, unrelated
    to this generalization. This script calls destandardize_windows() /
    sample_entry_stats() directly instead (both already exist and are
    documented for exactly this purpose in data_processor.py).

Usage:
    python doobh_evaluation.py --channel macro_change
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
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BASELINE)

from config import get_default_config  # noqa: E402
from data import DataProcessor  # noqa: E402
from exp_evaluation import portfolio_variance, max_drawdown, portfolio_cum_return, pinball_loss  # noqa: E402

QUANTILES = (0.01, 0.05, 0.95, 0.99)
ALL_CHANNELS = ["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"]


def window_z_start_end(values: np.ndarray, seq_len: int, n_windows: int):
    """z_start[w] = values[w], z_end[w] = values[w+seq_len-1] for window w,
    window_shift=1 -- exactly data_processor.py's _scan_macro_windows
    event_causal=False branch, but vectorized over ALL windows at once
    (instead of just the ones where both happen to be non-NaN) so we can
    intersect validity across channels before applying any threshold."""
    z_start = values[:n_windows]
    z_end = values[seq_len - 1: seq_len - 1 + n_windows]
    return z_start, z_end


def pooled_energy_score(real_windows_AT: np.ndarray, gen_windows_AT: np.ndarray) -> float:
    """Adapts exp_evaluation.py's per-window energy_score to this pipeline's
    POOLED generation: there's no per-window-matched K samples here (one
    shared 2000-sample batch represents the whole event-conditional
    distribution, not any single real window), so each of the real event
    windows is scored against that SAME shared pool, then averaged -- same
    "pooled" convention already used for the pinball loss above. term2 (the
    generated-sample diversity term) only depends on the generated pool, so
    it's computed once and reused across all real windows, not recomputed
    per window like a naive per-window loop would.
    real_windows_AT: (N, A, T), gen_windows_AT: (K, A, T)."""
    K = gen_windows_AT.shape[0]
    gen_flat = gen_windows_AT.reshape(K, -1)
    real_flat = real_windows_AT.reshape(len(real_windows_AT), -1)

    term1_per_window = cdist(real_flat, gen_flat, metric="euclidean").mean(axis=1)
    term2 = float(pdist(gen_flat, metric="euclidean").sum() / (K * (K - 1))) if K > 1 else 0.0

    return float((term1_per_window - term2).mean())


def channel_macro_values(channel: str, reference_index: pd.DatetimeIndex) -> np.ndarray:
    """Raw (pre-standardization) conditioning column for `channel`, reindexed
    onto `reference_index` -- the row set after DataProcessor.process_all(),
    which is determined ENTIRELY by the asset columns (byte-identical across
    all 4 channel CSVs), never by the conditioning column. So this reindex is
    just alignment, not a source of new NaNs beyond what each channel's own
    column already has."""
    csv_path = os.path.join(_ROOT, "explore", "standardization_results", f"data_for_diffusion_{channel}.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    col = df.columns[0]
    return df[col].reindex(reference_index).to_numpy(dtype=np.float64)


def common_valid_mask(reference_dp, config, split: str) -> np.ndarray:
    """A window only counts for ANY channel's event threshold/classification
    if its Z_start/Z_end are defined for ALL 4 channels -- otherwise channels
    with a sparser underlying column (macro_change: ~27% NaN) draw their
    7.5% from a smaller pool than the complete-data channels (EWMA/PCA: 0%
    NaN) do, making the resulting event COUNTS not comparable across
    channels even though the percentage is identical for each one
    individually. This forces every channel onto the exact same window pool,
    same date range, same row count -- a true apples-to-apples comparison."""
    seq_len = config.data.seq_len
    n_train = reference_dp.X_train.shape[0]
    n_test = reference_dp.X_test.shape[0]
    n_total = n_train + n_test
    split_offset, n_windows = (0, n_train) if split == "train" else (n_train, n_test)

    valid = np.ones(n_windows, dtype=bool)
    for ch in ALL_CHANNELS:
        values = channel_macro_values(ch, reference_dp.df.index)
        z_start_all, z_end_all = window_z_start_end(values, seq_len, n_total)
        z_start = z_start_all[split_offset: split_offset + n_windows]
        z_end = z_end_all[split_offset: split_offset + n_windows]
        valid &= ~np.isnan(z_start) & ~np.isnan(z_end)
    return valid


def build_event_mask(channel: str, reference_dp, config, split: str, common_valid: np.ndarray,
                      fit_on: np.ndarray = None):
    """Event mask for `channel`, restricted to `common_valid` (the window
    pool shared by all 4 channels). `fit_on` (if given) is the common-valid
    TRAIN metric, reused as the quantile-fitting population when scoring the
    test split, so the threshold itself is also fit on the common pool."""
    seq_len = config.data.seq_len
    n_train = reference_dp.X_train.shape[0]
    n_test = reference_dp.X_test.shape[0]
    n_total = n_train + n_test
    split_offset, n_windows = (0, n_train) if split == "train" else (n_train, n_test)

    values = channel_macro_values(channel, reference_dp.df.index)
    z_start_all, z_end_all = window_z_start_end(values, seq_len, n_total)
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

    if fit_on is None:
        # split == "train": fit the threshold on this channel's own common-valid train metric.
        fit_pool = metric[common_valid]
    else:
        fit_pool = fit_on

    top_fraction = config.hfunction.event_threshold
    if event_type == "lower_change":
        threshold = -np.quantile(fit_pool, top_fraction)
        event_valid = metric <= -threshold
    else:
        threshold = np.quantile(fit_pool, 1.0 - top_fraction)
        event_valid = metric >= threshold

    mask = event_valid & common_valid
    return mask, float(threshold), metric[common_valid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True,
                         choices=["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"])
    parser.add_argument("--split", default="train", choices=["train", "test"],
                         help="Which real events to compare against. 'train' is the split the "
                              "7.5%% threshold was actually fit on, so the event fraction is "
                              "~7.5%% for every channel by construction. Default: train.")
    args = parser.parse_args()

    channel_csv = os.path.join(_ROOT, "explore", "standardization_results",
                                f"data_for_diffusion_{args.channel}.csv")
    results_dir = os.path.join(_HERE, "doobh_results", args.channel)
    gen_path = os.path.join(results_dir, "generated_samples_test.pt")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"{gen_path} not found -- channel hasn't produced samples yet.")

    config = get_default_config()
    print(f"Rebuilding DataProcessor exactly as main.py's STEP 1 does, from {channel_csv} ...")
    data_processor = DataProcessor(
        csv_path=channel_csv,
        tickers=config.data.tickers,
        weekday_col=config.data.weekday_col,
        seq_len=config.data.seq_len,
        test_days=config.data.test_days,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        train_end_date=config.data.train_end_date,
        window_shift=config.data.window_shift,
        winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper,
        ema_span=config.data.ema_span,
        event_causal=config.data.event_causal,
        use_ema_standardization=config.data.use_ema_standardization,
        event_lag_gap=config.data.event_lag_gap,
    )
    data_processor.process_all()

    # Window pool shared by ALL 4 channels (a window only counts if every
    # channel's Z_start/Z_end are defined for it), so the 7.5% is drawn from
    # the exact same data for every channel -- not each channel's own,
    # differently-sized valid pool.
    common_valid_train = common_valid_mask(data_processor, config, "train")
    print(f"Common valid-window pool (all 4 channels, train): {common_valid_train.sum()}/{len(common_valid_train)}")

    train_mask, train_threshold, train_fit_pool = build_event_mask(
        args.channel, data_processor, config, "train", common_valid_train, fit_on=None)
    print(f"Event threshold (common pool): top {config.hfunction.event_threshold:.1%} -> "
          f"{train_threshold:.4f} ({config.hfunction.event_type})")

    if args.split == "train":
        mask = train_mask
    else:
        common_valid_test = common_valid_mask(data_processor, config, "test")
        mask, _, _ = build_event_mask(
            args.channel, data_processor, config, "test", common_valid_test, fit_on=train_fit_pool)

    n_events = int(mask.sum())
    print(f"Real {args.split}-set events for channel '{args.channel}' (common pool): "
          f"{n_events}/{len(mask)} windows")

    # make_sequences() (which populates X_train/X_test/mu_entry_*/sig_entry_*) stores
    # windows as (window, T, A) -- r_values[s:t, :] with NO transpose. But
    # get_diffusion_data() (what the diffusion model actually trains/generates on,
    # hence what the saved .pt is in) transposes to (A, T) -- z.T. destandardize_windows()
    # is documented for the (N, A, T) convention (broadcasts mu/sig as (N,A,1)), so
    # X_train/X_test's (T,A) windows must be transposed to (A,T) BEFORE calling it; the
    # generated .pt is already (A,T) and must NOT be transposed again.
    X_split = data_processor.X_train if args.split == "train" else data_processor.X_test
    mu_split = data_processor.mu_entry_train if args.split == "train" else data_processor.mu_entry_test
    sig_split = data_processor.sig_entry_train if args.split == "train" else data_processor.sig_entry_test

    real_z = np.transpose(X_split.numpy()[mask], (0, 2, 1))  # (n_events, A, T)
    real_mu = mu_split[mask]
    real_sig = sig_split[mask]
    real_raw_AT = data_processor.destandardize_windows(real_z, real_mu, real_sig)  # (n_events, A, T)

    # Generated samples: pooled, not tied to a specific window -- destandardize
    # using entry stats SAMPLED (with replacement) from the real event-window
    # pool, per data_processor.sample_entry_stats's own documented purpose
    # ("For CONDITIONAL samples pass the event mask").
    gen_z = torch.load(gen_path, map_location="cpu", weights_only=False)  # already (N, A, T)
    n_gen = gen_z.shape[0]
    gen_mu, gen_sig = data_processor.sample_entry_stats(
        n_draw=n_gen, split=args.split, mask=mask, seed=config.seed)
    gen_raw_AT = data_processor.destandardize_windows(gen_z, gen_mu, gen_sig)  # (n_gen, A, T)

    print(f"Real event windows (destandardized, A,T): {real_raw_AT.shape}, "
          f"generated samples (destandardized, A,T): {gen_raw_AT.shape}")

    # A1: pooled energy score (full joint return distribution, not just last-day marginals).
    energy = pooled_energy_score(real_raw_AT, gen_raw_AT)

    # A2: marginal Wasserstein-1 of last-day returns, per asset -- averaging
    # across 15 assets can hide a badly-matched asset behind 14 decent ones,
    # so the best- and worst-matched individual assets are reported
    # alongside the mean, not just the mean -- shows both how good the best
    # case is and how bad the worst case is, i.e. the actual spread.
    A = real_raw_AT.shape[1]
    w1_marginals = [wasserstein_distance(real_raw_AT[:, a, -1], gen_raw_AT[:, a, -1]) for a in range(A)]
    w1_marginal = float(np.mean(w1_marginals))
    best_asset_idx = int(np.argmin(w1_marginals))
    worst_asset_idx = int(np.argmax(w1_marginals))
    w1_marginal_best = float(w1_marginals[best_asset_idx])
    w1_marginal_worst = float(w1_marginals[worst_asset_idx])
    best_asset_ticker = config.data.tickers[best_asset_idx]
    worst_asset_ticker = config.data.tickers[worst_asset_idx]

    # B1/B2: portfolio variance / max drawdown, pooled real vs pooled gen.
    var_real = np.array([portfolio_variance(real_raw_AT[i]) for i in range(len(real_raw_AT))])
    var_gen = np.array([portfolio_variance(gen_raw_AT[i]) for i in range(len(gen_raw_AT))])
    mdd_real = np.array([max_drawdown(real_raw_AT[i]) for i in range(len(real_raw_AT))])
    mdd_gen = np.array([max_drawdown(gen_raw_AT[i]) for i in range(len(gen_raw_AT))])

    var_w1 = float(wasserstein_distance(var_real, var_gen))
    var_meandiff = float(abs(var_gen.mean() - var_real.mean()))
    mdd_w1 = float(wasserstein_distance(mdd_real, mdd_gen))
    mdd_meandiff = float(abs(mdd_gen.mean() - mdd_real.mean()))

    # B3: POOLED pinball loss -- the generated quantiles come from the single
    # pooled generated distribution (no per-window generated distribution
    # exists here), scored against EACH real event window's own realized
    # cumulative return, then averaged. Documented as "pooled" specifically
    # because this differs from Experiment 1's per-window pinball loss.
    gen_cum_returns = np.array([portfolio_cum_return(gen_raw_AT[i]) for i in range(len(gen_raw_AT))])
    real_cum_returns = np.array([portfolio_cum_return(real_raw_AT[i]) for i in range(len(real_raw_AT))])
    pinball_avg = {}
    for q in QUANTILES:
        q_hat = float(np.quantile(gen_cum_returns, q))
        pinball_avg[q] = float(np.mean([pinball_loss(y, q_hat, q) for y in real_cum_returns]))

    row = {
        "channel": args.channel,
        "split": args.split,
        "n_real_events": n_events,
        "n_generated": n_gen,
        "energy_score": energy,
        "wasserstein1_marginal": w1_marginal,
        "wasserstein1_marginal_best": w1_marginal_best,
        "wasserstein1_marginal_best_ticker": best_asset_ticker,
        "wasserstein1_marginal_worst": w1_marginal_worst,
        "wasserstein1_marginal_worst_ticker": worst_asset_ticker,
        # Real vs generated side-by-side -- a small distance metric alone doesn't tell
        # you whether that's because generated is genuinely CLOSE to real, or because
        # both real and generated happen to be small in absolute terms. Median is
        # reported alongside mean since a handful of exploding generated samples (see
        # the no-Q-model guidance-instability note) pulls the mean but not the median.
        "portfolio_variance_real_mean": float(var_real.mean()),
        "portfolio_variance_gen_mean": float(var_gen.mean()),
        "portfolio_variance_real_median": float(np.median(var_real)),
        "portfolio_variance_gen_median": float(np.median(var_gen)),
        "portfolio_variance_w1": var_w1,
        "portfolio_variance_meandiff": var_meandiff,
        "portfolio_mdd_real_mean": float(mdd_real.mean()),
        "portfolio_mdd_gen_mean": float(mdd_gen.mean()),
        "portfolio_mdd_real_median": float(np.median(mdd_real)),
        "portfolio_mdd_gen_median": float(np.median(mdd_gen)),
        "portfolio_mdd_w1": mdd_w1,
        "portfolio_mdd_meandiff": mdd_meandiff,
        "portfolio_cumret_real_mean": float(real_cum_returns.mean()),
        "portfolio_cumret_gen_mean": float(gen_cum_returns.mean()),
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
