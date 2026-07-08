"""
Compare portfolio returns of event sequences in training set vs test set.
No generation — real data only.

Usage:
  python compare_train_test_events.py
  python compare_train_test_events.py --results-dir results/event_compare
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_default_config
from data import DataProcessor
from utils import PortfolioAnalyzer, set_seed


def main(args):
    config = get_default_config()
    set_seed(config.seed)

    results_dir = args.results_dir or "results/event_compare"
    os.makedirs(results_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    data_processor = DataProcessor(
        csv_path=config.data.csv_path,
        tickers=config.data.tickers,
        weekday_col=config.data.weekday_col,
        seq_len=config.data.seq_len,
        test_days=config.data.test_days,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        train_end_date=config.data.train_end_date,
        winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper,
    )
    data_processor.process_all()

    # event_threshold is specified as "top X% of |Z_end - Z_start|", converted here
    # to the equivalent raw numeric cutoff (train-only, no leakage) — see main.py.
    event_top_fraction = config.hfunction.event_threshold
    config.hfunction.event_threshold = data_processor.get_event_threshold_from_percentile(event_top_fraction)
    print(f"Event threshold: top {event_top_fraction:.1%} -> {config.hfunction.event_threshold:.4f} std")

    # ── Event masks ───────────────────────────────────────────────────────────
    # Event mask must come from the real macro series (via get_z_windows),
    # not from X, which is stock-returns-only and has no macro channel.
    X_train = data_processor.X_train   # (N_train, 64, 4)
    X_test  = data_processor.X_test    # (N_test,  64, 4)

    Z_start_train, Z_end_train, valid_idx_train = data_processor.get_z_windows_train_aligned()
    Z_start_test,  Z_end_test,  valid_idx_test  = data_processor.get_z_windows_test()

    if config.hfunction.event_type == "change":
        event_valid_train = (Z_end_train - Z_start_train).abs() >= config.hfunction.event_threshold
        event_valid_test  = (Z_end_test  - Z_start_test).abs()  >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        event_valid_train = Z_end_train.abs() >= config.hfunction.event_threshold
        event_valid_test  = Z_end_test.abs()  >= config.hfunction.event_threshold
    else:
        raise NotImplementedError(
            f"event_type={config.hfunction.event_type!r} not supported by the "
            "macro-based mask; only 'change' and 'absval' are implemented."
        )
    mask_train = torch.zeros(X_train.shape[0], dtype=torch.bool)
    mask_train[valid_idx_train] = event_valid_train
    mask_test = torch.zeros(X_test.shape[0], dtype=torch.bool)
    mask_test[valid_idx_test] = event_valid_test

    n_train_ev = int(mask_train.sum())
    n_test_ev  = int(mask_test.sum())
    print(f"\nTrain: {X_train.shape[0]} sequences, {n_train_ev} events "
          f"({n_train_ev / X_train.shape[0]:.2%})")
    print(f"Test:  {X_test.shape[0]}  sequences, {n_test_ev}  events "
          f"({n_test_ev  / X_test.shape[0]:.2%})")

    if n_train_ev == 0 or n_test_ev == 0:
        print("Not enough events in one of the splits — check threshold or data window.")
        return

    # ── Portfolio returns ─────────────────────────────────────────────────────
    portfolio_analyzer = PortfolioAnalyzer(
        data_processor=data_processor,
        window_for_cov=config.portfolio.window_for_cov,
        last_days_sum=config.portfolio.last_days_sum,
    )

    print("\nAnalyzing train event sequences...")
    tr_mv, tr_rp, tr_avg = portfolio_analyzer.analyze_test_set(
        X_train, mask_train, start_weekdays=data_processor.start_weekdays_train
    )

    print("Analyzing test event sequences...")
    te_mv, te_rp, te_avg = portfolio_analyzer.analyze_test_set(
        X_test, mask_test, start_weekdays=data_processor.start_weekdays_test
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVENT PORTFOLIO COMPARISON: TRAIN vs TEST")
    print("=" * 70)
    portfolio_analyzer.summarize_statistics("TRAIN events", tr_mv, tr_rp, tr_avg)
    portfolio_analyzer.summarize_statistics("TEST  events", te_mv, te_rp, te_avg)

    print("\n── Mean |diff| per strategy ──")
    for name, tr, te in [("EW", tr_avg, te_avg), ("MV", tr_mv, te_mv), ("RP", tr_rp, te_rp)]:
        diff = abs(np.mean(tr) - np.mean(te))
        print(f"  {name}: train_mean={np.mean(tr):.5f}  test_mean={np.mean(te):.5f}  |diff|={diff:.5f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    strategies = [
        ("Equal-Weight", tr_avg, te_avg),
        ("Min-Variance", tr_mv,  te_mv),
        ("Risk-Parity",  tr_rp,  te_rp),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, tr, te) in zip(axes, strategies):
        tr_arr = np.array(tr)
        te_arr = np.array(te)
        tr_bins = np.linspace(tr_arr.min(), tr_arr.max(), 60)
        te_bins = np.linspace(te_arr.min(), te_arr.max(), 60)

        ax.hist(tr_arr, bins=tr_bins, alpha=0.55, density=True, color="C0",
                label=f"Train events (N={len(tr_arr)})")
        ax.hist(te_arr, bins=te_bins, alpha=0.55, density=True, color="C1",
                label=f"Test events (N={len(te_arr)})")
        ax.axvline(tr_arr.mean(), color="C0", lw=1.5, ls="--")
        ax.axvline(te_arr.mean(), color="C1", lw=1.5, ls="--")
        ax.set_title(f"{name}\nmean diff={abs(tr_arr.mean()-te_arr.mean()):.4f}", fontsize=11)
        ax.set_xlabel(f"Last {config.portfolio.last_days_sum}-day portfolio return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Event Portfolio: Train vs Test\n"
        f"({config.data.start_date} ~ {config.data.train_end_date} | "
        f"{config.data.train_end_date} ~ {config.data.end_date})",
        fontsize=12,
    )
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "train_test_event_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
