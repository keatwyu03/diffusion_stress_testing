"""
Pre-train diffusion model and plot return histograms.

Generates unconditional samples from the trained diffusion model and
produces two figures:
  1. Per-stock standardized return distributions (generated vs real)
  2. Portfolio return distributions for three strategies (generated vs real)

Usage:
  python pretrain_and_plot.py                        # train + plot
  python pretrain_and_plot.py --skip-training        # load checkpoint + plot
  python pretrain_and_plot.py --n-epochs 100         # quick training run
  python pretrain_and_plot.py --n-samples 1000       # more generated samples
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel
from utils import PortfolioAnalyzer, set_seed


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _shared_bins(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Compute n+1 bin edges covering the combined range of a and b."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    return np.linspace(lo, hi, n + 1)


def plot_stock_histograms(
    generated: torch.Tensor,
    real_data: torch.Tensor,
    tickers: list,
    n_bins: int = 100,
    save_path: str = None,
) -> None:
    """
    Histogram of per-stock standardized daily returns.

    generated : (N_gen, 4, 64)  – standardized generated windows
    real_data  : (N_real, 4, 64) – standardized real training windows
    """
    gen_np  = generated.cpu().numpy()
    real_np = real_data.cpu().numpy()

    fig, axes = plt.subplots(1, len(tickers), figsize=(5 * len(tickers), 4))
    if len(tickers) == 1:
        axes = [axes]

    for i, (ax, ticker) in enumerate(zip(axes, tickers)):
        real_vals = real_np[:, i, :].flatten()
        gen_vals  = gen_np[:, i, :].flatten()
        bins = _shared_bins(real_vals, gen_vals, n_bins)

        ax.hist(real_vals, bins=bins, alpha=0.55, density=True,
                color="C1", label="Real")
        ax.hist(gen_vals,  bins=bins, alpha=0.55, density=True,
                color="C0", label="Generated")
        ax.set_title(ticker, fontsize=13)
        ax.set_xlabel("Standardized log return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Per-Stock Return Distribution: Generated vs Real",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_portfolio_histograms(
    gen_mv:  list, gen_rp:  list, gen_avg:  list,
    real_mv: list, real_rp: list, real_avg: list,
    last_days: int,
    n_bins: int = 80,
    save_path: str = None,
) -> None:
    """
    Histogram of portfolio last-N-day returns for three strategies.
    """
    strategies = [
        ("Equal-Weight", gen_avg,  real_avg),
        ("Min-Variance", gen_mv,   real_mv),
        ("Risk-Parity",  gen_rp,   real_rp),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (name, gen, real) in zip(axes, strategies):
        gen_arr  = np.array(gen)
        real_arr = np.array(real)
        bins = _shared_bins(real_arr, gen_arr, n_bins)

        ax.hist(real_arr, bins=bins, alpha=0.55, density=True,
                color="C1", label="Real")
        ax.hist(gen_arr,  bins=bins, alpha=0.55, density=True,
                color="C0", label="Generated")
        ax.set_title(f"{name}\n(last {last_days}-day sum)", fontsize=12)
        ax.set_xlabel("Weighted log-return sum")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Portfolio Return Distribution: Generated vs Real",
                 fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = get_default_config()
    device = args.device or get_device()

    # Override device everywhere in config
    config.diffusion.device   = device
    config.hfunction.device   = device
    config.conditional.device = device

    # Override training epochs if requested
    if args.n_epochs is not None:
        config.diffusion.n_epochs = args.n_epochs

    config.wandb.enabled = False   # no wandb for this script
    set_seed(config.seed)

    print(f"Device: {device}")
    print(f"Tickers: {config.data.tickers}")

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("\n[1/3] Loading and processing data...")
    dp = DataProcessor(
        csv_path=config.data.csv_path,
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
    )
    dp.process_all()

    # Training windows for the diffusion model: shape (N, 4, 64)
    X_diffusion_train = dp.get_diffusion_data()
    print(f"Diffusion training windows: {X_diffusion_train.shape}")

    # ------------------------------------------------------------------ #
    # 2. Diffusion model: train or load
    # ------------------------------------------------------------------ #
    diffusion = DiffusionModel(
        in_channels=config.diffusion.in_channels,
        out_channels=config.diffusion.out_channels,
        sample_size=config.diffusion.sample_size,
        layers_per_block=config.diffusion.layers_per_block,
        block_out_channels=config.diffusion.block_out_channels,
        b_min=config.diffusion.b_min,
        b_max=config.diffusion.b_max,
        device=device,
        arch=config.diffusion.arch,
        embed_dim=config.diffusion.embed_dim,
        n_heads=config.diffusion.n_heads,
        n_layers=config.diffusion.n_layers,
        cond_dim=config.diffusion.cond_dim,
    )

    ckpt_path = "ckpt_new/diffusion_model.pt"

    if args.skip_training and os.path.exists(ckpt_path):
        print(f"\n[2/3] Loading diffusion model from {ckpt_path} ...")
        diffusion.load(ckpt_path)
    else:
        print(f"\n[2/3] Training diffusion model for "
              f"{config.diffusion.n_epochs} epochs ...")
        diffusion.train(
            train_data=X_diffusion_train,
            batch_size=config.diffusion.batch_size,
            n_epochs=config.diffusion.n_epochs,
            learning_rate=config.diffusion.learning_rate,
            scheduler_patience=config.diffusion.scheduler_patience,
            scheduler_factor=config.diffusion.scheduler_factor,
            use_wandb=False,
        )
        os.makedirs("ckpt_new", exist_ok=True)
        diffusion.save(ckpt_path)

    # ------------------------------------------------------------------ #
    # 3. Unconditional generation
    # ------------------------------------------------------------------ #
    # n_samples: default to test set size so generated vs real counts always match
    X_test = dp.X_test
    n_samples = args.n_samples if args.n_samples is not None else X_test.shape[0]

    print(f"\n[3/3] Generating {n_samples} unconditional samples ...")
    generated = diffusion.sample(
        batch_size=n_samples,
        num_steps=config.diffusion.num_steps,
        stoch=0.5,
        eps=config.diffusion.eps,
    )
    print(f"Generated shape: {generated.shape}")   # (N, 4, 64)

    # ------------------------------------------------------------------ #
    # 4. Plots
    # ------------------------------------------------------------------ #
    os.makedirs("results", exist_ok=True)

    # Draw a fixed random subset of test windows used consistently in both plots
    n_real = min(n_samples, X_test.shape[0])
    idx_real = torch.randperm(X_test.shape[0])[:n_real]
    X_real_subset = X_test[idx_real]              # (n_real, 64, 4)

    # --- Plot A: per-stock return histograms ---
    print("\nPlotting per-stock return histograms...")
    # X_test: (N, 64, 4) → permute to (N, 4, 64) for stock axis
    plot_stock_histograms(
        generated=generated,
        real_data=X_real_subset.permute(0, 2, 1),  # (n_real, 4, 64)
        tickers=config.data.tickers,
        save_path="results/stock_return_histograms.png",
    )

    # --- Plot B: portfolio return histograms ---
    print("Plotting portfolio return histograms...")

    portfolio_analyzer = PortfolioAnalyzer(
        data_processor=dp,
        window_for_cov=config.portfolio.window_for_cov,
        last_days_sum=config.portfolio.last_days_sum,
    )

    # Generated samples (N, 4, 64)
    gen_mv, gen_rp, gen_avg = portfolio_analyzer.analyze_samples(generated)

    # Real test windows — same subset as stock histograms
    mask_all = torch.ones(n_real, dtype=torch.bool)

    real_mv, real_rp, real_avg = portfolio_analyzer.analyze_test_set(
        X_real_subset, mask_all
    )

    portfolio_analyzer.summarize_statistics(
        "GENERATED", gen_mv, gen_rp, gen_avg
    )
    portfolio_analyzer.summarize_statistics(
        "REAL TEST", real_mv, real_rp, real_avg
    )

    plot_portfolio_histograms(
        gen_mv, gen_rp, gen_avg,
        real_mv, real_rp, real_avg,
        last_days=config.portfolio.last_days_sum,
        save_path="results/portfolio_return_histograms.png",
    )

    print("\nDone. Figures saved to results/")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train diffusion model and plot return histograms"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and load diffusion model from ckpt_new/",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=None,
        help="Override number of training epochs (default: use config)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of unconditional samples to generate (default: match test set size)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / mps / cpu (default: auto-detect)",
    )
    args = parser.parse_args()
    main(args)
