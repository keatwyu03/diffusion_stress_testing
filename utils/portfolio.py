"""
Portfolio analysis utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Callable


class PortfolioAnalyzer:
    """Analyze and compare portfolio strategies"""

    def __init__(
        self,
        data_processor,
        window_for_cov: int = 54,
        last_days_sum: int = 5,
    ):
        """
        Initialize Portfolio Analyzer

        Args:
            data_processor: DataProcessor instance
            window_for_cov: Number of days to compute covariance
            last_days_sum: Number of last days to sum for portfolio
        """
        self.data_processor = data_processor
        self.window_for_cov = window_for_cov
        self.last_days_sum = last_days_sum

    @staticmethod
    def minvar_weights(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute minimum variance portfolio weights"""
        D = cov.shape[0]
        cov = cov + eps * np.eye(D)
        ones = np.ones(D)
        w_unnorm = np.linalg.solve(cov, ones)
        w = w_unnorm / w_unnorm.sum()
        return w

    @staticmethod
    def risk_parity_weights(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute risk parity portfolio weights"""
        std = np.sqrt(np.clip(np.diag(cov), eps, None))
        inv_vol = 1.0 / std
        w = inv_vol / inv_vol.sum()
        return w

    def last_days_sum_with_weights(
        self, r_seq_values: np.ndarray, w: np.ndarray
    ) -> float:
        """Compute sum of last N days weighted by portfolio weights"""
        return float((r_seq_values[-self.last_days_sum :] @ w).sum())

    def analyze_samples(
        self, samples: torch.Tensor
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Analyze portfolio returns for generated samples

        Args:
            samples: Generated samples (N, channels, seq_len)

        Returns:
            mv_sums: Min-variance portfolio sums
            rp_sums: Risk-parity portfolio sums
            avg_sums: Equal-weight portfolio sums
        """
        mv_sums, rp_sums, avg_sums = [], [], []

        for n in range(samples.shape[0]):
            # Transpose to (seq_len, channels)
            sample = samples[n].T
            r_seq, _, _, _ = self.data_processor.invert_samples(sample)
            R = r_seq.values

            # Compute covariance from first window_for_cov days
            cov = np.cov(R[: self.window_for_cov, :], rowvar=False, ddof=1)

            # Compute weights
            w_mv = self.minvar_weights(cov)
            w_rp = self.risk_parity_weights(cov)
            w_avg = (1.0 / len(self.data_processor.tickers)) * np.ones(
                len(self.data_processor.tickers)
            )

            # Compute sums
            mv_sums.append(self.last_days_sum_with_weights(R, w_mv))
            rp_sums.append(self.last_days_sum_with_weights(R, w_rp))
            avg_sums.append(self.last_days_sum_with_weights(R, w_avg))

        return mv_sums, rp_sums, avg_sums

    def analyze_test_set(
        self, X_test: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Analyze portfolio returns for test set

        Args:
            X_test: Test set samples
            mask: Boolean mask for events

        Returns:
            mv_sums: Min-variance portfolio sums
            rp_sums: Risk-parity portfolio sums
            avg_sums: Equal-weight portfolio sums
        """
        mv_sums, rp_sums, avg_sums = [], [], []

        for n in np.array(np.nonzero(mask.cpu().numpy())).ravel():
            sample = X_test[n]  # Shape: (seq_len, channels) = (64, 4)
            r_seq, _, _, _ = self.data_processor.invert_samples(sample)
            R = r_seq.values

            cov = np.cov(R[: self.window_for_cov, :], rowvar=False, ddof=1)

            w_mv = self.minvar_weights(cov)
            w_rp = self.risk_parity_weights(cov)
            w_avg = (1.0 / len(self.data_processor.tickers)) * np.ones(
                len(self.data_processor.tickers)
            )

            mv_sums.append(self.last_days_sum_with_weights(R, w_mv))
            rp_sums.append(self.last_days_sum_with_weights(R, w_rp))
            avg_sums.append(self.last_days_sum_with_weights(R, w_avg))

        return mv_sums, rp_sums, avg_sums

    @staticmethod
    def summarize_statistics(
        name: str, mv: List[float], rp: List[float], avg: List[float]
    ) -> None:
        """Print summary statistics"""
        mv_mean, mv_std = np.mean(mv), np.std(mv)
        rp_mean, rp_std = np.mean(rp), np.std(rp)
        avg_mean, avg_std = np.mean(avg), np.std(avg)

        mv_median = np.median(mv)
        rp_median = np.median(rp)
        avg_median = np.median(avg)

        mv_p5, mv_p10 = np.percentile(mv, 5), np.percentile(mv, 10)
        rp_p5, rp_p10 = np.percentile(rp, 5), np.percentile(rp, 10)
        avg_p5, avg_p10 = np.percentile(avg, 5), np.percentile(avg, 10)

        print(
            f"[{name}] N={len(mv)} | "
            f"MV mean={mv_mean:.4g}, median={mv_median:.4g}, std={mv_std:.4g}, "
            f"p5={mv_p5:.4g}, p10={mv_p10:.4g} | "
            f"RP mean={rp_mean:.4g}, median={rp_median:.4g}, std={rp_std:.4g}, "
            f"p5={rp_p5:.4g}, p10={rp_p10:.4g} | "
            f"Avg mean={avg_mean:.4g}, median={avg_median:.4g}, std={avg_std:.4g}, "
            f"p5={avg_p5:.4g}, p10={avg_p10:.4g}"
        )

    @staticmethod
    def plot_comparison(
        gen_mv: List[float],
        gen_rp: List[float],
        gen_avg: List[float],
        real_mv: List[float],
        real_rp: List[float],
        real_avg: List[float],
        save_path: str = None,
        n_bins: int = 80,
    ) -> None:
        """Plot portfolio comparison with shared bin edges and density normalisation."""
        strategies = [
            ("Equal-Weight",  gen_avg, real_avg),
            ("Min-Variance",  gen_mv,  real_mv),
            ("Risk-Parity",   gen_rp,  real_rp),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

        for ax, (name, gen, real) in zip(axes, strategies):
            gen_arr  = np.array(gen)
            real_arr = np.array(real)
            lo = min(gen_arr.min(), real_arr.min())
            hi = max(gen_arr.max(), real_arr.max())
            bins = np.linspace(lo, hi, n_bins + 1)

            ax.hist(real_arr, bins=bins, alpha=0.55, density=True,
                    color="C1", label="Real")
            ax.hist(gen_arr,  bins=bins, alpha=0.55, density=True,
                    color="C0", label="Generated")
            ax.set_title(f"{name} (last 5-day sum)", fontsize=12)
            ax.set_xlabel("Sum of Log Returns")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()
