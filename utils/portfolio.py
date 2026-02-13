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
    ) -> None:
        """
        Plot portfolio comparison

        Args:
            gen_mv: Generated min-variance sums
            gen_rp: Generated risk-parity sums
            gen_avg: Generated equal-weight sums
            real_mv: Real min-variance sums
            real_rp: Real risk-parity sums
            real_avg: Real equal-weight sums
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        # Equal-Weight
        axes[0].hist(gen_avg, bins=30, alpha=0.6, label="Generated", color="C0")
        axes[0].hist(
            real_avg, bins=30, alpha=0.6, label="Real (Test Set)", color="C1"
        )
        axes[0].set_title(f"Equal-Weight Portfolio (last {5}-day sum)")
        axes[0].set_xlabel("Sum of Log Returns")
        axes[0].legend()

        # Min-Variance
        axes[1].hist(gen_mv, bins=30, alpha=0.6, label="Generated", color="C0")
        axes[1].hist(real_mv, bins=30, alpha=0.6, label="Real (Test Set)", color="C1")
        axes[1].set_title(f"Min-Variance Portfolio (last {5}-day sum)")
        axes[1].set_xlabel("Sum of Log Returns")
        axes[1].legend()

        # Risk-Parity
        axes[2].hist(gen_rp, bins=30, alpha=0.6, label="Generated", color="C0")
        axes[2].hist(real_rp, bins=30, alpha=0.6, label="Real (Test Set)", color="C1")
        axes[2].set_title(f"Risk-Parity Portfolio (last {5}-day sum)")
        axes[2].set_xlabel("Sum of Log Returns")
        axes[2].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()
