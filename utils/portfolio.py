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
        config=None,
    ):
        self.data_processor = data_processor
        self.window_for_cov = window_for_cov
        self.last_days_sum = last_days_sum
        self.config = config

    @staticmethod
    def minvar_weights(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute minimum variance portfolio weights"""
        if cov.ndim == 0:
            return np.array([1.0])

        D = cov.shape[0]
        cov = cov + eps * np.eye(D)
        ones = np.ones(D)
        w_unnorm = np.linalg.solve(cov, ones)
        w = w_unnorm / w_unnorm.sum()
        return w

    @staticmethod
    def risk_parity_weights(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute risk parity portfolio weights"""
        if cov.ndim == 0:
            return np.array([1.0])
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

        portfolio_tickers = self.config.portfolio.portfolio_tickers
        n_tickers = len(portfolio_tickers)

        for n in range(samples.shape[0]):
            sample = samples[n].T
            r_seq, _, _, _ = self.data_processor.invert_samples(sample, monthly=False)
            r_seq = r_seq[portfolio_tickers]
            R = r_seq.values

            cov = np.cov(R[: self.window_for_cov, :], rowvar=False, ddof=1)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)

            w_mv = self.minvar_weights(cov)
            w_rp = self.risk_parity_weights(cov)
            w_avg = (1.0 / n_tickers) * np.ones(n_tickers)

            mv_sums.append(self.last_days_sum_with_weights(R, w_mv))
            rp_sums.append(self.last_days_sum_with_weights(R, w_rp))
            avg_sums.append(self.last_days_sum_with_weights(R, w_avg))

        return mv_sums, rp_sums, avg_sums

    def analyze_test_set(
        self, X_test: torch.Tensor, mask: torch.Tensor, start_weekdays=None
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Analyze portfolio returns for test set.

        start_weekdays: int array (same length as X_test) with the weekday of
                        the first day of each sequence.
        """
        mv_sums, rp_sums, avg_sums = [], [], []

        portfolio_tickers = self.config.portfolio.portfolio_tickers
        n_tickers = len(portfolio_tickers)

        for n in np.array(np.nonzero(mask.cpu().numpy())).ravel():
            sample = X_test[n]

            start_weekday = int(start_weekdays[n]) if start_weekdays is not None else None
            r_seq, _, _, _ = self.data_processor.invert_samples(
                sample, monthly=False, start_weekday=start_weekday
            )
            r_seq = r_seq[portfolio_tickers]
            R = r_seq.values

            cov = np.cov(R[: self.window_for_cov, :], rowvar=False, ddof=1)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)

            w_mv = self.minvar_weights(cov)
            w_rp = self.risk_parity_weights(cov)
            w_avg = (1.0 / n_tickers) * np.ones(n_tickers)

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
    def build_stats_df(
        gen_mv: List[float], gen_rp: List[float], gen_avg: List[float],
        real_mv: List[float], real_rp: List[float], real_avg: List[float],
    ) -> "pd.DataFrame":
        """Build a DataFrame of summary statistics for generated and real portfolios."""
        rows = []
        for source, mv, rp, avg in [
            ("Generated", gen_mv, gen_rp, gen_avg),
            ("Real",      real_mv, real_rp, real_avg),
        ]:
            for strategy, vals in [
                ("Equal-Weight", avg),
                ("Min-Variance", mv),
                ("Risk-Parity",  rp),
            ]:
                arr = np.array(vals)
                rows.append({
                    "source":   source,
                    "strategy": strategy,
                    "N":        len(arr),
                    "mean":     arr.mean(),
                    "median":   np.median(arr),
                    "std":      arr.std(),
                    "p5":       np.percentile(arr, 5),
                    "p10":      np.percentile(arr, 10),
                    "p90":      np.percentile(arr, 90),
                    "p95":      np.percentile(arr, 95),
                })
        return pd.DataFrame(rows)

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
        gen_label: str = "Conditional Generated",
        real_label: str = "Real",
    ) -> None:
        """Plot portfolio comparison with shared bin edges and density normalisation."""
        strategies = [
            ("Equal-Weight", gen_avg, real_avg),
            ("Min-Variance", gen_mv,  real_mv),
            ("Risk-Parity",  gen_rp,  real_rp),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

        for ax, (name, gen, real) in zip(axes, strategies):
            gen_arr  = np.array(gen)
            real_arr = np.array(real)
            bins_actual = min(n_bins, max(15, min(len(gen_arr), len(real_arr)) // 2))
            real_bins = np.linspace(real_arr.min(), real_arr.max(), bins_actual + 1)
            gen_bins  = np.linspace(gen_arr.min(),  gen_arr.max(),  bins_actual + 1)

            ax.hist(real_arr, bins=real_bins, alpha=0.55, density=True,
                    color="C1", label=real_label)
            ax.hist(gen_arr,  bins=gen_bins,  alpha=0.55, density=True,
                    color="C0", label=gen_label)
            ax.set_title(f"{name} (last 5-day sum)", fontsize=12)
            ax.set_xlabel("Sum of Log Returns")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()
