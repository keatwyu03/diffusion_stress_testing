"""Baseline rolling-local-periodogram Q_n(u) plot from Paparoditis (2010).

This script implements only the time-indexed diagnostic curve in equations
(2.3)-(2.5) of:

    E. Paparoditis (2010), "Validating Stationarity Assumptions in Time
    Series Analysis by Rolling Local Periodograms," JASA 105(490), 839-851.

It intentionally does not implement the bootstrap test, a critical threshold,
stage/change diagnosis, or cross-validated bandwidth selection.

The input series are demeaned because the paper assumes a mean-zero process.
Default implementation choices follow the paper's numerical guidance:

    N = 128
    h = 0.05
    b = (n/N)^(1/5) h
    cosine taper over the first and last 20% of each local window
    Bartlett-Priestley smoothing kernel

Kernel convention
-----------------
The paper writes both smoothing operations with prefactors 1/n and 1/N. We
therefore use the spectral-analysis normalization

    integral_{-pi}^{pi} K(x) dx = 2*pi,

so that

    K(x) = (3/2) * (1 - x^2/pi^2) 1{|x| <= pi}.

This is algebraically equivalent to using the unit-integral form
3/(4*pi)*(1-x^2/pi^2) together with prefactors 2*pi/n and 2*pi/N. The present
form preserves the paper's displayed 1/n and 1/N equations directly.

Example
-------
python paparoditis_qn_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TICKERS = ("AAPL", "TXN")
DEFAULT_WINDOW = 128
DEFAULT_H = 0.05
DEFAULT_TAPER_FRACTION = 0.20
DEFAULT_INTEGRATION_POINTS = 1025


def centered_fourier_indices(length: int) -> np.ndarray:
    """Integer indices used by the paper's centered Fourier grid."""
    return np.arange(-((length - 1) // 2), length // 2 + 1)


def fourier_frequencies(length: int) -> np.ndarray:
    """lambda_j = 2*pi*j/length on the paper's centered Fourier grid."""
    return 2.0 * np.pi * centered_fourier_indices(length) / length


def circular_difference(x: np.ndarray) -> np.ndarray:
    """Return frequency differences on the 2*pi-periodic frequency circle."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def bartlett_priestley_kernel(x: np.ndarray) -> np.ndarray:
    """Bartlett-Priestley kernel with integral 2*pi on [-pi, pi]."""
    x = np.asarray(x, dtype=float)
    inside = np.abs(x) <= np.pi
    values = np.zeros_like(x)
    values[inside] = 1.5 * (1.0 - (x[inside] / np.pi) ** 2)
    return values


def scaled_kernel(difference: np.ndarray, bandwidth: float) -> np.ndarray:
    """K_bandwidth(x) = bandwidth^(-1) K(x / bandwidth)."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    difference = circular_difference(np.asarray(difference, dtype=float))
    return bartlett_priestley_kernel(difference / bandwidth) / bandwidth


def cosine_taper(length: int, edge_fraction: float = 0.20) -> np.ndarray:
    """Cosine taper tau(t/N), covering the first/last edge_fraction, scaled
    so mean(tau^2) = 1 (so E[2*pi*I_{N,epsilon}] = 1, per the discussion
    following eq. (2.2))."""
    if not 0.0 < edge_fraction < 0.5:
        raise ValueError("edge_fraction must lie strictly between 0 and 0.5")

    x = np.arange(1, length + 1, dtype=float) / length
    taper = np.ones(length, dtype=float)

    left = x < edge_fraction
    taper[left] = 0.5 * (1.0 - np.cos(np.pi * x[left] / edge_fraction))

    right = x > 1.0 - edge_fraction
    taper[right] = 0.5 * (
        1.0 - np.cos(np.pi * (1.0 - x[right]) / edge_fraction)
    )

    return taper / np.sqrt(np.mean(taper**2))


def global_periodogram(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Global periodogram from the display immediately below equation (2.4).

    I_n(lambda_j) = (1/(2*pi*n)) |sum_{t=1}^n X_t exp(-i*t*lambda_j)|^2.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    indices = centered_fourier_indices(n)
    frequencies = 2.0 * np.pi * indices / n
    dft = np.fft.fft(x)[indices % n]
    periodogram = np.abs(dft) ** 2 / (2.0 * np.pi * n)
    return frequencies, periodogram


def global_spectral_estimate(
    evaluation_frequencies: np.ndarray,
    global_frequencies: np.ndarray,
    global_periodogram_values: np.ndarray,
    h: float,
) -> np.ndarray:
    r"""Equation (2.4): f_hat(lambda) = n^(-1) sum_j K_h(lambda-lambda_j) I_n(lambda_j)."""
    differences = (
        np.asarray(evaluation_frequencies)[:, None]
        - np.asarray(global_frequencies)[None, :]
    )
    weights = scaled_kernel(differences, h)
    return weights @ np.asarray(global_periodogram_values) / global_frequencies.size


def local_periodograms(
    x: np.ndarray,
    window_length: int,
    taper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Paper's tapered local periodogram for every admissible time s.

    I_N(u,lambda_j) = (1/(2*pi*N))
        |sum_{t=1}^N tau(t/N) X_[un]+t-M_N exp(-i*t*lambda_j)|^2.

    Returns the paper time indices s, local Fourier frequencies, and an array
    whose rows are rolling windows and columns are local frequencies.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    N = int(window_length)
    if N > n:
        raise ValueError(f"window length N={N} exceeds series length n={n}")
    if taper.shape != (N,):
        raise ValueError("taper length must equal window_length")

    windows = np.lib.stride_tricks.sliding_window_view(x, N) * taper[None, :]
    indices = centered_fourier_indices(N)
    frequencies = 2.0 * np.pi * indices / N
    dft = np.fft.fft(windows, axis=1)[:, indices % N]
    values = np.abs(dft) ** 2 / (2.0 * np.pi * N)

    paper_time_indices = np.arange(windows.shape[0]) + N // 2
    return paper_time_indices, frequencies, values


def qn_curve(
    x: np.ndarray,
    window_length: int = DEFAULT_WINDOW,
    h: float = DEFAULT_H,
    b: float | None = None,
    integration_points: int = DEFAULT_INTEGRATION_POINTS,
    taper_fraction: float = DEFAULT_TAPER_FRACTION,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Calculate Q_n(u_s) from equations (2.3)-(2.5) for all valid s.

    q_hat_n(u_s,lambda) = (1/N) sum_j K_b(lambda-lambda_j)
                           [I_N(u_s,lambda_j)/f_hat(lambda_j) - 1],

    Q_n(u_s) = integral_{-pi}^{pi} q_hat_n(u_s,lambda)^2 d lambda.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite values")

    x = x - np.mean(x)
    n = x.size
    N = int(window_length)
    if b is None:
        b = (n / N) ** (1.0 / 5.0) * h

    taper = cosine_taper(N, taper_fraction)
    global_freqs, global_I = global_periodogram(x)
    time_indices, local_freqs, local_I = local_periodograms(x, N, taper)

    f_hat_local = global_spectral_estimate(local_freqs, global_freqs, global_I, h)
    if np.any(~np.isfinite(f_hat_local)) or np.any(f_hat_local <= 0.0):
        raise ValueError(
            "The global spectral estimate is nonpositive. Increase h; no "
            "clipping is applied because that would change the statistic."
        )

    ratio_minus_one = local_I / f_hat_local[None, :] - 1.0

    lambda_grid = np.linspace(-np.pi, np.pi, int(integration_points))
    smoothing_weights = scaled_kernel(
        lambda_grid[:, None] - local_freqs[None, :], b
    )

    q_hat = smoothing_weights @ ratio_minus_one.T / N
    Q = np.trapezoid(q_hat**2, lambda_grid, axis=0)
    return time_indices, Q, float(b)


def load_series(
    csv_path: Path,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """Load one ticker and its dates, dropping only rows missing that ticker,
    optionally restricted to [start_date, end_date] (inclusive)."""
    frame = pd.read_csv(csv_path, usecols=["Date", ticker], parse_dates=["Date"])
    frame = frame.dropna(subset=["Date", ticker]).sort_values("Date").reset_index(drop=True)
    if start_date is not None:
        frame = frame[frame["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(end_date)]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No usable observations were found for {ticker}")
    return frame["Date"], frame[ticker].to_numpy(dtype=float)


def plot_tickers(
    csv_path: Path,
    output_dir: Path,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    window_length: int = DEFAULT_WINDOW,
    h: float = DEFAULT_H,
    b: float | None = None,
    integration_points: int = DEFAULT_INTEGRATION_POINTS,
    start_date: str | None = None,
    end_date: str | None = None,
    name_suffix: str = "",
) -> list[Path]:
    """Calculate and plot Q_n(u) for each ticker, one PNG per ticker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {"AAPL": "ROLPER_apple", "TXN": "ROLPER_txn"}
    output_paths = []

    for ticker in tickers:
        dates, x = load_series(csv_path, ticker, start_date=start_date, end_date=end_date)
        time_indices, Q, _ = qn_curve(
            x,
            window_length=window_length,
            h=h,
            b=b,
            integration_points=integration_points,
        )
        plot_dates = dates.iloc[time_indices - 1].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(13.0, 3.8), constrained_layout=True)
        ax.plot(plot_dates, Q, color="#2468A2", linewidth=0.85)
        ax.fill_between(plot_dates, 0.0, Q, color="#78A9D1", alpha=0.18)
        ax.set_ylabel(r"$Q_n(u_s)$")
        ax.set_xlabel("Date")
        ax.grid(True, color="#D7DCE2", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        name = output_names.get(ticker, ticker)
        output_path = output_dir / f"{name}{name_suffix}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Paparoditis (2010) Q_n(u) for AAPL and TXN."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent.parent / "standardization_results" / "nn_standardized_macro.csv",
        help="CSV containing Date, AAPL, and TXN columns of the NN-standardized "
             "residuals z_t (conditionally mean-zero/variance-one by construction).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("rolling_periodogram_results"),
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Local smoothing bandwidth. Default: (n/N)^(1/5) h.",
    )
    parser.add_argument(
        "--integration-points", type=int, default=DEFAULT_INTEGRATION_POINTS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_paths = plot_tickers(
        csv_path=args.csv,
        output_dir=args.output_dir,
        window_length=args.window,
        h=args.h,
        b=args.b,
        integration_points=args.integration_points,
    )
    for path in output_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
