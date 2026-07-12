import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
from statsmodels.tsa.stattools import acf, ccf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from scipy.stats import ttest_ind

import matplotlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor


#ACF of returns
config = get_default_config()
data_processor = DataProcessor(
    csv_path=config.data.csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=config.data.test_days,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)
data_processor.process_all()

tickers = config.data.tickers          # all assets, e.g. ["unemp", "sp500", "baa"]
n_assets = len(tickers) - 1 

# X shape: (N, T, A)   gen shape: (N, A, T)
X_train = data_processor.X_train
X_test  = data_processor.X_test

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)

gen_train = torch.load(os.path.join(_root, 'generated_samples_train.pt'), map_location='cpu')
gen_test  = torch.load(os.path.join(_root, 'generated_samples_test.pt'),  map_location='cpu')

def get_mask(X):
    last_window = X[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        return last_window.sum(dim=1) <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        return (last_window[:, -1] - last_window[:, 0]).abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        return last_window[:, -1].abs() >= config.hfunction.event_threshold


mask_train = get_mask(X_train)
mask_test  = get_mask(X_test)


splits = [
    (X_train, mask_train, gen_train, "Train"),
    (X_test,  mask_test,  gen_test,  "Test"),
]

#ACF

for X, mask, gen, split_label in splits:
    fig, axes = plt.subplots(n_assets, 1, figsize=(8, 4 * n_assets))
    if n_assets == 1:
        axes = [axes]

    for ch, ticker in enumerate(tickers[1:]):
        acf_list_real = []
        acf_list_gen  = []

        for i in range(len(X)):
            acf_list_real.append(acf(X[i, :, ch], nlags=20))

        for i in range(len(gen)):
            acf_list_gen.append(acf(gen[i, ch, :], nlags=20))

        mean_acf_real = np.mean(acf_list_real, axis=0)
        mean_acf_gen  = np.mean(acf_list_gen,  axis=0)

        lags = np.arange(0, 21)
        ax = axes[ch]
        ax.plot(lags, mean_acf_real, color="darkorange", linewidth=1.5, label="Real")
        ax.plot(lags, mean_acf_gen,  color="steelblue",  linewidth=1.5, label="Generated")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{ticker.upper()} — ACF of Returns ({split_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"ACF of Returns — Real vs Generated ({split_label})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
    out = os.path.join(_dir, "results", f"acf_returns_{split_label.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


method = "simple"
def get_residuals(series, method):
    if method == "simple":
        return series ** 2
    if method == "ar":
        model = auto_arima(series, seasonal = True, information_criterion = "aic", suppress_warnings = True)
        return model.resid() ** 2

fig, axes = plt.subplots(n_assets, 2, figsize = (14, 4 * n_assets))
if n_assets == 1:
    axes = axes[np.newaxis, :]

for col, (X, mask, gen, split_label) in enumerate(splits):
    for ch, ticker in enumerate(tickers[1:]):
        # Real: reconstruct full time series from last-day returns across windows
        full_series_real = X[:, -1, ch].numpy()
        mean_acf_squared_real = acf(full_series_real ** 2, nlags=20)

        # Generated: windows are independent so per-window average is the only option
        acf_sq_gen = []
        for j in range(len(gen)):
            series_g = gen[j, ch, :]
            residuals_g = get_residuals(series_g, method)
            acf_sq_gen.append(acf(residuals_g, nlags=20))
        mean_acf_squared_gen = np.mean(acf_sq_gen, axis=0)

        lags = np.arange(0, 21)
        ax = axes[ch, col]
        ax.plot(lags, mean_acf_squared_real, color="darkorange", linewidth=1.5, label="Real")
        ax.plot(lags, mean_acf_squared_gen,  color="steelblue",  linewidth=1.5, label="Generated")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{ticker.upper()} — ACF Squared Residuals ({split_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

fig.suptitle(f"ACF of Squared Residuals [{method}] — Real vs Generated", fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(_dir, "results", f"acf_squared_{method}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")


seq_len = config.data.seq_len
fig_2, axes_2 = plt.subplots(n_assets, 2, figsize=(14, 4 * n_assets))
if n_assets == 1:
    axes_2 = axes_2[np.newaxis, :]

for col, (X, mask, gen, split_label) in enumerate(splits):
    for ch, ticker in enumerate(tickers[1:]):
        sq_acf_rl = []
        non_overlap = np.arange(0, X.shape[0], seq_len)
        for i in non_overlap:
            residuals_rr = get_residuals(X[i, :, ch].numpy(), method)
            sq_acf_rl.append(acf(residuals_rr))
        acf_sq_real_again = np.array(sq_acf_rl)

        sq_acf_gen = []
        for j in range(len(gen)):
            residuals_gg = get_residuals(gen[j, ch, :], method)
            sq_acf_gen.append(acf(residuals_gg))

        acf_sq_gen_again = np.array(sq_acf_gen)

        t_stat, p_vals = ttest_ind(acf_sq_real_again, acf_sq_gen_again, equal_var = False, axis = 0)

        lags = np.arange(0, 21, 1)
        ax = axes_2[ch, col]
        ax.plot(lags, p_vals, color="crimson", marker="o", markersize=3, linewidth=1.2, label="p-value")
        ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--", label="p = 0.05")
        ax.set_title(f"{ticker.upper()} — Real vs Gen p-value ({split_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag")
        ax.set_ylabel("p-value")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

fig_2.suptitle("Two-Sample t-test: Real vs Generated Squared-Residual ACF (p-values)", fontsize=13, fontweight="bold")
fig_2.tight_layout()
out = os.path.join(_dir, "results", "acf_squared_pvalues.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")