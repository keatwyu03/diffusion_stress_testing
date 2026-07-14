import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
from statsmodels.tsa.stattools import acf, ccf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from scipy.stats import ttest_ind
from scipy.stats import norm

import matplotlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor

n_lags = 10   # max lag used everywhere below (acf/Ljung-Box nlags, plotting x-axis)
method = "simple"
def get_residuals(series, method):
    if method == "simple":
        return series ** 2
    if method == "ar":
        model = auto_arima(series, seasonal = True, information_criterion = "aic", suppress_warnings = True)
        return model.resid() ** 2


#ACF of returns
config = get_default_config()
data_processor = DataProcessor(
    csv_path=config.data.csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=config.data.test_days,
    window_shift=config.data.window_shift,
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
os.makedirs(os.path.join(_dir, "results"), exist_ok=True)

gen_train = torch.load(os.path.join(_root, 'generated_samples_train.pt'), map_location='cpu')
gen_test  = torch.load(os.path.join(_root, 'generated_samples_test.pt'),  map_location='cpu')

def get_mask(X):
    last_window = X[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        return last_window.sum(dim=1) <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "abs_change":
        return (last_window[:, -1] - last_window[:, 0]).abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        return last_window[:, -1].abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "upper_change":
        return (last_window[:, -1] - last_window[:, 0]) >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "lower_change":
        return (last_window[:, -1] - last_window[:, 0]) <= -config.hfunction.event_threshold


mask_train = get_mask(X_train)
mask_test  = get_mask(X_test)


splits = [
    (X_train, mask_train, gen_train, "Train"),
    (X_test,  mask_test,  gen_test,  "Test"),
]

seq_len = config.data.seq_len


#ACF of squared residuals
fig_real, axes_real = plt.subplots(n_assets, 2, figsize = (14, 4 * n_assets))
if n_assets == 1:
    axes_real = axes_real[np.newaxis, :]

for col, (X, mask, gen, split_label) in enumerate(splits):
    for ch, ticker in enumerate(tickers[1:]):
        non_overlap_r = np.arange(0, X.shape[0], seq_len)
        acf_seq_real = []
        for i in non_overlap_r:
            residuals_r = get_residuals(X[i, :, ch].numpy(), method)
            acf_seq_real.append(acf(residuals_r, nlags = n_lags))
        
        acf_seq_real = np.array(acf_seq_real)
        mean_acf_real = acf_seq_real.mean(axis = 0)
        se_acf_real = acf_seq_real.std(axis = 0, ddof = 1) / np.sqrt(len(acf_seq_real)) 
        ci_upper_real = mean_acf_real + norm.ppf(0.975) * se_acf_real
        ci_lower_real = mean_acf_real - norm.ppf(0.975) * se_acf_real
    
        lags = np.arange(0, n_lags + 1)
        ax = axes_real[ch, col]
        ax.plot(lags[1:], mean_acf_real[1:], color="darkorange", linewidth=1.5, label="Real mean ACF")
        ax.plot(lags[1:], ci_upper_real[1:], color="black", linestyle=":", linewidth=1, label="95% band")
        ax.plot(lags[1:], ci_lower_real[1:], color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(-0.05, 0.2)
        ax.set_title(f"{ticker.upper()} — Real ({split_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


fig_real.suptitle("Real: ACF of Squared Residuals + 95% Significance Band", fontsize=13, fontweight="bold")
fig_real.tight_layout()
plt.savefig(os.path.join(_dir, "results", "acf_squared_real_band.png"), dpi=150, bbox_inches="tight")
plt.show()




fig_gen, axes_gen = plt.subplots(n_assets, 2, figsize = (14, 4 * n_assets))
if n_assets == 1:
    axes_gen = axes_gen[np.newaxis, :]

for col, (X, mask, gen, split_label) in enumerate(splits):
    for ch, ticker in enumerate(tickers[1:]):
        acf_seq_gen = []
        for j in range(len(gen)):
            residuals_g = get_residuals(gen[j, ch, :], method)
            acf_seq_gen.append(acf(residuals_g, nlags=n_lags))
        acf_seq_gen = np.array(acf_seq_gen)

        mean_acf_gen = acf_seq_gen.mean(axis=0)
        se_acf_gen = acf_seq_gen.std(axis=0, ddof=1) / np.sqrt(len(acf_seq_gen))
        ci_upper_gen = mean_acf_gen + norm.ppf(0.975) * se_acf_gen
        ci_lower_gen = mean_acf_gen - norm.ppf(0.975) * se_acf_gen

        lags = np.arange(0, n_lags + 1)
        ax = axes_gen[ch, col]
        ax.plot(lags[1:], mean_acf_gen[1:], color="steelblue", linewidth=1.5, label="Generated mean ACF")
        ax.plot(lags[1:], ci_upper_gen[1:], color="black", linestyle=":", linewidth=1, label="95% band")
        ax.plot(lags[1:], ci_lower_gen[1:], color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(-0.05, 0.2)
        ax.set_title(f"{ticker.upper()} — Generated ({split_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


fig_gen.suptitle("Generated: Mean ACF of Squared Residuals + 95% Significance Band", fontsize=13, fontweight="bold")
fig_gen.tight_layout()
plt.savefig(os.path.join(_dir, "results", "acf_squared_gen_band.png"), dpi=150, bbox_inches="tight")
plt.show()




fig_2, axes_2 = plt.subplots(n_assets, 2, figsize=(14, 4 * n_assets))
if n_assets == 1:
    axes_2 = axes_2[np.newaxis, :]

for col, (X, mask, gen, split_label) in enumerate(splits):
    for ch, ticker in enumerate(tickers[1:]):
        sq_acf_rl = []
        non_overlap = np.arange(0, X.shape[0], seq_len)
        for i in non_overlap:
            residuals_rr = get_residuals(X[i, :, ch].numpy(), method)
            sq_acf_rl.append(acf(residuals_rr, nlags=n_lags))
        acf_sq_real_again = np.array(sq_acf_rl)

        sq_acf_gen = []
        for j in range(len(gen)):
            residuals_gg = get_residuals(gen[j, ch, :], method)
            sq_acf_gen.append(acf(residuals_gg, nlags=n_lags))

        acf_sq_gen_again = np.array(sq_acf_gen)

        t_stat, p_vals = ttest_ind(acf_sq_real_again, acf_sq_gen_again, equal_var = False, axis = 0)

        lags = np.arange(0, n_lags + 1, 1)
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


