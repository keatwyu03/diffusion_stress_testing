import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor

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

tickers  = config.data.tickers
SP500_CH = tickers.index("sp500")
BAA_CH   = tickers.index("baa")

X_train = data_processor.X_train  # (N, 64, 5)
X_test  = data_processor.X_test

def get_mask(X, config):
    last_window = X[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        return last_window.sum(dim=1) <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        return (last_window[:, -1] - last_window[:, 0]).abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        return last_window[:, -1].abs() >= config.hfunction.event_threshold

mask_train = get_mask(X_train, config)
mask_test  = get_mask(X_test,  config)

# Load conditional generated samples
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
gen_train = torch.load(os.path.join(_root, 'generated_samples_train.pt'), map_location='cpu')  # (N, 5, 64)
gen_test  = torch.load(os.path.join(_root, 'generated_samples_test.pt'),  map_location='cpu')

# Real: only event windows, all 64 time steps flattened
# Generated: all samples, all 64 time steps flattened
real = {
    "SP500": {
        "train": X_train[mask_train, :, SP500_CH].flatten().numpy(),
        "test":  X_test[ mask_test,  :, SP500_CH].flatten().numpy(),
    },
    "BAA": {
        "train": X_train[mask_train, :, BAA_CH].flatten().numpy(),
        "test":  X_test[ mask_test,  :, BAA_CH].flatten().numpy(),
    },
}

gen = {
    "SP500": {
        "train": gen_train[:, SP500_CH, :].flatten().numpy(),
        "test":  gen_test[ :, SP500_CH, :].flatten().numpy(),
    },
    "BAA": {
        "train": gen_train[:, BAA_CH, :].flatten().numpy(),
        "test":  gen_test[ :, BAA_CH, :].flatten().numpy(),
    },
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, asset in enumerate(["SP500", "BAA"]):
    for col, split in enumerate(["train", "test"]):
        ax = axes[row, col]
        real_vals = real[asset][split]
        gen_vals  = gen[asset][split]

        for vals, color, label in [
            (real_vals, "darkorange", f"Real event windows {split} (n={len(real_vals)})"),
            (gen_vals,  "steelblue",  f"Conditional generated {split} (n={len(gen_vals)})"),
        ]:
            kde = gaussian_kde(vals, bw_method='silverman')
            x   = np.linspace(min(real_vals.min(), gen_vals.min()) - 0.5,
                               max(real_vals.max(), gen_vals.max()) + 0.5, 500)
            ax.plot(x, kde(x), color=color, linewidth=2, label=label)
            ax.hist(vals, bins=40, density=True, alpha=0.2, color=color)

        split_label = "In-Sample (Train)" if split == "train" else "Out-of-Sample (Test)"
        ax.set_title(f"{asset} — {split_label}", fontsize=11)
        ax.set_xlabel("Standardized Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

fig.suptitle("Conditional Generated vs. Real Event Windows: Marginal Distribution", fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(_dir, "conditional_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()