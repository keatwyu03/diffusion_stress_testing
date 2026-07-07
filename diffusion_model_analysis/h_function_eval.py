"""
H-function calibration check — bypasses the sampling/guidance pipeline entirely.

Takes real (X, B_label) pairs, forward-noises them at several fixed tau values,
and reports h_model(y_tau, tau) split by true label (event vs no-event).

If h has learned anything real, the positive- and negative-label groups should
separate cleanly at low tau (close to real data) and converge toward the same
value (~ the base event rate) at high tau (near-pure noise). If they don't
separate even at low tau, h's own fit is broken, independent of the guidance
formula, eta, or data-scarcity questions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor
from models import HFunctionDirectTrainer

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

n_assets = len(config.data.tickers) - 1
config.diffusion.in_channels  = n_assets
config.diffusion.out_channels = n_assets
config.hfunction.asset_dim    = n_assets

device = config.hfunction.device
tau_values = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]

# ── Load the trained h-function ───────────────────────────────────────────────
h_trainer = HFunctionDirectTrainer(
    cfg=config.hfunction,
    b_min=config.diffusion.b_min,
    b_max=config.diffusion.b_max,
)
h_trainer.load("ckpt_new/hfunction.pt")
h_trainer.model.eval()


@torch.no_grad()
def h_by_tau(X: torch.Tensor, labels: torch.Tensor):
    """X: (N, A, T) channels-first. labels: (N,) in {0,1}. Returns dict tau -> (pos_mean, neg_mean, pos_std, neg_std)."""
    X = X.to(device)
    labels = labels.to(device)
    pos_mask = labels == 1
    neg_mask = labels == 0

    out = {}
    for tau_val in tau_values:
        tau = torch.full((X.shape[0],), tau_val, device=device)
        y_tau = h_trainer._forward_noise(X, tau)
        probs = h_trainer.model(y_tau, tau).squeeze(-1)

        out[tau_val] = (
            probs[pos_mask].mean().item() if pos_mask.any() else float("nan"),
            probs[neg_mask].mean().item() if neg_mask.any() else float("nan"),
            probs[pos_mask].std().item()  if pos_mask.sum() > 1 else float("nan"),
            probs[neg_mask].std().item()  if neg_mask.sum() > 1 else float("nan"),
        )
    return out


# ── Train split: real windows the h-function was actually trained on ─────────
X_train_direct = data_processor.get_diffusion_data()
Z_start, Z_end, valid_idx = data_processor.get_z_windows()
X_train_direct = X_train_direct[valid_idx]
B_train = h_trainer._compute_labels(Z_start, Z_end)

print(f"Train: N={len(B_train)}  pos={int(B_train.sum().item())}  pos_ratio={B_train.mean().item():.4f}")
train_results = h_by_tau(X_train_direct, B_train)

# ── Test split: same event definition, applied to held-out windows ───────────
X_test = data_processor.X_test  # (N, T, A) channels-last
X_test_direct = X_test.permute(0, 2, 1)  # -> (N, A, T)

last_window_test = X_test[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
if config.hfunction.event_type == "sum":
    B_test = (last_window_test.sum(dim=1) <= config.hfunction.event_threshold).float()
elif config.hfunction.event_type == "change":
    B_test = ((last_window_test[:, -1] - last_window_test[:, 0]).abs() >= config.hfunction.event_threshold).float()
elif config.hfunction.event_type == "absval":
    B_test = (last_window_test[:, -1].abs() >= config.hfunction.event_threshold).float()

print(f"Test:  N={len(B_test)}  pos={int(B_test.sum().item())}  pos_ratio={B_test.mean().item():.4f}")
test_results = h_by_tau(X_test_direct, B_test)


def print_table(name, results):
    print(f"\n{name}")
    print(f"{'tau':>6} | {'pos mean':>10} {'pos std':>9} | {'neg mean':>10} {'neg std':>9} | {'separation':>10}")
    print("-" * 66)
    for tau_val in tau_values:
        pos_m, neg_m, pos_s, neg_s = results[tau_val]
        print(f"{tau_val:>6.2f} | {pos_m:>10.4f} {pos_s:>9.4f} | {neg_m:>10.4f} {neg_s:>9.4f} | {pos_m - neg_m:>10.4f}")


print_table("TRAIN (real windows h was trained on)", train_results)
print_table("TEST (held-out windows)", test_results)


# ── Plot: mean h(tau) by group, train vs test ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, results, title in [(axes[0], train_results, "Train"), (axes[1], test_results, "Test")]:
    pos_means = [results[t][0] for t in tau_values]
    neg_means = [results[t][1] for t in tau_values]
    pos_stds  = [results[t][2] for t in tau_values]
    neg_stds  = [results[t][3] for t in tau_values]

    ax.errorbar(tau_values, pos_means, yerr=pos_stds, marker="o", color="crimson",
                label="True event (B=1)", capsize=3)
    ax.errorbar(tau_values, neg_means, yerr=neg_stds, marker="o", color="steelblue",
                label="True no-event (B=0)", capsize=3)
    ax.set_xlabel("tau (0 = real data, 1 = pure noise)")
    ax.set_ylabel("h_model(y_tau, tau)")
    ax.set_title(f"{title} — h calibration by true label")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("H-Function Calibration Check (bypasses sampling/guidance entirely)", fontsize=13, fontweight="bold")
fig.tight_layout()

_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
out = os.path.join(_dir, "results", "h_function_eval.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved {out}")
