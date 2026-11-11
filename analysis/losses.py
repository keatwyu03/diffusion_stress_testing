import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--gan", action="store_true",
                    help="plot the cGAN baseline's joint minimax value instead of the "
                         "diffusion model's score/H-function losses.")
parser.add_argument("--score_loss", type=str, default=None,
                    help="Path to score-function loss CSV. Auto-resolves to most recent if omitted.")
parser.add_argument("--h_loss", type=str, default=None,
                    help="Path to H-function loss CSV. Auto-resolves to most recent if omitted.")
args = parser.parse_args()

if args.gan:
    gan_losses_path = os.path.join(ROOT, "gan_baseline", "gan_results", "cgan_losses.json")
    with open(gan_losses_path) as f:
        gan_df = pd.DataFrame(json.load(f))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gan_df["epoch"], gan_df["minimax_value"], linewidth=1.5, color="seagreen")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("V(D,G)", fontsize=12)
    ax.set_title("cGAN Joint Minimax Value  V(D,G) = -d_loss", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    results_dir = os.path.join(_dir, "gan_results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "cgan_losses.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {out_path}")
    sys.exit(0)


from config import get_default_config, get_run_tag
_cfg = get_default_config()
_run_tag = get_run_tag(_cfg)

# ONLY the loss files of the main.py run matching the current config (no fallback to other / untagged files)
score_path = args.score_loss or os.path.join(ROOT, "ckpt_new", f"score_losses_{_run_tag}.csv")
h_path = args.h_loss or os.path.join(ROOT, "ckpt_new", f"h_losses_{_run_tag}.csv")

if not os.path.exists(score_path):
    raise FileNotFoundError(
        f"{score_path} not found -- run main.py for this config (run_tag {_run_tag}) first, or pass --score_loss <path>."
    )
if not os.path.exists(h_path):
    h_path = None

print(f"Score loss : {score_path}")
print(f"H-function : {h_path or 'not found'}")

score_df = pd.read_csv(score_path)
has_h    = h_path is not None and os.path.exists(h_path)
if has_h:
    h_df = pd.read_csv(h_path)

# ── Layout: 1 col (score only), 2 cols (+ h-loss), or 3 cols (+ h-accuracy) ───
# Stale/cross-mode h_losses.csv (e.g. left over from a two-step run, which only
# logs epoch/loss) shouldn't crash the plot — just skip the accuracy panel.
has_h_metrics = has_h and "accuracy" in h_df.columns and "pos_ratio" in h_df.columns
ncols = 3 if has_h_metrics else (2 if has_h else 1)
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
if ncols == 1:
    axes = [axes]

# Score function loss
axes[0].plot(score_df["epoch"], score_df["loss"], linewidth=1.5, color="steelblue")
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Score Function Loss", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3)

if has_h:
    # H-function loss
    axes[1].plot(h_df["epoch"], h_df["loss"], linewidth=1.5, color="darkorange")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("H-Function Loss", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    if not has_h_metrics:
        print("Note: h_losses.csv has no accuracy/pos_ratio columns "
              "(likely a two-step-trainer or stale file) — skipping that panel.")

if has_h_metrics:
    # H-function accuracy and pos_ratio (one-step / legacy trainers only)
    axes[2].plot(h_df["epoch"], h_df["accuracy"],  linewidth=1.5, color="seagreen",    label="Accuracy")
    axes[2].plot(h_df["epoch"], h_df["pos_ratio"], linewidth=1.5, color="mediumpurple",
                 linestyle="--", label="Pos Rate")
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Value", fontsize=12)
    axes[2].set_title("H-Function Accuracy & Pos Rate", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

fig.tight_layout()

import run_paths
results_dir = run_paths.analysis_results_dir(_cfg)
out_path = os.path.join(results_dir, "train_losses.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {out_path}")
