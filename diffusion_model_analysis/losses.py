import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--score_loss", type=str, default=None,
                    help="Path to score-function loss CSV. Auto-resolves to most recent if omitted.")
parser.add_argument("--h_loss", type=str, default=None,
                    help="Path to H-function loss CSV. Auto-resolves to most recent if omitted.")
args = parser.parse_args()


def _latest_csv(pattern):
    """Return the most recently modified file matching pattern, or None."""
    hits = glob.glob(os.path.join(ROOT, "**", pattern), recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


score_path = args.score_loss or _latest_csv("score_losses.csv") or _latest_csv("train_losses.csv")
h_path     = args.h_loss    or _latest_csv("h_losses.csv")      or _latest_csv("h_function_losses.csv")

if score_path is None:
    raise FileNotFoundError(
        "No score-function loss CSV found. Pass --score_loss <path> or train the model first."
    )

print(f"Score loss : {score_path}")
print(f"H-function : {h_path or 'not found'}")

score_df = pd.read_csv(score_path)
has_h    = h_path is not None and os.path.exists(h_path)
if has_h:
    h_df = pd.read_csv(h_path)

# ── Layout: 1 col (score only) or 3 cols (score + h-loss + h-accuracy) ────────
ncols = 3 if has_h else 1
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

    # H-function accuracy and pos_ratio
    axes[2].plot(h_df["epoch"], h_df["accuracy"],  linewidth=1.5, color="seagreen",    label="Accuracy")
    axes[2].plot(h_df["epoch"], h_df["pos_ratio"], linewidth=1.5, color="mediumpurple",
                 linestyle="--", label="Pos Rate")
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Value", fontsize=12)
    axes[2].set_title("H-Function Accuracy & Pos Rate", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

fig.tight_layout()

_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
out_path = os.path.join(_dir, "results", "train_losses.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {out_path}")
