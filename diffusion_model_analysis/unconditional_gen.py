import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import combinations
from matplotlib.patches import Patch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel

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

tickers  = config.data.tickers   # all assets
n_assets = len(tickers)

# Use the actual diffusion training/test data (same preprocessing as model training)
# Shape: (N, A, T) — matches uncond format directly
X_real_train = data_processor.get_diffusion_data()   # (N_train, A, T) from df_z_wins
X_real_test  = data_processor.X_test.permute(0, 2, 1)  # (N_test, A, T)

config.diffusion.in_channels  = n_assets
config.diffusion.out_channels = n_assets

# ── Unconditional generation ──────────────────────────────────────────────────
diffusion_model = DiffusionModel(
    in_channels=config.diffusion.in_channels,
    out_channels=config.diffusion.out_channels,
    sample_size=config.diffusion.sample_size,
    layers_per_block=config.diffusion.layers_per_block,
    block_out_channels=config.diffusion.block_out_channels,
    b_min=config.diffusion.b_min,
    b_max=config.diffusion.b_max,
    device=config.diffusion.device,
    arch=config.diffusion.arch,
    embed_dim=config.diffusion.embed_dim,
    n_heads=config.diffusion.n_heads,
    n_layers=config.diffusion.n_layers,
    cond_dim=config.diffusion.cond_dim,
)
diffusion_model.load("ckpt_new/diffusion_model.pt")

N_samples  = 2000
batch_size = 128
print(f"Generating {N_samples} unconditional samples (batch={batch_size})...")
chunks = []
for start in range(0, N_samples, batch_size):
    bs = min(batch_size, N_samples - start)
    chunks.append(diffusion_model.sample(
        batch_size=bs,
        num_steps=config.diffusion.num_steps,
        stoch=0,
    ).cpu())
uncond = torch.cat(chunks, dim=0)  # (N_samples, A, T)

_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_dir, "results"), exist_ok=True)


def diagnose_score_target(dm, x, t_values=(1.0, 0.6, 0.35, 0.15, 0.01)):
    dm.model.eval()
    x = x.to(dm.device)
    with torch.no_grad():
        for t_val in t_values:
            b = x.shape[0]
            t = torch.ones(b, device=dm.device) * t_val
            z = torch.randn_like(x)
            std  = dm.marginal_prob_std_fn(t)[:, None, None]
            mean = dm.marginal_prob_mean_fn(t)[:, None, None]
            perturbed_x = mean * x + std * z
            pred_score   = dm.model(perturbed_x, t).sample
            target_score = -z / std
            corr = torch.corrcoef(torch.stack([pred_score.flatten(), target_score.flatten()]))[0, 1].item()
            mse  = torch.mean((pred_score - target_score) ** 2).item()
            print(f"\n[SCORE DIAG t={t_val}]  corr={corr:.4f}  MSE={mse:.4f}"
                  f"  pred|mean|={pred_score.abs().mean():.4f}"
                  f"  target|mean|={target_score.abs().mean():.4f}")


print("Real train mean per asset:", X_real_train.mean(dim=(0,2)).tolist())
print("Generated uncond mean per asset:", uncond.mean(dim=(0,2)).tolist())

# ── Diagnostics table ─────────────────────────────────────────────────────────
rows = []
for i, ticker in enumerate(tickers):
    real_last = X_real_train[:, i, -1].numpy()
    gen_last  = uncond[:, i, -1].numpy()
    real_cum  = X_real_train[:, i, :].sum(dim=1).numpy()
    gen_cum   = uncond[:, i, :].sum(dim=1).numpy()

    for split, vals_last, vals_cum in [
        ("real train", real_last, real_cum),
        ("generated",  gen_last,  gen_cum),
    ]:
        qs_l = np.quantile(vals_last, [.01, .05, .5, .95, .99]).round(3)
        qs_c = np.quantile(vals_cum,  [.01, .05, .5, .95, .99]).round(3)
        rows.append([ticker.upper() if split == "real train" else "",
                     split,
                     f"{vals_last.mean():.3f}", f"{vals_last.std():.3f}", str(qs_l),
                     f"{vals_cum.mean():.3f}",  f"{vals_cum.std():.3f}",  str(qs_c)])

col_labels = ["Asset", "Split",
              "Mean (last)", "Std (last)", "q[1,5,50,95,99] last day",
              "Mean (cum)",  "Std (cum)",  "q[1,5,50,95,99] 64-day sum"]
fig_d, ax_d = plt.subplots(figsize=(22, 0.45 * len(rows) + 1.5))
ax_d.axis("off")
tbl = ax_d.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(col_labels))))
fig_d.suptitle("Unconditional Generation — Diagnostics", fontsize=12, fontweight="bold")
fig_d.tight_layout()
out_diag = os.path.join(_dir, "results", "unconditional_diagnostics.png")
plt.savefig(out_diag, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_diag}")


def kde_plot(ax, real_vals, gen_vals, real_label, gen_label, xlabel):
    x_min = min(real_vals.min(), gen_vals.min()) - 0.5
    x_max = max(real_vals.max(), gen_vals.max()) + 0.5
    x = np.linspace(x_min, x_max, 500)
    for vals, color, label in [
        (real_vals, "darkorange", real_label),
        (gen_vals,  "steelblue",  gen_label),
    ]:
        kde = gaussian_kde(vals, bw_method="silverman")
        ax.plot(x, kde(x), color=color, linewidth=2, label=label)
        ax.hist(vals, bins=40, density=True, alpha=0.2, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def make_figure(extract_real_fn, extract_gen_fn, suptitle, filename, xlabel):
    """rows=assets, col0=train, col1=test"""
    fig, axes = plt.subplots(n_assets, 2, figsize=(14, 4 * n_assets))
    if n_assets == 1:
        axes = axes[np.newaxis, :]

    splits = [
        (0, X_real_train, "In-Sample (Train)"),
        (1, X_real_test,  "Out-of-Sample (Test)"),
    ]

    for row, ticker in enumerate(tickers):
        for col, X, split_label in splits:
            ax = axes[row, col]
            real_vals = extract_real_fn(X, row)
            gen_vals  = extract_gen_fn(row)

            kde_plot(
                ax, real_vals, gen_vals,
                real_label=f"Real (n={len(real_vals)})",
                gen_label =f"Unconditional generated (n={len(gen_vals)})",
                xlabel=xlabel,
            )
            ax.set_title(f"{ticker.upper()} — {split_label}", fontsize=11, fontweight="bold")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_dir, "results", filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# ── Figure 1: Last-day returns ────────────────────────────────────────────────
# X is (N, A, T): asset dim=1, time dim=2
make_figure(
    extract_real_fn=lambda X, ch: X[:, ch, -1].numpy(),
    extract_gen_fn =lambda ch:    uncond[:, ch, -1].numpy(),
    suptitle="Unconditional Generation — Last-Day Return",
    filename="unconditional_lastday.png",
    xlabel="Standardized Return (day 64)",
)

# ── Figure 2: Cumulative returns (64-day sum) ─────────────────────────────────
make_figure(
    extract_real_fn=lambda X, ch: X[:, ch, :].sum(dim=1).numpy(),
    extract_gen_fn =lambda ch:    uncond[:, ch, :].sum(dim=1).numpy(),
    suptitle="Unconditional Generation — Cumulative Return (64-day sum)",
    filename="unconditional_cumulative.png",
    xlabel="Cumulative Standardized Return",
)

# ── Joint distributions (all pairwise combinations) ───────────────────────────
pairs   = list(combinations(range(n_assets), 2))
n_pairs = len(pairs)

fig2, axes2 = plt.subplots(n_pairs, 2, figsize=(14, 6 * n_pairs))
if n_pairs == 1:
    axes2 = axes2[np.newaxis, :]

for row, (i, j) in enumerate(pairs):
    t1, t2 = tickers[i], tickers[j]
    for col, (X, split_label) in enumerate([(X_real_train, "In-Sample (Train)"), (X_real_test, "Out-of-Sample (Test)")]):
        ax = axes2[row, col]

        real_t1 = X[:, i, -1].numpy()
        real_t2 = X[:, j, -1].numpy()
        gen_t1  = uncond[:, i, -1].numpy()
        gen_t2  = uncond[:, j, -1].numpy()

        x_min = min(real_t1.min(), gen_t1.min()) - 0.5
        x_max = max(real_t1.max(), gen_t1.max()) + 0.5
        y_min = min(real_t2.min(), gen_t2.min()) - 0.5
        y_max = max(real_t2.max(), gen_t2.max()) + 0.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80),
                             np.linspace(y_min, y_max, 80))
        grid = np.vstack([xx.ravel(), yy.ravel()])

        zz_real = gaussian_kde(np.vstack([real_t1, real_t2]), bw_method="silverman")(grid).reshape(xx.shape)
        zz_gen  = gaussian_kde(np.vstack([gen_t1,  gen_t2]),  bw_method="silverman")(grid).reshape(xx.shape)

        ax.contourf(xx, yy, zz_real, levels=10, cmap="Oranges", alpha=0.5)
        ax.contour( xx, yy, zz_real, levels=10, colors="darkorange", linewidths=0.8, alpha=0.8)
        ax.contourf(xx, yy, zz_gen,  levels=10, cmap="Blues",   alpha=0.5)
        ax.contour( xx, yy, zz_gen,  levels=10, colors="steelblue",  linewidths=0.8, alpha=0.8)

        ax.legend(handles=[
            Patch(color="darkorange", alpha=0.7, label=f"Real last-day (n={len(real_t1)})"),
            Patch(color="steelblue",  alpha=0.7, label=f"Generated (n={len(gen_t1)})"),
        ], fontsize=9, loc="upper right")

        ax.set_title(f"Joint {t1.upper()} × {t2.upper()} — {split_label}", fontsize=11)
        ax.set_xlabel(f"{t1.upper()} Std Return (day 64)")
        ax.set_ylabel(f"{t2.upper()} Std Return (day 64)")
        ax.grid(True, alpha=0.3)

fig2.suptitle("Unconditional Generation — Joint Pairwise Distributions (Last Day)", fontsize=13, fontweight="bold")
fig2.tight_layout()
out2 = os.path.join(_dir, "results", "unconditional_joint.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out2}")
