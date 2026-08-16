import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config, get_run_tag
from data import DataProcessor
from models import DiffusionModel
from utils import set_seed

_parser = argparse.ArgumentParser()
_parser.add_argument("--gan", action="store_true",
                      help="cGAN mode. The cGAN baseline has no unconditional generator "
                           "(cgan_tsgm.generate_conditional requires a binary event label) "
                           "so there is nothing to compare against Real here -- this flag "
                           "only exists so evaluation_main.py can forward it uniformly across "
                           "scripts; it makes this script a no-op that exits immediately.")
_args = _parser.parse_args()

if _args.gan:
    print("unconditional_gen.py: --gan given, but the cGAN baseline has no unconditional "
          "generator to evaluate. Nothing to do; exiting.")
    sys.exit(0)

config = get_default_config()
set_seed(config.seed)
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

tickers      = config.data.tickers   # all assets
n_assets     = len(tickers)
plot_tickers = tickers
n_plot       = len(plot_tickers)

X_train = data_processor.X_train  # (N_train, T, A)
X_test  = data_processor.X_test   # (N_test,  T, A)

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
diffusion_model.load(f"ckpt_new/score_{get_run_tag(config)}.pt")

_dir  = os.path.dirname(os.path.abspath(__file__))  # analysis/
_root = os.path.dirname(_dir)                        # repo root
_uncond_path = os.path.join(_root, "unconditional_samples.pt")

# Cached to a shared file so cov.py (and anything else that wants an
# unconditional baseline) reuses the EXACT same batch instead of each script
# paying the sampling cost again and comparing against a different random
# draw. Reproducibility comes from the single global set_seed(config.seed)
# call above (seeds torch's global RNG, which the unseeded torch.randn(...)
# inside diffusion_model.sample() then draws from) -- no separate per-batch
# seed here.
N_samples  = config.conditional.n_gen_samples
batch_size = 128
n_batches = -(-N_samples // batch_size)  # ceil
print(f"Generating {N_samples} unconditional samples (batch={batch_size}, {n_batches} batches)...")
chunks = []
for start in tqdm(range(0, N_samples, batch_size), total=n_batches, desc="Unconditional batches"):
    bs = min(batch_size, N_samples - start)
    chunks.append(diffusion_model.sample(
        batch_size=bs,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
    ).cpu())
uncond = torch.cat(chunks, dim=0)  # (N_samples, A, T)
torch.save(uncond, _uncond_path)
print(f"Saved {_uncond_path}")

_results_dir = os.path.join(_dir, "diffusion_results")
os.makedirs(_results_dir, exist_ok=True)


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


print("Real X_train mean per asset:", X_train.mean(dim=(0,1)).tolist())
print("Generated uncond mean per asset:", uncond.mean(dim=(0,2)).tolist())

# ── Diagnostics table ─────────────────────────────────────────────────────────
rows = []
for i, ticker in zip(range(n_assets), plot_tickers):
    real_last = X_train[:, -1, i].numpy()
    gen_last  = uncond[:, i, -1].numpy()
    real_cum  = X_train[:, :, i].sum(dim=1).numpy()
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
out_diag = os.path.join(_results_dir, "unconditional_diagnostics.png")
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
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 4 * n_plot))
    if n_plot == 1:
        axes = axes[np.newaxis, :]

    splits = [
        (0, X_train, "In-Sample (Train)"),
        (1, X_test,  "Out-of-Sample (Test)"),
    ]

    for row, (ch, ticker) in enumerate(zip(range(n_assets), plot_tickers)):
        for col, X, split_label in splits:
            ax = axes[row, col]
            real_vals = extract_real_fn(X, ch)
            gen_vals  = extract_gen_fn(ch)

            kde_plot(
                ax, real_vals, gen_vals,
                real_label=f"Real (n={len(real_vals)})",
                gen_label =f"Unconditional generated (n={len(gen_vals)})",
                xlabel=xlabel,
            )
            ax.set_title(f"{ticker.upper()} — {split_label}", fontsize=11, fontweight="bold")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_results_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# ── Figure 1: Last-day returns ────────────────────────────────────────────────
make_figure(
    extract_real_fn=lambda X, ch: X[:, -1, ch].numpy(),
    extract_gen_fn =lambda ch:    uncond[:, ch, -1].numpy(),
    suptitle="Unconditional Generation — Last-Day Return",
    filename="unconditional_lastday.png",
    xlabel="Standardized Return (day 64)",
)

# Cumulative returns (64-day sum) -- data still computed in the diagnostics
# table below (real_cum/gen_cum), plot intentionally not produced.

