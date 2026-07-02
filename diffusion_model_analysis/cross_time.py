import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel, ConditionalGenerator
from models import HFunctionDirectTrainer, EllTrainer, HFunctionTwoStepTrainer
from scipy.stats import gaussian_kde



config = get_default_config()

data_processor = DataProcessor(
    csv_path=config.data.ct_csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=1,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)

data_processor.process_all()

tickers      = config.data.tickers
n_assets     = len(tickers) - 1
plot_tickers = tickers[1:]
n_plot       = len(plot_tickers)

config.diffusion.in_channels  = n_assets
config.diffusion.out_channels = n_assets
config.hfunction.asset_dim    = n_assets
def get_mask(X):
    last_window = X[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        return last_window.sum(dim=1) <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        return (last_window[:, -1] - last_window[:, 0]).abs() >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        return last_window[:, -1].abs() >= config.hfunction.event_threshold


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

N_uncond = 2000
batch_size = 128

print(f"Generating {N_uncond} unconditional samples...")

chunks = []

for start in range(0, N_uncond, batch_size):
    bs = min(batch_size, N_uncond - start)
    chunks.append(diffusion_model.sample(
        batch_size = bs,
        num_steps=config.diffusion.num_steps,
        stoch = 0,
    ).cpu())

uncond = torch.cat(chunks, dim = 0)

X_old = data_processor.X_train
mask_old = get_mask(X_old)

N_cond = mask_old.sum().item()



if config.hfunction.one_two_step == "one":
    h_trainer = HFunctionDirectTrainer(cfg=config.hfunction, b_min=config.diffusion.b_min, b_max=config.diffusion.b_max)
    h_trainer.load("ckpt_new/hfunction.pt")
else:
    ell_trainer = EllTrainer(cfg=config.hfunction)
    ell_trainer.load("ckpt_new/ell_function.pt")
    h_trainer = HFunctionTwoStepTrainer(cfg=config.hfunction, diffusion_model=diffusion_model, ell_model=ell_trainer.model)
    h_trainer.load("ckpt_new/hfunction.pt")

cond_generator = ConditionalGenerator(
    score_model=diffusion_model.model,
    h_model=h_trainer.model,
    diffusion_coeff_fn=diffusion_model.diffusion_coeff_fn,
    drift_coeff_fn=diffusion_model.drift_coeff_fn,
    make_vp_std_grid_fn=DiffusionModel.make_vp_std_grid,
    b_min=config.diffusion.b_min,
    b_max=config.diffusion.b_max,
    device=config.conditional.device,
)

print(f"Generating {N_cond} conditional samples...")
cond = cond_generator.generate(
    num_samples=N_cond,
    batch_size=config.conditional.batch_size,
    num_steps=config.conditional.num_steps,
    stoch=config.conditional.stoch,
    eta=config.conditional.eta,
    use_q_model=False,
)
panels = [
    ("Real Old Period (all)",          X_old[:, -1, 1:].numpy()),
    ("Real Old Period (event windows)", X_old[mask_old, -1, 1:].numpy()),
    ("Unconditional Generated",         uncond[:, 1:, -1].numpy()),
    ("Conditional Generated",           cond[:, 1:, -1].numpy()),
]


def plot_matrices(panels, title, fname):
    tick_lbl = [t.upper() for t in plot_tickers]
    font_size = max(8, min(13, 40 // n_plot))

    fig, axes = plt.subplots(2, 2, figsize = (11, 9))
    for ax, (lbl, arr) in zip(axes.ravel(), panels):
        C = np.corrcoef(arr.T)
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_plot))
        ax.set_xticklabels(tick_lbl, fontsize=10)
        ax.set_yticks(range(n_plot))
        ax.set_yticklabels(tick_lbl, fontsize=10)
        ax.set_title(f"{lbl}\n(n={len(arr)})", fontsize=10, fontweight="bold", pad=8)

        for r in range(n_plot):
            for c in range(n_plot):
                v = C[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=font_size,
                        fontweight="bold", color="white" if abs(v) > 0.6 else "black")
                
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs("diffusion_model_analysis/results", exist_ok=True)
    plt.savefig(f"diffusion_model_analysis/results/{fname}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")



def plot_marginals(panels, title, fname):
    real_all = panels[0][1]
    real_event = panels[1][1]
    uncond_arr = panels[2][1]
    cond_arr = panels[3][1]
    
    fig, axes = plt.subplots(n_plot, 2, figsize= (12, 3 * n_plot))

    for i in range(n_plot):
        ax_l = axes[i,0]
        ax_r = axes[i, 1]

        for vals, color, lbl in [
            (real_all[:, i],   "darkorange", f"Real (n={len(real_all)})"),
            (uncond_arr[:, i], "steelblue",  f"Unconditional (n={len(uncond_arr)})"),
        ]:
            x = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 500)
            kde = gaussian_kde(vals)
            ax_l.plot(x, kde(x), color=color, label=lbl)
            ax_l.fill_between(x, kde(x), alpha=0.3, color=color)

        for vals, color, lbl in [
            (real_event[:, i],   "darkorange", f"Real (n={len(real_event)})"),
            (cond_arr[:, i], "steelblue",  f"Conditional (n={len(cond_arr)})"),
        ]:
            x = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 500)
            kde = gaussian_kde(vals)
            ax_r.plot(x, kde(x), color=color, label=lbl)
            ax_r.fill_between(x, kde(x), alpha=0.3, color=color)

        ax_l.set_title(f"{plot_tickers[i].upper()} — Unconditional", fontsize=10, fontweight="bold")
        ax_r.set_title(f"{plot_tickers[i].upper()} — Conditional", fontsize=10, fontweight="bold")
        ax_l.set_xlabel("Standardized Return (day 64)")
        ax_r.set_xlabel("Standardized Return (day 64)")
        ax_l.set_ylabel("Density")
        ax_l.legend(fontsize=7)
        ax_r.legend(fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs("diffusion_model_analysis/results", exist_ok=True)
    plt.savefig(f"diffusion_model_analysis/results/{fname}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")



plot_matrices(panels, "Correlation Matrices — Cross-Time Generalization", "corr_cross_time.png")
plot_marginals(panels, "Marginal Distributions — Cross-Time Generalization", "marginals_cross_time.png")