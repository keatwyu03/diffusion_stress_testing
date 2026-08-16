import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import get_default_config, get_run_tag
from data import DataProcessor
from models import DiffusionModel
from utils import set_seed

_parser = argparse.ArgumentParser()
_parser.add_argument("--gan", action="store_true",
                      help="compare Real vs. the cGAN baseline's generated samples "
                           "(gan_baseline/gan_results/gan_generated_samples_{train,test}.pt), "
                           "saving outputs to analysis/gan_results/. Skips loading the diffusion "
                           "checkpoint and the 'Unconditional Generated' panel entirely -- this "
                           "mode is GAN-only, with no diffusion dependency. Default (no --gan): "
                           "diffusion model's generated_samples_{train,test}.pt plus an "
                           "'Unconditional Generated' diffusion panel, saved to "
                           "analysis/diffusion_results/.")
_args = _parser.parse_args()

config = get_default_config()
set_seed(config.seed)
data_processor = DataProcessor(
    csv_path=config.data.csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=config.data.test_days,
    start_date=config.data.start_date,
    end_date=config.data.end_date,
    train_end_date=config.data.train_end_date,
    window_shift=config.data.window_shift,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
    event_causal=config.data.event_causal,
    event_lag_gap=config.data.event_lag_gap,
)
data_processor.process_all()

tickers      = config.data.tickers
n_assets     = len(tickers)
plot_tickers = tickers
n_plot       = len(plot_tickers)

X_train = data_processor.X_train  # (N_train, T, A)
X_test  = data_processor.X_test   # (N_test,  T, A)

config.diffusion.in_channels  = n_assets
config.diffusion.out_channels = n_assets

# ── Event mask — from the real macro/latent series, same as main.py ───────────
# (X is stock-returns-only and has no macro channel, so the mask must come from
# get_z_windows_*; the config threshold is a "top X%" fraction that must be
# converted to a raw cutoff first)
event_type  = config.hfunction.event_type
h_threshold = data_processor.get_event_threshold_from_percentile(
    config.hfunction.event_threshold, event_type)
print(f"Event threshold: top {config.hfunction.event_threshold:.1%} -> "
      f"{h_threshold:.4f} std ({event_type})")

def event_mask(Z_start, Z_end):
    if event_type == "abs_change":
        return (Z_end - Z_start).abs() >= h_threshold
    elif event_type == "absval":
        return Z_end.abs() >= h_threshold
    elif event_type == "upper_change":
        return Z_end - Z_start >= h_threshold
    elif event_type == "lower_change":
        return Z_end - Z_start <= -h_threshold
    elif event_type == "start_upper":
        return Z_start >= h_threshold
    raise NotImplementedError(f"event_type={event_type!r}")

Zs_tr, Ze_tr, vidx_tr = data_processor.get_z_windows_train_aligned()
Zs_te, Ze_te, vidx_te = data_processor.get_z_windows_test()
X_train_events = X_train[vidx_tr][event_mask(Zs_tr, Ze_tr)]
X_test_events  = X_test[vidx_te][event_mask(Zs_te, Ze_te)]

# ── Load generated samples (optional — only needed for the "Conditional
# Generated" panel, which requires an h-function/ConditionalGenerator run,
# or the cGAN baseline, run under --gan) ────────────────────────────────────
_dir  = os.path.dirname(os.path.abspath(__file__))  # analysis/
_root = os.path.dirname(_dir)                        # repo root

if _args.gan:
    _gen_dir = os.path.join(_root, 'gan_baseline', 'gan_results')
    _gen_train_path = os.path.join(_gen_dir, 'gan_generated_samples_train.pt')
    _cond_panel_label = "cGAN Generated"
    _results_dir = os.path.join(_dir, "gan_results")
else:
    _gen_train_path = os.path.join(_root, 'generated_samples_train.pt')
    _cond_panel_label = "Conditional Generated"
    _results_dir = os.path.join(_dir, "diffusion_results")

# ONE conditional-generated batch is compared against both the train-event and
# test-event panels below (gen_train reused as gen_test) -- generation has no
# knowledge of which real split it's being compared to, so a second draw would
# just be a fresh noisy sample from the same underlying distribution rather
# than anything meaningfully different. Only the train path is read; the
# _test.pt file main.py also writes is redundant (same tensor) and unused here.
gen_train = torch.load(_gen_train_path, map_location='cpu') if os.path.exists(_gen_train_path) else None
gen_test  = gen_train

if gen_train is None:
    print(f"No {_gen_train_path} found — skipping '{_cond_panel_label}' panel.")

# ── Unconditional samples (diffusion mode only -- --gan skips this entirely:
# no diffusion checkpoint load, no sampling, GAN-only comparison) ──
# Loaded from the shared, seeded file unconditional_gen.py writes, rather than
# regenerated here -- two independent (unseeded) generations would cost the
# sampling time twice AND give two different random draws, so this script's
# "Unconditional Generated" panel and unconditional_gen.py's own diagnostics
# would silently be looking at different samples instead of the same baseline.
if not _args.gan:
    _uncond_path = os.path.join(_root, "unconditional_samples.pt")
    if os.path.exists(_uncond_path):
        uncond = torch.load(_uncond_path, map_location='cpu')
        print(f"Loaded {_uncond_path} (n={len(uncond)}) -- run analysis/unconditional_gen.py "
              f"first if this is stale.")
    else:
        print(f"No {_uncond_path} found -- generating (run analysis/unconditional_gen.py "
              f"beforehand to cache this and skip regenerating it here).")
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

        N_uncond   = config.conditional.n_gen_samples
        batch_size = 128
        n_batches  = -(-N_uncond // batch_size)  # ceil
        print(f"Generating {N_uncond} unconditional samples (batch={batch_size}, {n_batches} batches)...")
        chunks = []
        for start in tqdm(range(0, N_uncond, batch_size), total=n_batches, desc="Unconditional batches"):
            bs = min(batch_size, N_uncond - start)
            chunks.append(diffusion_model.sample(
                batch_size=bs,
                num_steps=config.conditional.num_steps,
                stoch=config.conditional.stoch,
            ).cpu())
        uncond = torch.cat(chunks, dim=0)  # (N, A, T)
        torch.save(uncond, _uncond_path)
        print(f"Saved {_uncond_path}")

# ── Last-day return matrices: shape (N, A) ────────────────────────────────────
# Real: X shape is (N, T, A) → last day = X[:, -1, :]
# Generated: shape is (N, A, T) → last day = gen[:, :, -1]

panels_train = [
    ("Real Train (all)",           X_train[:, -1, :].numpy()),
    ("Real Train (event windows)", X_train_events[:, -1, :].numpy()),
]
if not _args.gan:
    panels_train.append(("Unconditional Generated", uncond[:, :, -1].numpy()))
if gen_train is not None:
    panels_train.append((_cond_panel_label, gen_train[:, :, -1].numpy()))

panels_test = [
    ("Real Test (all)",            X_test[:, -1, :].numpy()),
    ("Real Test (event windows)",  X_test_events[:, -1, :].numpy()),
]
if not _args.gan:
    panels_test.append(("Unconditional Generated", uncond[:, :, -1].numpy()))
if gen_test is not None:
    panels_test.append((_cond_panel_label, gen_test[:, :, -1].numpy()))

# ── Print matrices ─────────────────────────────────────────────────────────────
for split_label, panels in [("TRAIN", panels_train), ("TEST", panels_test)]:
    print(f"\n{'='*60}")
    print(f"── Correlation Matrices ({split_label}) ──")
    for lbl, arr in panels:
        C = np.corrcoef(arr.T)
        print(f"\n{lbl}  (n={len(arr)}):")
        print(pd.DataFrame(C, index=plot_tickers, columns=plot_tickers).round(3).to_string())


# ── Off-diagonal correlation RMSE vs Real (event windows) ─────────────────────
def off_diag_rmse(C_a: np.ndarray, C_b: np.ndarray) -> float:
    """RMSE of the off-diagonal entries between two same-shaped correlation
    matrices (diagonal is always 1 for both, so excluding it isolates the
    actual cross-asset structure being compared)."""
    mask = ~np.eye(C_a.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((C_a[mask] - C_b[mask]) ** 2)))


def event_ref_corr(panels):
    """Correlation matrix of this split's 'Real ... (event windows)' panel —
    the target every generated panel's off-diagonal RMSE is measured against."""
    for lbl, arr in panels:
        if "event windows" in lbl:
            return np.corrcoef(arr.T)
    raise ValueError("no 'Real ... (event windows)' panel found")


def print_corr_rmse(split_label, panels):
    C_ref = event_ref_corr(panels)
    print(f"\n── Off-Diagonal Corr RMSE vs Real {split_label} (event windows) ──")
    rows = []
    for lbl, arr in panels:
        if "event windows" in lbl:
            continue
        rmse = off_diag_rmse(np.corrcoef(arr.T), C_ref)
        print(f"  {lbl:<28} RMSE = {rmse:.4f}")
        rows.append((lbl, rmse))
    return rows


# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_matrices(panels, title, fname, vmin, vmax, fmt, rmse_rows=None):
    tick_lbl  = [t.upper() for t in plot_tickers]
    font_size = max(8, min(13, 40 // n_plot))

    n_panels = len(panels)
    n_cols   = 2
    n_rows   = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    for ax, (lbl, arr) in zip(axes, panels):
        C  = np.corrcoef(arr.T) if "corr" in fname else np.cov(arr.T)
        im = ax.imshow(C, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(n_plot)); ax.set_xticklabels(tick_lbl, fontsize=10, rotation=45, ha="right")
        ax.set_yticks(range(n_plot)); ax.set_yticklabels(tick_lbl, fontsize=10)
        ax.set_title(f"{lbl}\n(n={len(arr)})", fontsize=10, fontweight="bold", pad=8)
        for r in range(n_plot):
            for c in range(n_plot):
                v = C[r, c]
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=font_size,
                        fontweight="bold", color="white" if abs(v) > 0.6 * abs(vmax) else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    event_lbl = (data_processor.macro_col.upper()
                 if config.data.latent_method is None
                 else f"latent ({config.data.latent_method})")
    fig.suptitle(
        f"{title}\n(event: {event_lbl} {event_type} ≥ {h_threshold} std"
        f" [top {config.hfunction.event_threshold:.1%}],  last-day returns)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    if rmse_rows:
        rmse_line = "  |  ".join(f"{lbl} off-diag RMSE = {rmse:.4f}" for lbl, rmse in rmse_rows)
        fig.subplots_adjust(bottom=0.12)
        fig.text(0.5, 0.02, rmse_line, ha="center", va="bottom",
                  fontsize=10, fontweight="bold")

    os.makedirs(_results_dir, exist_ok=True)
    out = os.path.join(_results_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# ── Train figures ─────────────────────────────────────────────────────────────
rmse_train = print_corr_rmse("Train", panels_train)

plot_matrices(panels_train, "Correlation Matrices — Last-Day Returns (Train)",
              "corr_matrices_train.png", vmin=-1, vmax=1, fmt="{:.2f}", rmse_rows=rmse_train)

# ── Test figures ──────────────────────────────────────────────────────────────
rmse_test = print_corr_rmse("Test", panels_test)

plot_matrices(panels_test, "Correlation Matrices — Last-Day Returns (Test)",
              "corr_matrices_test.png", vmin=-1, vmax=1, fmt="{:.2f}", rmse_rows=rmse_test)
