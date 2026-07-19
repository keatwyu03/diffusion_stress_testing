import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
    start_date=config.data.start_date,
    end_date=config.data.end_date,
    train_end_date=config.data.train_end_date,
    window_shift=config.data.window_shift,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)
data_processor.process_all()

tickers      = config.data.tickers
n_assets     = len(tickers) - 1
plot_tickers = tickers[1:]
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
    raise NotImplementedError(f"event_type={event_type!r}")

Zs_tr, Ze_tr, vidx_tr = data_processor.get_z_windows_train_aligned()
Zs_te, Ze_te, vidx_te = data_processor.get_z_windows_test()
X_train_events = X_train[vidx_tr][event_mask(Zs_tr, Ze_tr)]
X_test_events  = X_test[vidx_te][event_mask(Zs_te, Ze_te)]

# ── Load generated samples (optional — only needed for the "Conditional
# Generated" panel, which requires an h-function/ConditionalGenerator run) ─────
_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)

_gen_train_path = os.path.join(_root, 'generated_samples_train.pt')
_gen_test_path  = os.path.join(_root, 'generated_samples_test.pt')

gen_train = torch.load(_gen_train_path, map_location='cpu') if os.path.exists(_gen_train_path) else None
gen_test  = torch.load(_gen_test_path,  map_location='cpu') if os.path.exists(_gen_test_path)  else None

if gen_train is None or gen_test is None:
    print("No generated_samples_{train,test}.pt found — skipping 'Conditional Generated' panel.")

# ── Generate unconditional samples ────────────────────────────────────────────
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

N_uncond   = 2000
batch_size = 128
print(f"Generating {N_uncond} unconditional samples (batch={batch_size})...")
chunks = []
for start in range(0, N_uncond, batch_size):
    bs = min(batch_size, N_uncond - start)
    chunks.append(diffusion_model.sample(
        batch_size=bs,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
    ).cpu())
uncond = torch.cat(chunks, dim=0)  # (N, A, T)

# ── Last-day return matrices: shape (N, A) ────────────────────────────────────
# Real: X shape is (N, T, A) → last day = X[:, -1, :]
# Generated: shape is (N, A, T) → last day = gen[:, :, -1]

panels_train = [
    ("Real Train (all)",           X_train[:, -1, :].numpy()),
    ("Real Train (event windows)", X_train_events[:, -1, :].numpy()),
    ("Unconditional Generated",    uncond[:, :, -1].numpy()),
]
if gen_train is not None:
    panels_train.append(("Conditional Generated", gen_train[:, :, -1].numpy()))

panels_test = [
    ("Real Test (all)",            X_test[:, -1, :].numpy()),
    ("Real Test (event windows)",  X_test_events[:, -1, :].numpy()),
    ("Unconditional Generated",    uncond[:, :, -1].numpy()),
]
if gen_test is not None:
    panels_test.append(("Conditional Generated", gen_test[:, :, -1].numpy()))

# ── Print matrices ─────────────────────────────────────────────────────────────
for split_label, panels in [("TRAIN", panels_train), ("TEST", panels_test)]:
    print(f"\n{'='*60}")
    print(f"── Correlation Matrices ({split_label}) ──")
    for lbl, arr in panels:
        C = np.corrcoef(arr.T)
        print(f"\n{lbl}  (n={len(arr)}):")
        print(pd.DataFrame(C, index=plot_tickers, columns=plot_tickers).round(3).to_string())

    print(f"\n── Covariance Matrices ({split_label}) ──")
    for lbl, arr in panels:
        C = np.cov(arr.T)
        print(f"\n{lbl}  (n={len(arr)}):")
        print(pd.DataFrame(C, index=plot_tickers, columns=plot_tickers).round(4).to_string())


# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_matrices(panels, title, fname, vmin, vmax, fmt):
    tick_lbl  = [t.upper() for t in plot_tickers]
    font_size = max(8, min(13, 40 // n_plot))

    n_panels = len(panels)
    n_cols   = 2
    n_rows   = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    for ax, (lbl, arr) in zip(axes, panels):
        C  = np.corrcoef(arr.T) if "corr" in fname else np.cov(arr.T)
        im = ax.imshow(C, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(n_plot)); ax.set_xticklabels(tick_lbl, fontsize=10)
        ax.set_yticks(range(n_plot)); ax.set_yticklabels(tick_lbl, fontsize=10)
        ax.set_title(f"{lbl}\n(n={len(arr)})", fontsize=10, fontweight="bold", pad=8)
        for r in range(n_plot):
            for c in range(n_plot):
                v = C[r, c]
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=font_size,
                        fontweight="bold", color="white" if abs(v) > 0.6 * abs(vmax) else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    event_lbl = (tickers[config.hfunction.event_asset_idx].upper()
                 if config.data.latent_method is None
                 else f"latent ({config.data.latent_method})")
    fig.suptitle(
        f"{title}\n(event: {event_lbl} {event_type} ≥ {h_threshold:.3f} std"
        f" [top {config.hfunction.event_threshold:.0%}],  last-day returns)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    os.makedirs(os.path.join(_dir, "results"), exist_ok=True)
    out = os.path.join(_dir, "results", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# ── Train figures ─────────────────────────────────────────────────────────────
all_cov = [np.cov(arr.T) for _, arr in panels_train]
cov_lim = float(np.abs(np.concatenate([C.ravel() for C in all_cov])).max())

plot_matrices(panels_train, "Correlation Matrices — Last-Day Returns (Train)",
              "corr_matrices_train.png", vmin=-1, vmax=1, fmt="{:.2f}")
plot_matrices(panels_train, "Covariance Matrices — Last-Day Returns (Train)",
              "cov_matrices_train.png", vmin=-cov_lim, vmax=cov_lim, fmt="{:.3f}")

# ── Test figures ──────────────────────────────────────────────────────────────
all_cov_t = [np.cov(arr.T) for _, arr in panels_test]
cov_lim_t = float(np.abs(np.concatenate([C.ravel() for C in all_cov_t])).max())

plot_matrices(panels_test, "Correlation Matrices — Last-Day Returns (Test)",
              "corr_matrices_test.png", vmin=-1, vmax=1, fmt="{:.2f}")
plot_matrices(panels_test, "Covariance Matrices — Last-Day Returns (Test)",
              "cov_matrices_test.png", vmin=-cov_lim_t, vmax=cov_lim_t, fmt="{:.3f}")
