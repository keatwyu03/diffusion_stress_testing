# CDG Finance — Project Overview

## What It Does

Conditional Diffusion Generation (CDG) for financial time series. Trains a VP-SDE score model using a **dual-axis Transformer** architecture on financial returns, then conditions the reverse diffusion on a user-defined market event using Doob's h-transform. Portfolio strategies (min-variance, risk-parity, equal-weight) are evaluated on generated vs. real event windows.

---

## Directory Structure

```
CDG_Finance/
├── config/config.py                         # All hyperparameters as dataclasses
├── data/data_processor.py                   # Full preprocessing pipeline (note: file contains two DataProcessor class defs — second one is the current version)
├── models/
│   ├── transformer_score.py                 # Dual-axis Transformer score network (primary score model)
│   ├── diffusion_model.py                   # VP-SDE wrapper: train / sample (UNet1D or Transformer)
│   ├── hfunction.py                         # HFunctionCNN + HFunctionTransformer + HFunctionTrainer
│   └── conditional_generator.py             # Doob h-transform guided sampler + Q-model (Transformer-based)
├── utils/
│   ├── helpers.py                           # set_seed
│   └── portfolio.py                         # Portfolio strategies + stats/plots
├── main.py                                  # Full end-to-end pipeline
├── pretrain_and_plot.py                     # Train diffusion only and plot distributions
├── sample_insample.py                       # Load checkpoints, generate for train events
├── sample_outsample.py                      # Load checkpoints, generate for test events
├── compare_train_test_events.py             # Real train events vs real test events (no generation)
├── analyze_regime.py                        # Sliding-window search for best train/test data window
├── diffusion_model_analysis/
│   ├── unconditional_gen.py                 # Diagnostics table, marginal KDEs (last-day + cumulative), pairwise joint contour plots
│   ├── conditonal_gen.py                    # Conditional vs real event window marginal KDEs
│   ├── cov.py                               # Correlation/covariance matrix comparison (real vs uncond vs cond)
│   └── losses.py                            # Score + H-function loss/accuracy curves (auto-discovers CSVs)
├── explore/macro_data_new.csv               # Macro dataset (FRED + yfinance): unemp, sp500, baa
├── Stocks_logret.csv                        # Stock log-return dataset: AAPL, AMZN, JPM, TSLA
├── ckpt_new/                                # Active checkpoint directory (main.py reads/writes here)
│   ├── diffusion_model.pt
│   ├── hfunction.pt
│   └── q_model.pt
├── checkpoints/                             # Old checkpoint directory (legacy)
├── run_training.sh                          # Full training launch script
├── run_sampling.sh                          # Sampling launch script
└── run_sweep_pretrain.sh                    # (η, stoch, Q-model) sweep over pretrained checkpoints
```

---

## Data

Two datasets exist; which is used depends on `config.data.csv_path` and `config.data.tickers`:

| Dataset | Path | Tickers | Notes |
|---|---|---|---|
| Macro (config default) | `explore/macro_data_new.csv` | `["unemp", "sp500", "baa"]` | 3 channels |
| Stocks (README target) | `Stocks_logret.csv` | `["AAPL", "AMZN", "JPM", "TSLA"]` | 4 channels; used by `analyze_regime.py` |

The README documents the intended production configuration as the stock dataset with a specific data window selected by `analyze_regime.py`:

| Split | Date Range | Sequences |
|---|---|---|
| Full | 2014-01-28 ~ 2025-10-17 | 2822 |
| Train | 2014-01-28 ~ 2022-11-07 | 2148 |
| Test | 2022-11-08 ~ 2025-10-17 | 674 |

These dates are set via `config.data.start_date`, `config.data.end_date`, `config.data.train_end_date` (all `None` by default in config.py).

**Preprocessing pipeline** (`DataProcessor.process_all`):
1. Load CSV, parse Date index, filter to `[start_date, end_date]` if set
2. Remove weekday effect (subtract per-weekday mean using `weekday_col` from CSV)
3. Standardize (divide by std only — no mean subtraction in current version)
4. Winsorize (clip at `winsorize_lower` / `winsorize_upper` percentiles; **both default 0.0** = no winsorization)
5. Create rolling 64-day windows → shape `(N, seq_len, channels)` for event detection
6. Track `start_weekdays` per window for accurate return inversion
7. Train/test split: by `train_end_date` if set, else last `test_days=2000` rows

`get_diffusion_data()` returns transposed windows `(N, channels, seq_len)` = `(N, A, T)` for UNet1D/Transformer input, training portion only.

---

## Models

### 1. FinancialTransformerScore (`models/transformer_score.py`)
The primary score network. A dual-axis Transformer conditioned on diffusion time `t` via AdaLN.

- **Input projection:** each scalar return value → `embed_dim` (linear)
- **Positional embeddings:** learnable temporal `(1, 1, T, D)` + per-asset `(1, A, 1, D)`
- **Time conditioning:** Gaussian Fourier features → 2-layer MLP → `cond_dim`
- **`DualAxisBlock`** (N of these):
  1. Temporal self-attention: `(B*A, T, D)` — each asset over 64 time steps
  2. Cross-asset self-attention: `(B*T, A, D)` — all assets at each time step
  3. Position-wise FFN (GELU, 4× expansion)
  - Each sub-layer uses `AdaLN` (adaptive layer norm): time embedding → scale + shift
- **Output:** linear projection back to scalar per position → `ScoreOutput.sample` shape `(B, A, T)`
- **Default params:** `embed_dim=128, n_heads=4, n_layers=6, cond_dim=128` → ~2M parameters

### 2. DiffusionModel (`models/diffusion_model.py`)
- **Architecture:** Transformer (`arch="transformer"`) or UNet1D (`arch="unet"`)
  - Config default: `arch="unet"` — README/intended: Transformer
  - When `arch="transformer"`: instantiates `FinancialTransformerScore`
  - When `arch="unet"`: instantiates HuggingFace `UNet1DModel`
- **SDE type:** Variance Preserving (VP)
  - `b_min=0.1`, `b_max=3.25`
  - `marginal_prob_mean(t) = exp(-0.5 * ∫β)`, `marginal_prob_std(t) = sqrt(1 - exp(-∫β))`
- **Loss:** Denoising score matching: `mean(sum((score * σ + z)², dim=(channels, seq)))`
- **Optimizer:** AdamW + `ReduceLROnPlateau` (patience=50, factor=0.5)
- **Config defaults:** `n_epochs=100, batch_size=75, lr=1e-4`; README target: `n_epochs=600`
- **Sampling:** Euler-Maruyama on adaptive VP std grid (`num_steps=200`)
  - `adjust = (1 + stoch²) / 2`
  - `drift += adjust * g² * score * dt`
  - `x += stoch * sqrt(dt) * g * noise`
  - Last step returns `mean_x` (no final noise)
- **Checkpoint:** `ckpt_new/diffusion_model.pt`

### 3. HFunctionCNN / HFunctionTransformer (`models/hfunction.py`)
Both implement `forward(x, t) → sigmoid scalar ∈ [0,1]`.

**HFunctionCNN:**
- Gaussian Fourier time embedding → Linear → SiLU
- Conv1d stack: channels → 16 → 64 → 256, GroupNorm + SiLU, AdaptiveAvgPool1d(1)
- FC head: 256+embed_dim → 128 → 64 → 1 → Sigmoid

**HFunctionTransformer:**
- Same `DualAxisBlock` stack as the score network, with global average pooling → Sigmoid head
- Config default: `arch="cnn"` — README/intended: Transformer

**HFunctionTrainer:**
- **Purpose:** Learn P(event | x_t, t) — Doob's h-function
- **Training data:** synthetic diffusion paths from `diffusion_model.sample(..., return_path=True)`
  - `train_batch_size=2048` paths, `train_stoch=0.5`
  - Mini-batch training: `h_mini_batch_size=512` per gradient step
- **Loss:** MSE on binary (or soft sigmoid) event labels from terminal states
- **Grad clipping:** `max_norm=1.0`
- **Optimizer:** AdamW + `ReduceLROnPlateau`
- **Config defaults:** `n_epochs=1000, lr=1e-4, weight_decay=1e-4`; README target: `n_epochs=300`
- **Checkpoint:** `ckpt_new/hfunction.pt`

### 4. ConditionalGenerator + GradientHUNet (`models/conditional_generator.py`)
- **Guided reverse SDE:**
  - Base drift: `g² * score`
  - Guidance: `(1 + eta) * g² * (grad_h / h)` where `grad_h = ∇H` and `h = H(x, t)`
  - Default `eta=1.0` (config); note: `generate()` signature defaults `eta=150.0` — config value is what's passed
- **Gradient computation:** autograd by default (enables grad inside `@no_grad` context)
- **Q-model (`GradientHUNet`):** `FinancialTransformerScore`-based network trained to approximate `∇H / H` via covariation loss — avoids autograd at sampling time
  - Q-model architecture: `embed_dim=64, n_heads=4, n_layers=4, cond_dim=64`
- **Known:** `_sample_batch` hardcodes `(batch_size, 4, 64)` — not driven by config channels/seq_len
- **Checkpoint:** `ckpt_new/q_model.pt`

---

## Event Condition

Configured in `HFunctionConfig`:

| Field | Config Default | README Target | Meaning |
|---|---|---|---|
| `event_type` | `"absval"` | `"sum"` | Metric applied to last window |
| `event_asset_idx` | `0` | `3` (TSLA) | Which channel to watch |
| `event_window` | `3` | `10` | Lookback period (days) |
| `event_threshold` | `1.2` | `-0.10 / σ_TSLA` | Trigger threshold |
| `constraint_mode` | `"hard"` | `"hard"` | Hard Doob h-constraint |

**Event types:**
- `"sum"`: `last_window.sum(dim=1) <= threshold` (negative shock)
- `"change"`: `|last_window[:,-1] - last_window[:,0]| >= threshold`
- `"absval"`: `|last_window[:,-1]| >= threshold`

**Mask logic (channels-last, for real data):**
```python
last_window = X[:, -event_window:, event_asset_idx]  # (N, T, A) → (N, window)
```

**Mask logic in H-function training (channels-first, generated paths):**
```python
terminal[:, event_asset_idx, -event_window:]  # (N, A, T) → (N, window)
```

---

## Pipeline (main.py)

```
Step 1: DataProcessor.process_all()
Step 2: DiffusionModel.train() → ckpt_new/diffusion_model.pt
Step 3: HFunctionTrainer.train() → ckpt_new/hfunction.pt
Step 4: Extract event masks from train + test sets
Step 5 (optional): ConditionalGenerator.train_q_model() → ckpt_new/q_model.pt
Step 6: ConditionalGenerator.generate() for train events + test events
         → generated_samples_train.pt, generated_samples_test.pt
Step 7: PortfolioAnalyzer → min-var / risk-parity / equal-weight stats + plots
         → results/portfolio_comparison_insample.png
         → results/portfolio_comparison_outsample.png
```

**CLI flags for main.py:**
- `--skip-diffusion-training` — load from `ckpt_new/diffusion_model.pt`
- `--skip-hfunction-training` — load from `ckpt_new/hfunction.pt`
- `--skip-qmodel-training` — skip Q-model
- `--skip-conditional` — exit after step 4 (no generation or portfolio analysis)
- `--train-q-model` — force Q-model training
- `--no-wandb` — disable W&B logging

**Data Window Analysis (`analyze_regime.py`):**
Sliding-window search over `Stocks_logret.csv` (hardcoded to AAPL/AMZN/JPM/TSLA) to find the `[start, end]` window and 75/25 train/test split where event portfolio returns are most similar across strategies. Outputs ranked CSV + plot to `results/regime/`.

---

## Config Defaults Summary

```python
# Data
csv_path      = "explore/macro_data_new.csv"
tickers       = ["unemp", "sp500", "baa"]
seq_len       = 64
test_days     = 2000                  # used only when train_end_date is None
start_date    = None                  # None = use all data
end_date      = None
train_end_date= None
winsorize_lower = 0.0                 # 0.0 = no winsorization
winsorize_upper = 0.0

# Diffusion
in_channels   = 3, out_channels = 3, sample_size = 64
layers_per_block = 3, block_out_channels = (64, 128, 256)   # UNet only
b_min=0.1, b_max=3.25
arch="unet"                           # "unet" or "transformer"
embed_dim=128, n_heads=4, n_layers=6, cond_dim=128           # Transformer params
n_epochs=100, batch_size=75, lr=1e-4, num_steps=200
scheduler_patience=50, scheduler_factor=0.5

# H-Function
asset_dim=3, time_steps=64, embed_dim=128
event_type="absval", event_asset_idx=0, event_window=3, event_threshold=1.2
arch="cnn"                            # "cnn" or "transformer"
constraint_mode="hard", reward_sharpness=50.0
train_batch_size=2048, train_stoch=0.5
h_mini_batch_size=512
n_epochs=1000, lr=1e-4, weight_decay=1e-4

# Conditional Gen
batch_size=32, num_steps=200, stoch=0, eta=1.0
use_q_model=False, constraint_mode="hard"
q_model_epochs=500, q_model_lr=1e-4
q_model_train_batch_size=4096, q_model_mini_batch_size=256, q_model_train_stoch=0.5
q_embed_dim=64, q_n_heads=4, q_n_layers=4, q_cond_dim=64

# Portfolio
window_for_cov=54, last_days_sum=5
portfolio_tickers=["sp500", "baa"]
```

---

## Analysis Scripts

All analysis scripts auto-detect config from `get_default_config()` and load checkpoints from `ckpt_new/`.

- **`diffusion_model_analysis/unconditional_gen.py`** — three figures:
  1. Diagnostics table: per-asset mean/std/quantiles for last-day and cumulative returns (real vs generated)
  2. Marginal KDEs: last-day returns and 64-day cumulative returns per asset (real train/test vs unconditional generated)
  3. Pairwise joint density contour plots for all asset pairs (real vs generated, last-day)
  - Outputs to `diffusion_model_analysis/results/`

- **`diffusion_model_analysis/conditonal_gen.py`** — two figures:
  1. Last-day return KDEs: real event windows vs conditional generated
  2. Cumulative return KDEs: real event windows vs conditional generated
  - Reads `generated_samples_train.pt` / `generated_samples_test.pt` from repo root
  - Outputs to `diffusion_model_analysis/`

- **`diffusion_model_analysis/cov.py`** — correlation and covariance matrix heatmaps:
  - Four panels per figure: Real (all) / Real (event windows) / Unconditional Generated / Conditional Generated
  - Generates unconditional samples on-the-fly (`N_uncond=5000, stoch=0`)
  - Prints matrices to stdout; saves 4 PNGs to `diffusion_model_analysis/results/`

- **`diffusion_model_analysis/losses.py`** — training loss curves:
  - Auto-discovers `score_losses.csv` / `h_losses.csv` by searching recursively from repo root
  - Plots score loss; if H-function CSV found, adds H-loss + accuracy/pos_rate panels
  - Output: `diffusion_model_analysis/results/train_losses.png`

---

## Key Tensor Shapes

| Tensor | Shape | Notes |
|---|---|---|
| `X_train` / `X_test` | `(N, seq_len, channels)` = `(N, 64, A)` | Channels-last; for event detection |
| Diffusion training data | `(N, channels, seq_len)` = `(N, A, 64)` | Channels-first; for UNet1D/Transformer input |
| `generated_samples_*.pt` | `(N, channels, seq_len)` = `(N, A, 64)` | Output of ConditionalGenerator |
| H-function input | `(B, channels, seq_len)` = `(B, A, 64)` | Same layout as diffusion |
| `t_grid` (from sample path) | `(num_steps, batch_size)` | Time steps per trajectory |
| `y_grid` (from sample path) | `(num_steps, batch_size, channels, seq_len)` | Full reverse-SDE paths |

---

## Known Issues / Gotchas

1. **Duplicate `DataProcessor` in `data/data_processor.py`:** The file contains two complete class definitions. The first (lines ~1–234) is the legacy version; the second (lines ~236–489) is the current version with `start_date`/`end_date`/`train_end_date` support. Python will use whichever is defined last. This should be cleaned up.

2. **`_sample_batch` hardcodes shape:** `ConditionalGenerator._sample_batch` initializes `init_x = torch.randn(batch_size, 4, 64, ...)` — hardcoded 4 channels and 64 seq_len. Breaks for other asset counts or sequence lengths.

3. **`arch` default mismatch:** `DiffusionConfig.arch = "unet"` but `DiffusionModel.__init__` defaults to `arch="transformer"`. Same pattern for H-function: `HFunctionConfig.arch = "cnn"` but `HFunctionTrainer` defaults to `arch="transformer"`. The config value is what main.py passes, so config governs at runtime.

4. **`eta` discrepancy:** `ConditionalGenConfig.eta = 1.0` (default) but `ConditionalGenerator.generate()` signature defaults `eta=150.0`. The config value is passed at runtime, so this only matters if `generate()` is called directly without a config.

5. **Config vs README target:** Config defaults still point to macro data (`explore/macro_data_new.csv`, `["unemp", "sp500", "baa"]`, no date filtering). The README describes the intended production setup using `Stocks_logret.csv` with AAPL/AMZN/JPM/TSLA and date-bounded splits — these must be overridden in config at runtime.

6. **Wandb disabled by default** (`WandbConfig.enabled = False`).

---

## Architecture Reference: Dual-Axis Transformer

```
Input: (B, A, T)  where A=assets, T=seq_len
  ↓ input_proj (Linear 1→D)
  → (B, A, T, embed_dim)
  + temporal_pos (1, 1, T, D) + asset_emb (1, A, 1, D)

For each DualAxisBlock:
  1. Temporal attn:   reshape (B*A, T, D) → MHA → residual
  2. Asset attn:      reshape (B*T, A, D) → MHA → residual
  3. FFN:             GELU, 4× expansion, residual
  (each sub-layer: AdaLN with t_emb conditioning)

  ↓ output_proj (Linear D→1)
Output: (B, A, T)  — score or noise prediction
```

AdaLN: `LayerNorm(x) * (1 + scale(t)) + shift(t)` where scale/shift come from a linear projection of the time embedding.

---

## Experiments

### 2026-05-19
**Issue:** Generated std ≈ 2× real std. `corr(score, x)` collapses to ~0 by t=0.59.  
**Cause:** `(score * σ + z)²` loss has σ²(t) weighting → no gradient signal at small t. Also `adjust` hardcoded to 1.0 instead of `(1+stoch²)/2`.  
**Fix:** Switched to noise parameterization — predict `z`, loss = `(eps_pred - z)²`. Score recovered as `-eps_pred / σ(t)` at sampling. Fixed `adjust`.  
**Config:** `n_epochs=600`, `warmup_epochs=150`, `lr=1e-4`, `batch_size=75`, cosine schedule.

> Note: The loss function in the current code still uses `(score * σ + z)²`. The sampling `adjust` fix is in place. The noise-parameterization experiment may have been reverted.
