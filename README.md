# Conditional Diffusion Generation for Financial Time Series

Generates synthetic financial time series conditioned on rare market events (TSLA crash scenarios) using VP-SDE diffusion with Doob h-transform guidance.

## Architecture

Three models trained sequentially:

1. **DiffusionModel** — VP-SDE score network (Dual-axis Transformer). Unconditional 64-day path generation.
2. **HFunctionTrainer** — Predicts P(event | x_t, t) from noisy diffusion trajectories.
3. **ConditionalGenerator** — Guided sampling via Doob h-transform: drift += `(1 + η) · g² · ∇log h`. Optional Q-model to avoid autograd at sampling time.

**Dual-axis Transformer**: each block alternates temporal self-attention (each asset over 64 time steps) and cross-asset self-attention (all 4 assets at each time step), conditioned on diffusion time `t` via AdaLN.

## Assets & Event Condition

- **Assets**: AAPL, AMZN, JPM, TSLA
- **Event**: TSLA last 10-day raw log return sum ≤ −10%
  - Stored as `event_threshold_raw = -0.10` in config
  - Converted to standardised threshold at runtime: `event_threshold = -0.10 / σ_TSLA`

## Data Window

Selected via `analyze_regime.py` for train/test distribution similarity:

| Split | Date Range | Sequences |
|-------|-----------|-----------|
| Full  | 2014-01-28 ~ 2025-10-17 | 2822 |
| Train | 2014-01-28 ~ 2022-11-07 | 2148 |
| Test  | 2022-11-08 ~ 2025-10-17 | 674  |

Set in `config/config.py`: `start_date`, `end_date`, `train_end_date`.

## Project Structure

```
cdg_finance/
├── config/
│   └── config.py                  # All hyperparameters (DataConfig, DiffusionConfig, ...)
├── data/
│   └── data_processor.py          # Load → weekday removal → standardize → winsorize → sequences
├── models/
│   ├── transformer_score.py       # Dual-axis Transformer score network
│   ├── diffusion_model.py         # VP-SDE wrapper: train / sample
│   ├── hfunction.py               # H-function trainer
│   └── conditional_generator.py   # Doob h-transform guided sampler + Q-model
├── utils/
│   ├── portfolio.py               # Min-Var / Risk-Parity / Equal-Weight analysis & plots
│   └── helpers.py                 # set_seed, misc
├── main.py                        # Full pipeline: diffusion → H-function → Q-model → analysis
├── pretrain_and_plot.py           # Train diffusion only and plot distributions
├── sample_insample.py             # Pretrain-filtered vs conditional (train set)
├── sample_outsample.py            # Real test events vs conditional (test set)
├── compare_train_test_events.py   # Real train events vs real test events (no generation)
├── analyze_regime.py              # Sliding-window search for best data window
├── run_sweep_pretrain.sh          # Parameter sweep (η, stoch, Q-model)
└── cleanup_wandb.sh               # Remove wandb credentials before sharing
```

## Quick Start

### 1. Full Training + Sweep

```bash
nohup bash -c "python -u main.py --train-q-model --no-wandb 2>&1 | tee logs/train_all.log \
  && bash run_sweep_pretrain.sh 2>&1 | tee logs/sweep_pretrain_master.log" > /dev/null 2>&1 &
```

### 2. Retrain H-function + Q-model only (diffusion already trained)

```bash
python -u main.py --skip-diffusion-training --train-q-model --no-wandb 2>&1 | tee logs/train_hq.log
```

### 3. Single run (no sweep)

```bash
# In-sample: pretrain-filtered baseline vs conditional
python sample_insample.py --stoch 0 --eta 1.0 --compare-pretrain --no-wandb

# Out-of-sample: real test events vs conditional
python sample_outsample.py --stoch 0 --eta 1.0 --no-wandb

# Train/test event distribution comparison (no generation)
python compare_train_test_events.py
```

### 4. Data window analysis

```bash
python -u analyze_regime.py 2>&1 | tee results/regime/analyze_regime.log
```

## Key Config Parameters

```python
# config/config.py

# Data window
start_date      = "2014-01-28"
end_date        = "2025-10-17"
train_end_date  = "2022-11-07"

# Event condition (raw; standardised threshold computed at runtime)
event_threshold_raw = -0.10   # TSLA last 10-day log return sum ≤ -10%

# Diffusion model
n_epochs   = 600
embed_dim  = 256
n_heads    = 8
n_layers   = 8

# H-function
train_stoch = 0.5
n_epochs    = 300

# Conditional generation (recommended)
stoch = 0      # ODE (stoch=0) consistently outperforms SDE (stoch=1)
eta   = 0      # guidance scale; sweep range typically -1.5 ~ 1.0
```

## Checkpoints

All models saved to `ckpt_new/`:
- `ckpt_new/diffusion_model.pt`
- `ckpt_new/hfunction.pt`
- `ckpt_new/q_model.pt`

## Sweep Output

`run_sweep_pretrain.sh` sweeps over (stoch, η, Q-model) combinations:
- **In-sample plots**: `results/sweep_pretrain/portfolio_insample_*.png`
- **Out-of-sample plots**: `results/sweep_pretrain/portfolio_outsample_*.png`
- **Stats CSV**: `results/sweep_pretrain/insample_stats.csv`, `outsample_stats.csv`
- **Pretrain cache**: `results/sweep_pretrain/pretrain_events_cache_stoch{s}.pt` (reused across η sweep)

## Implementation Notes

- `stoch=0` (ODE/DDIM-like) recommended for conditional generation: noise overwhelms guidance signal at `stoch=1`
- `eta=-1` → no guidance (identical to pretrain); `eta>0` → stronger conditioning
- Last sampling step always returns `mean_x` (no final noise injection)
- Weekday-effect removal is inverted using actual historical weekday per sequence, not random assignment
- Event detection on real data: `X[:, -event_window:, event_asset_idx]` (shape is N, T, A)
- Event detection on generated samples: `X[:, event_asset_idx, -event_window:]` (shape is N, A, T)
