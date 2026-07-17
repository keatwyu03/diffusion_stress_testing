# CDG Finance — Project Overview

## What It Does

Conditional Diffusion Generation (CDG) for financial time series. Trains a VP-SDE score model using a **dual-axis Transformer** architecture on financial returns, then conditions the reverse diffusion on a user-defined market event using Doob's h-transform. The conditioning variable is a **daily latent macro state** estimated from monthly growth/inflation panels via a tracking regression + joint Kalman filter (new this session — see Latent State Estimation). Portfolio strategies (min-variance, risk-parity, equal-weight) are evaluated on generated vs. real event windows.

---

## Pipeline (top level)

```
latent_state_estimation/macro_importer.py   # step 0 (rare): refresh raw FRED macro panels
explore/import_data.py                      # build dataset — bakes the conditioning series
                                            # (latent state or raw FRED) into column 0
explore/diagnosis.py                        # data/event sanity checks + stationarity
main.py                                     # train diffusion + h-function → ckpt_new/
diffusion_model_analysis/                   # model evaluation scripts
evaluation/                                 # distribution / dependency metrics
```

Everything downstream of `import_data.py` **trusts that the first column of the CSV
(`tickers[0]`) is the chosen conditioning series** — no other script runs the latent
estimation or swaps columns.

---

## Directory Structure

```
CDG_Finance/Code/
├── config/config.py                         # All hyperparameters as dataclasses; paths root-anchored via _ROOT
├── data/data_processor.py                   # Full preprocessing pipeline (note: file contains two DataProcessor class defs — second one is the current version)
├── models/
│   ├── transformer_score.py                 # Dual-axis Transformer score network (primary score model)
│   ├── diffusion_model.py                   # VP-SDE wrapper: train / sample (UNet1D or Transformer)
│   ├── hfunction.py                         # HFunctionCNN + HFunctionTransformer + HFunctionTrainer (legacy, kept for reference)
│   ├── hfunction_direct.py                  # HFunctionTransformerDirect + HFunctionDirectTrainer (one-step BCE)
│   ├── hfunction_twostep.py                 # EllTransformer + EllTrainer + HFunctionTransformerTwoStep + HFunctionTwoStepTrainer (two-step MSE)
│   └── conditional_generator.py             # Doob h-transform guided sampler + Q-model (Transformer-based)
├── utils/
│   ├── helpers.py                           # set_seed
│   └── portfolio.py                         # Portfolio strategies + stats/plots
├── main.py                                  # Full end-to-end pipeline
├── latent_state_estimation/
│   ├── macro_importer.py                    # Downloads FRED macro panels → growth/inflation {macro,daily}.csv
│   ├── tracking_regression.py               # PCA monthly factor + tracking regression → daily tracking portfolio u_t
│   ├── state_space.py                       # StateSpace — vector-form daily-state Kalman filter (see below)
│   ├── macro_main.py                        # LatentStateEstimator class ONLY (no script code)
│   ├── growth_macro.csv / growth_daily.csv  # Raw macro panels (inputs to the estimator)
│   └── inflation_macro.csv / inflation_daily.csv
├── diffusion_model_analysis/
│   ├── unconditional_gen.py                 # Diagnostics table, marginal KDEs
│   ├── conditional_gen.py                   # Conditional vs real event window marginal KDEs + diagnostics table
│   ├── cov.py                               # Correlation/covariance matrix comparison (real vs uncond vs cond)
│   ├── cross_time.py                        # Cross-time OOD generalization check
│   ├── h_function_eval.py                   # H-function calibration check — bypasses sampling/guidance entirely
│   └── losses.py                            # Score + H-function loss/accuracy curves (auto-discovers CSVs)
├── evaluation/
│   ├── distribution_metrics.py
│   └── dependency_metric.py
├── explore/
│   ├── import_data.py                       # Builds macro_data_new.csv + cross_test_data.csv; runs LatentStateEstimator
│   ├── diagnosis.py                         # Event/correlation/stationarity diagnostics → explore/diagnosis_plots/
│   ├── macro_data_new.csv                   # conditioning series (col 0) + AAPL/ORCL/MSFT/IBM log-returns
│   └── cross_test_data.csv                  # cross-period test window dataset
└── ckpt_new/                                # Active checkpoint directory (created by training)
    ├── diffusion_model.pt
    ├── hfunction.pt                         # h-function checkpoint (one-step or two-step, same path)
    ├── ell_function.pt                      # EllTransformer checkpoint (two-step only)
    ├── q_model.pt
    └── score_losses.csv                     # Written by diffusion_model.py after training
```

**Deleted this session (2026-07-17)** — 16 stale root-level files: `example.py`, `generate_data.py`, `Stocks_logret.csv`, `analyze_regime.py`, `train.log`, `setup.py`, root `__init__.py`, `cleanup_wandb.sh`, `PRIVACY.md`, `run_training.sh`, `run_sampling.sh`, `run_sweep_pretrain.sh`, `sample_insample.py`, `sample_outsample.py`, `pretrain_and_plot.py`, `compare_train_test_events.py`. The root sample/analysis scripts were superseded by `diffusion_model_analysis/` + `evaluation/`. All recoverable from git.

---

## Latent State Estimation (`latent_state_estimation/`) — new this session

The conditioning variable is no longer a raw FRED series but a **single daily latent
macro state** estimated from monthly growth and inflation panels. Two stages per
macro variable, then one joint filter across both:

### Stage 1 — `TrackingRegression` (per variable: growth, inflation)
- Monthly factor `z_m` = first principal component of the standardized monthly macro panel
  (sign convention: positive loadings on average).
- Tracking regression: OLS of `z_{m+1}` on `z_m` + monthly-summed daily asset returns
  → betas → **daily tracking portfolio** `u_t = daily_returns @ betas`.

### Stage 2 — `StateSpace` (joint, vector form)
`state_space.py`'s `StateSpace(y, x)` was generalized in place from scalar to vector
form: `y` = DataFrame of **n** monthly factors (growth + inflation), `x` = DataFrame of
**k** daily tracking portfolios. **The state stays 2-dimensional `[s, c]`** — one common
daily latent state `s_t` plus one intramonth cumulator `c_t`:

```
s_t = b0 + b1·s_{t-1} + Σ_j b2_j·x_{j,t-1} + η_t        (daily transition)
c_t = γ_t·c_{t-1} + (same daily increment)               (γ=0 on first day of month)
y_{j,m} = a0_j + a1_j·c_t + ε_j,  ε_j ~ N(0, σ²_j)      (each monthly factor observed
                                                          at month end; NaN months skipped)
```

Parameters `[b0, b1, b2 (k), a0 (n), a1 (n), log σ² (n)]` (10 for k=n=2) fitted by MLE
(prediction-error decomposition, Nelder-Mead). Both monthly factors act as two noisy
sensors of the *same* cumulated latent state, weighted by their fitted signal-to-noise.
Kalman update uses matrix form (`np.linalg.solve` / `slogdet`); with n=k=1 the class
reduces **exactly** to the previous scalar version (verified to 1e-14 against the old
implementation).

Fitted (2026-07-17, full sample): loglik −1471.4, converged; `b1 ≈ 0.924` (persistent,
mean-reverting daily state), `b2_growth ≈ 2.08`, `b2_inflation ≈ −0.55`, both `a1 > 0`.

### `LatentStateEstimator` (`macro_main.py`)
`macro_main.py` is **only** a class — importing it has no side effects.
`LatentStateEstimator(method).fit()` returns the latent state as a daily `pd.Series`
named `"latent"`. Fitted `TrackingRegression`s and the `StateSpace` remain accessible
via `.trackers` / `.state_space` for diagnostics.

`config.data.latent_method` selects the method:

| Value | Meaning |
|---|---|
| `"state_space"` (default) | Joint Kalman filter — one latent state from both indicators/factors |
| `"tracking_regression"` | Standardized average of the two daily tracking portfolios (no Kalman) |
| `None` | No latent state — condition on the raw FRED macro series (`tickers[0]`) |

**The estimator runs in exactly one place: `explore/import_data.py`**, which bakes the
chosen series into the first column of `macro_data_new.csv` at build time. There is no
`latent_states.csv` anymore (deleted) — the series lives only in the dataset CSV.

### Stationarity of the conditioning series (from `diagnosis.py`)
ADF test strongly rejects a unit root (stat −8.96, p ≈ 0.0000): **stationary in mean**
(rolling mean hugs the full-sample mean, no drift). **Not stationary in variance**:
252-day rolling std spikes ~3× in 2008–09 and ~5× in 2020 vs. a baseline of ~1
(volatility clustering in crisis regimes). ACF decays geometrically to ~0 by lag 50,
consistent with the fitted `b1 ≈ 0.92`. Consequence: events (latent upward spikes)
cluster in crisis periods rather than arriving uniformly, so the test window's event
rate depends on which regimes fall in it.

---

## Data

### CSV: `explore/macro_data_new.csv`

Generated by `explore/import_data.py`. **Always regenerated on every run** — the
"found existing dataset, skipping download" logic was removed this session, so
switching `latent_method` (or dates/tickers) takes effect by simply rerunning
`import_data.py`. Contains daily rows with:

| Column | Type | Description |
|---|---|---|
| `T10YFF` (name = `tickers[0]`) | Conditioning series | **Content depends on `latent_method` at build time**: latent state (default) or the raw FRED series. NaN on days with no observation; no interpolation. Column *name* stays `tickers[0]` regardless of content. |
| `AAPL`, `ORCL`, `MSFT`, `IBM` | Log return | Daily stock log-returns from yfinance |

`cross_test_data.csv` (cross-period window, `ct_start_date`–`ct_end_date`) gets the
same conditioning series. Caveat: the latent state only starts 2000-09 (macro panel
coverage), so with a latent method the pre-2000-09 portion of the cross-test window
drops out via `dropna()`.

### Tickers

`config.data.tickers = ["T10YFF", "AAPL", "ORCL", "MSFT", "IBM"]` — 5 channels. `tickers[0]` is always the macro conditioning variable and is **excluded** from `X`/the diffusion channels — `r_dw = self.df[self.tickers[1:]]`, so `X`/`X_train`/`X_test` are stock-returns-only (4 channels) with no macro column at all. `n_assets = len(tickers) - 1` is derived dynamically in `main.py` and used to override `config.diffusion.in_channels`, `config.diffusion.out_channels`, and `config.hfunction.asset_dim`.

**Important:** because the macro series lives only in `self.df`/`self.df_z`, not in `X`, any event-mask logic must read the macro value via `get_z_windows*()` (below) — indexing into `X[:, :, event_asset_idx]` silently reads a stock channel instead of the macro variable. This bug was found in `main.py` etc. on 2026-07-08 and found **again** in `diffusion_model_analysis/cov.py` on 2026-07-17 (see Experiments log).

### Paths — root-anchored (new this session)

`config/config.py` computes `_ROOT` from its own file location and builds `csv_path` /
`ct_csv_path` as absolute paths, so **every script works regardless of the current
working directory** (previously, relative paths broke when running from inside a
subdirectory).

### Preprocessing pipeline (`DataProcessor.process_all`)

1. Load CSV, parse Date index, filter to `[start_date, end_date]` if set
2. ~~Remove weekday effect~~ — **disabled**: `r_dw = self.df[self.tickers[1:]]` (weekday removal is commented out; `self.weekday_mean` stays `None`)
3. **Standardize using train-set stats only** — `mu_seq` and `sigma_seq` computed from train rows only (first `len - test_days` rows, or up to `train_end_date`). Applied as `z = (data - mu) / sigma` to all data. Prevents data leakage.
4. Winsorize (clip at `winsorize_lower` / `winsorize_upper` percentiles; defaults 0.005/0.995)
5. Create rolling `seq_len`-day windows → shape `(N, seq_len, channels)` for event detection, stepped by `config.data.window_shift` (default `1`)
6. Train/test split: by `train_end_date` if set, else last `test_days` rows (default 2000)

`get_diffusion_data()` returns transposed windows `(N, A, T)` from the winsorized data, training portion only. It has its own independent window-slicing loop (separate from `make_sequences()`'s), also stepped by `window_shift`.

**`window_shift`** (`config.data.window_shift`, default `1`): controls the stride between consecutive windows. Threaded through three independently-coded window-scanning loops that all need to agree on it: `make_sequences()`, `get_diffusion_data()`, and `_scan_macro_windows()` (the latter converts a window *index* to a raw row offset via `i = w_idx * window_shift`). `get_z_windows()`'s `n_train_windows` count is likewise `(n_train - seq_len) // window_shift + 1`.

**Z-window extraction (macro `Z_start`/`Z_end` per window)** — several sibling methods, all built around `_macro_std_values_and_n_train()` + `_scan_macro_windows()`. **`macro_window_tolerance` was removed this session** — a window is now valid only if the conditioning series has an actual observation at **both** exact endpoints (no ±w-day search). With the (dense) latent series this keeps ~98% of windows; gaps are holiday NaNs.
- `get_z_windows()` — aligned with `get_diffusion_data()` (has one extra trailing window vs. `X_train`).
- `get_z_windows_train_aligned()` — aligned exactly with `X_train.shape[0]`; `valid_idx` indexes `X_train` directly. Use this (not `get_z_windows()`) whenever the mask must line up with `X_train`.
- `get_z_windows_test()` — same idea, aligned exactly with `X_test.shape[0]`.
- `get_event_threshold_from_percentile(top_fraction, event_type)` — converts a "top X%" fraction into the equivalent raw numeric threshold, computed from **train windows only** (no leakage). The quantile computation branches on `event_type` (see Event Condition section).

All three `get_z_windows*` variants return `(Z_start, Z_end, valid_idx)`, where `valid_idx` filters out windows lacking a conditioning-series observation at an endpoint.

---

## Models

### 1. FinancialTransformerScore (`models/transformer_score.py`)
The primary score network. A dual-axis Transformer conditioned on diffusion time `t` via AdaLN. Stores `self.n_assets` and `self.seq_len` as attributes (used by `ConditionalGenerator._sample_batch`).

- **Input projection:** each scalar return value → `embed_dim` (linear)
- **Positional embeddings:** learnable temporal `(1, 1, T, D)` + per-asset `(1, A, 1, D)`
- **Time conditioning:** Gaussian Fourier features → 2-layer MLP → `cond_dim`
- **`DualAxisBlock`** (N of these):
  1. Temporal self-attention: `(B*A, T, D)` — each asset over the window
  2. Cross-asset self-attention: `(B*T, A, D)` — all assets at each time step
  3. Position-wise FFN (GELU, 4× expansion)
  - Each sub-layer uses `AdaLN`: time embedding → scale + shift
- **Output:** linear projection back to scalar per position → `ScoreOutput.sample` shape `(B, A, T)`

### 2. DiffusionModel (`models/diffusion_model.py`)
- **Architecture:** Transformer (`arch="transformer"`) — config default
- **SDE type:** Variance Preserving (VP): `b_min=0.1`, `b_max=3.25`
- **Loss:** Denoising score matching + covariance penalty (`cov_weight`, applied only to low-t batch examples via `cov_t_max`)
- **Optimizer:** AdamW + `ReduceLROnPlateau` (patience=50, factor=0.5). `weight_decay` is an explicit `train()` param (`config.diffusion.weight_decay`, default 0.0)
- **Loss logging:** saves `ckpt_new/score_losses.csv` (epoch, loss, lr) after training
- **Checkpoint:** `ckpt_new/diffusion_model.pt`

### 3. H-Function Training — controlled by `config.hfunction.one_two_step`

**`"one"` — One-Step BCE (`models/hfunction_direct.py`)** — this is the trainer actually used by `main.py` (`config.hfunction.one_two_step = "one"`)
- **Network:** `HFunctionTransformerDirect` — dual-axis Transformer with AdaLN time conditioning → raw logit (Sigmoid applied outside, or `return_logits=True` for loss computation)
- **Positional embedding:** sinusoidal (`GaussianFourierFeatures` + small MLP) over normalized day index
- **Pooling:** explicit `[h_start, h_end, h_end - h_start]` concatenation (asset-averaged) fed to the head — the label only depends on the window's start/end days
- **Purpose:** Learn `h(t, y) = P(event | Y_t = y)` directly from real `(X, Z)` pairs
- **Forward noising:** `Y_τ = α(τ)·X + σ(τ)·ε`, `τ ~ Uniform[0, h_t_max]`
- **Label (`_compute_labels`):** branches on `event_type` (`abs_change`/`upper_change`/`lower_change`/`absval`; `"sum"` raises `NotImplementedError`)
- **Loss:** `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)`; `pos_weight` persisted in the checkpoint
- **Training loop:** real per-epoch shuffling (`torch.randperm`) with mini-batches
- **Checkpoint:** `ckpt_new/hfunction.pt`
- **Note:** `constraint_mode`/`reward_sharpness` (soft labels) are **not wired into this trainer** — only the legacy `models/hfunction.py` implements the soft branch.

**`"two"` — Two-Step MSE (`models/hfunction_twostep.py`)**

Step 1 — `EllTrainer` trains `EllTransformer`: learns `ℓ_S(x) = P(Z ∈ S | X = x)` from real `(X, B)` pairs; weighted BCE; `ckpt_new/ell_function.pt`.

Step 2 — `HFunctionTwoStepTrainer` trains `HFunctionTransformerTwoStep`: generates synthetic paths from the frozen `DiffusionModel`, labels terminal states with the frozen `EllTransformer`, regresses `h_φ(t, Y_t)` with MSE; `ckpt_new/hfunction.pt`.

Both paths save to `ckpt_new/hfunction.pt` and pass `h_trainer.model` to `ConditionalGenerator`.

### 4. ConditionalGenerator (`models/conditional_generator.py`)
- **Guided reverse SDE:** base drift `g² * score` + guidance `(1 + eta) * g² * (∇H / H)`
- **`h_t_max` cutoff:** guidance is only added when `time_step <= h_t_max` (matches the range `h` was trained on)
- **`pos_weight` param:** constructor accepts it but the correction is **not applied** — guidance uses `h`'s raw output directly (matches the reference implementation)
- **`stop_early_steps`:** stops the reverse SDE this many steps before `t=eps`, leaving residual diversity
- **`_sample_batch`:** derives `n_assets`/`seq_len` dynamically from the score model; calls `.eval()` on all models
- **Q-model:** optional; approximates `∇H / H` to avoid autograd at sampling time; `ckpt_new/q_model.pt`

---

## Event Condition

Configured in `HFunctionConfig`:

| Field | Current Default | Meaning |
|---|---|---|
| `event_type` | `"upper_change"` | `Z_end - Z_start >= threshold` (one-sided: only large upward moves of the conditioning series) |
| `event_asset_idx` | `0` | Watches `tickers[0]` — the conditioning series (latent state), not a stock |
| `event_window` | `10` | Lookback period in days (= `seq_len`) |
| `event_threshold` | `0.10` | **Percentage/quantile semantics** — "top X%" (e.g. `0.10` = top 10% rarest), **not** a raw standardized-units cutoff |
| `h_t_max` | `0.6` | Cap on τ for both training the classifier and applying guidance at sampling time |
| `constraint_mode` | `"hard"` | Hard Doob h-constraint — only implemented in legacy `hfunction.py` |

**Why percentage, not raw standardized units:** `Z_start`/`Z_end` are two points close together in time on a persistent series, so they're highly correlated — `Var(Z_end - Z_start)` is much smaller than 1 and raw-unit thresholds don't correspond to percentile intuition. The fraction is converted **once** to a raw cutoff via `get_event_threshold_from_percentile(top_fraction, event_type)` (train windows only, no leakage). With the current latent-state data: top 10% → **≈ 1.207 std**, giving 421 train / 226 test events.

The conversion + mask-from-macro-series pattern is used by every consumer: `main.py`, `explore/diagnosis.py`, `diffusion_model_analysis/{cov,conditional_gen,h_function_eval}.py`. (The former root-level sample/compare scripts that also did this were deleted this session.)

**Event types:**
- `"abs_change"`: `|Z_end - Z_start| >= threshold`
- `"upper_change"` (default): `Z_end - Z_start >= threshold` — one-sided positive
- `"lower_change"`: `Z_end - Z_start <= -threshold` — one-sided negative (note the negated threshold)
- `"absval"`: `|Z_end| >= threshold`
- `"sum"`: legacy `models/hfunction.py` only

**Quantile computation branches on `event_type`** (fixed 2026-07-14): `abs_change`/`absval` take the quantile of the absolute value; `upper_change` takes `quantile(signed_diffs, 1-top_fraction)`; `lower_change` takes `-quantile(signed_diffs, top_fraction)`.

**Mask/label logic must be macro-series-based, not `X`-based** — masks come from `Z_start`/`Z_end` via `get_z_windows_train_aligned()` / `get_z_windows_test()`, never from `X[:, :, event_asset_idx]` (X has no macro channel). Fixed in main.py & co. on 2026-07-08; the same bug was found and fixed in `cov.py` on 2026-07-17.

---

## Pipeline (main.py)

```
Step 0 (separate): explore/import_data.py — build macro_data_new.csv with the
         conditioning series in column 0 (runs LatentStateEstimator per
         config.data.latent_method; always regenerates)
Step 1: DataProcessor.process_all()
         → weekday removal skipped; standardize with train-only mu/sigma
         → event_threshold: top-X% fraction converted to raw numeric cutoff
           (get_event_threshold_from_percentile), in-place on config.hfunction
         → n_assets derived; overrides config.diffusion.in/out_channels, config.hfunction.asset_dim
Step 2: DiffusionModel.train() → ckpt_new/diffusion_model.pt, ckpt_new/score_losses.csv
Step 3: H-Function Training (controlled by config.hfunction.one_two_step)
         → X_train = get_diffusion_data()[valid_idx]
         → Z_start, Z_end, valid_idx = get_z_windows_train_aligned()
         if "one": HFunctionDirectTrainer — one-step BCE on forward-noised real data (default path)
         if "two": EllTrainer (BCE on real data) → ckpt_new/ell_function.pt
                   HFunctionTwoStepTrainer (MSE on synthetic paths) → ckpt_new/hfunction.pt
Step 4: Extract event masks (train + test) — from Z_start/Z_end (get_z_windows_train_aligned/
         get_z_windows_test), NOT from X (X has no macro channel)
Step 5: ConditionalGenerator.generate() using h_trainer.model as h_model
         → num_samples = config.conditional.n_gen_samples (decoupled from real event
           count, to reduce Monte Carlo noise in the generated-side comparison)
         → generated_samples_train.pt, generated_samples_test.pt
Step 6: PortfolioAnalyzer → results/
```

**CLI flags:** `--skip-diffusion-training`, `--skip-hfunction-training`, `--skip-qmodel-training`, `--skip-conditional`, `--train-q-model`, `--no-wandb`

**Interpreter note:** `/usr/local/bin/python3` (the VSCode default here) now has sklearn installed (added this session); the conda `~/anaconda3/bin/python3` also works. The `(base)` prompt prefix does not apply when invoking `/usr/local/bin/python3` by absolute path.

---

## Config Defaults Summary (current)

```python
# Data
csv_path        = <ROOT>/explore/macro_data_new.csv    # root-anchored absolute path
ct_csv_path     = <ROOT>/explore/cross_test_data.csv
latent_method   = "state_space"    # "state_space" | "tracking_regression" | None
tickers         = ["T10YFF", "AAPL", "ORCL", "MSFT", "IBM"]  # tickers[0] = conditioning series, excluded from X
start_date      = "2000-01-01"
end_date        = "2026-07-08"
ct_start_date   = "1998-01-01"; ct_end_date = "2007-12-31"
window_shift    = 1
seq_len         = 10
test_days       = 2000             # used only when train_end_date is None
winsorize_lower = 0.005; winsorize_upper = 0.995

# Diffusion
in_channels=4, out_channels=4      # overridden dynamically in main.py to n_assets
sample_size=10
arch="transformer"
b_min=0.1, b_max=3.25
embed_dim=128, n_heads=16, n_layers=8, cond_dim=128
n_epochs=750, batch_size=75, lr=1e-4, weight_decay=0.0, num_steps=500
cov_weight=1.0, cov_t_max=0.3

# H-Function (HFunctionDirectTrainer — one_two_step="one" is the active path)
asset_dim=4                        # overridden dynamically in main.py
time_steps=10, embed_dim=64, n_heads=4, n_layers=2, cond_dim=64, dropout=0.0
h_t_max=0.6
event_type="upper_change", event_asset_idx=0, event_window=10
event_threshold=0.10               # top 10% — percentage, converted to raw at startup (≈1.207 std currently)
constraint_mode="hard"             # not wired into hfunction_direct.py; legacy hfunction.py only
one_two_step="one"
n_epochs=50, lr=1e-4, weight_decay=5e-4, scheduler_patience=75

# Conditional Gen
batch_size=32, num_steps=500, stoch=0.5, eta=0
use_q_model=False
stop_early_steps=20
n_gen_samples=2000                 # decoupled from real event count (generated-side MC noise)

# Portfolio
window_for_cov=54, last_days_sum=5
```

---

## Diagnostics (`explore/diagnosis.py`)

Reads the dataset CSV as-is (column 0 = conditioning series) and writes to
`explore/diagnosis_plots/`:

- **`winsorized_standardized_returns.png`** — per-stock standardized return series with test-start marker
- **`acf_squared_residuals.png`** — ACF of squared residuals (volatility clustering)
- **`event_detection.png`** — ΔZ scatter for train/test windows with event threshold line; prints valid-window and event counts
- **`correlation_matrices.png`** — **2×2** last-day-return correlation heatmaps (new layout this session): top-left train-unconditional, top-right train-conditional (event windows), bottom-left test-unconditional, bottom-right test-conditional. Current data: conditioning raises average off-diagonal correlation ~0.45→0.56 in train, directionally consistent in test (n=226, so ±0.13 sampling noise per pair).
- **`conditional_series.png`** — stationarity check for the conditioning series (new this session): level + 252-day rolling mean, rolling std, ACF (120 lags), and ADF test verdict in the title (also printed to console).

Note the **unconditional** correlation baseline shifts between train and test (e.g.
AAPL–MSFT 0.40 train vs 0.67 test — mega-cap era regime drift). When judging
out-of-sample generation, part of any gap is this drift, not conditioning failure.

## Analysis Scripts (`diffusion_model_analysis/`)

All scripts run from the **project root**. All outputs save to `diffusion_model_analysis/results/`.

- **`unconditional_gen.py`** — diagnostics table + marginal KDEs. Generates 2000 unconditional samples.
- **`conditional_gen.py`** — conditional vs real event window KDEs + diagnostics table. Loads pre-generated `.pt` files from root.
- **`cov.py`** — correlation/covariance heatmaps (real all / real events / uncond generated / cond generated). Event mask rebuilt this session (see Experiments 2026-07-17): now sourced from the conditioning series via `get_z_windows_*` with the percentile-converted threshold, matching main.py/diagnosis.py exactly (421 train / 226 test events). "Conditional Generated" panel (needs `generated_samples_*.pt`) remains optional.
- **`h_function_eval.py`** — forward-noises real windows at fixed τ and reports `h_model` output split by true label (calibration check, no sampling).
- **`cross_time.py`** — cross-time OOD generalization check on `ct_csv_path`.
- **`losses.py`** — auto-discovers `ckpt_new/*.csv` loss curves.

**Warning:** the other analysis scripts may still contain the old `X`-based event-mask
pattern that was fixed in `cov.py` this session — audit `h_function_eval.py`,
`conditional_gen.py`, `cross_time.py` before trusting their event splits.

---

## Key Tensor Shapes

| Tensor | Shape | Notes |
|---|---|---|
| `X_train` / `X_test` | `(N, seq_len, A)` | Channels-last; from `make_sequences()` using `df_z`; seq_len=10 |
| Diffusion training data | `(N, A, seq_len)` | Channels-first; from `get_diffusion_data()` using `df_z_wins` |
| `generated_samples_*.pt` | `(N, A, seq_len)` | Output of `ConditionalGenerator` |
| H-function input | `(B, A, seq_len)` | Channels-first (same as diffusion) |

---

## Known Issues / Gotchas

1. **Duplicate `DataProcessor` in `data/data_processor.py`:** Two complete class definitions in one file (first is commented out; active one is the second). Should be cleaned up.

2. **`invert_samples()` doesn't add back `mu_seq`:** Since we standardize as `z = (data - mu) / sigma`, the inverse should be `r = z * sigma + mu`. `invert_samples()` currently only does `r_dw_t = z_seq[t] * sigma_seq`, no `+ mu_seq` term.

3. **`invert_samples()` references `self.weekday_mean`:** weekday removal is disabled, so `self.weekday_mean` stays `None`; calling `invert_samples()` will crash with `AttributeError`.

4. **`constraint_mode`/`reward_sharpness` (soft labels) are not wired into `hfunction_direct.py`** — only the legacy `models/hfunction.py` implements the soft branch. Setting `constraint_mode="soft"` currently has no effect on the active training path.

5. **Conditional generation is fundamentally data-limited at rare event thresholds** (diagnosed 2026-07-08): at top-10% thresholds there are only a few hundred positive train windows from ~a dozen historical episodes. Untried levers: soft labels, training `h` on synthetic diffusion-generated trajectories.

6. **Score model's own baseline under-dispersion:** unconditional generation is somewhat under-dispersed vs. real (~65-79% std ratio) before any conditioning. Accepted for now.

7. **Checkpoints are data-tied:** `ckpt_new/` doesn't currently exist (nothing trained since the latent-state dataset rebuild); the old `checkpoints/` directory holds stale February-era weights that no current code loads. Retrain via `main.py` after any dataset/`latent_method` change.

8. **Config `latent_method` vs. CSV content can drift:** the dataset's conditioning column is whatever `import_data.py` last baked in; scripts *label* plots from the config value. After changing `latent_method`, rerun `import_data.py` to keep them in sync.

9. **Latent state coverage starts 2000-09** — the cross-test window (1998-01-01 start) loses its pre-2000-09 rows when a latent method is active.

---

## Theoretical Framework: Conditioning on an External Event

### Setup

Let `(X, Z) ~ π` where `X ∈ R^d` is the asset return path and `Z` is an external variable (the latent macro state). For an event `S` in the state space of `Z`, the target terminal law is:

```
π^S_X(dx) = P_π(X ∈ dx | Z ∈ S)
```

The reference (pretrained) generative process has law `P_θ` with `Y_T ~ p_θ ≈ p_X`. The dependence between `X` and `Z` enters only through the **event likelihood**:

```
ℓ_S(x) := P_π(Z ∈ S | X = x),    ρ := E_{p_X}[ℓ_S(X)] = P_π(Z ∈ S)
```

Bayes' rule gives: `π^S_X(dx) = (ℓ_S(x) / ρ) p_X(dx)`

### Doob h-Transform

Define the propagated likelihood:

```
h(t, y) := E_{P_θ}[ℓ_S(Y_T) | Y_t = y]
```

`h` solves the backward PDE: `(∂_t + L^θ_t) h = 0`, with terminal condition `h(T, y) = ℓ_S(y)`.

By Itô's formula and Girsanov's theorem, the **conditioned process** is:

```
dY^S_t = [b_θ(t, Y^S_t) + a(t, Y^S_t) ∇_y log h(t, Y^S_t)] dt + σ(t, Y^S_t) dW^S_t
```

This is the Doob h-transform: the score drift is augmented by `a ∇ log h`. The terminal law is `(ℓ_S(x) / ρ_θ) p_θ(dx)`, which equals the target when `p_θ = p_X` and `ℓ_S` is exact.

### Two Approaches to Learning h

#### Approach 1 — Two-Step MSE (`hfunction_twostep.py`)

Sample noisy trajectories from the frozen diffusion model, label terminal states with `ℓ_S(Y_T)`, and regress:

```
ϕ* = argmin_ϕ  (1/T) ∫₀ᵀ E_{Y_{0:T} ~ P_θ} [(h_ϕ(t, Y_t) - ℓ_S(Y_T))²] dt
```

The population minimizer is `h(t, y) = E_{P_θ}[ℓ_S(Y_T) | Y_t = y]`. Propagation is explicitly under `P_θ` — safer under model mismatch.

#### Approach 2 — Direct BCE Classifier (`hfunction_direct.py`)

When paired `(X, Z)` data are available, train a time-dependent classifier directly with binary cross-entropy:

```
ϕ* = argmin_ϕ  E_{(X,Z)~π, τ~Unif[0,T], (Y_t)~P_θ(·|Y_T=X)} [BCE(B, h_ϕ(τ, Y_τ))]
```

where `B = 1{Z ∈ S}`. The population minimizer satisfies:

```
h_{ϕ*}(t, y) = P(Z ∈ S | Y_t = y) = E[ℓ_S(Y_T) | Y_t = y] = h(t, y)
```

Mathematically equivalent to Approach 1 when `p_θ = p_X`; combines the two steps into one. Under model mismatch the two-step MSE is safer.

### Inference (Both Approaches)

Keep `b_θ` frozen and run:

```
dY^ϕ_t = [b_θ(t, Y^ϕ_t) + a(t, Y^ϕ_t) ∇ log(h_{ϕ*}(t, Y^ϕ_t) + δ)] dt + σ(t, Y^ϕ_t) dW_t
```

`δ > 0` is a numerical floor only; the exact Doob transform corresponds to `δ = 0`. In code this maps to `eta * g² * ∇log h` in `ConditionalGenerator`.

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

### 2026-07-17 — Latent-state conditioning pipeline, vector-form Kalman filter, codebase cleanup, cov.py mask bug

**Latent macro state replaces the raw FRED series as the conditioning variable.**
- `latent_state_estimation/state_space.py`: `StateSpace` generalized **in place** from scalar to vector form — `y` can be a DataFrame of n monthly factors and `x` a DataFrame of k daily indicators; `Z` becomes n×2, `F` n×n (`np.linalg.solve`/`slogdet`), missing monthly observations are skipped row-wise. The state stays `[s, c]` (one common latent + intramonth cumulator) — the dimension that grows is observations/inputs, *not* the latent. Verified the n=k=1 case reproduces the old scalar implementation's loglik and filtered states to 1e-14. Params: `[b0, b1, b2 (k), a0 (n), a1 (n), log_var (n)]`; Nelder-Mead MLE (`maxiter=5000, maxfev=10000`).
- Joint fit on growth+inflation (full sample): loglik −1471.4, converged; `b1=0.924`, `b2_growth=2.08`, `b2_inflation=−0.55`, both `a1>0`; growth PCA explained variance 0.683, inflation 0.606.
- `macro_main.py` rewritten as **only** the `LatentStateEstimator` class (methods `"state_space"` = joint KF, `"tracking_regression"` = standardized average of the two daily tracking portfolios). `fit()` returns a daily `pd.Series` named `"latent"`. No CSV output; `latent_states.csv` deleted. Iterated through several architectures (CSV handoff → class called by every consumer → final design) before settling on:
- **Single-injection-point architecture:** `explore/import_data.py` is the only place the estimator runs; it bakes the chosen conditioning series into column 0 of `macro_data_new.csv` (and `cross_test_data.csv`). `config.data.latent_method: Optional[str] = "state_space"` (`"tracking_regression"` | `None` = raw FRED series). All downstream scripts (main.py, diagnosis.py, analysis scripts) consume `tickers[0]` blindly — no estimator imports or column swaps anywhere downstream. Event thresholds rescale automatically since `get_event_threshold_from_percentile` re-derives the cutoff from the data each run (raw FRED: 0.2035 std → latent: 1.2070 std at top-10%).

**Config paths root-anchored:** `_ROOT` computed from `config.py`'s location; `csv_path`/`ct_csv_path` absolute. Fixes a class of `FileNotFoundError`s when running scripts from subdirectories (the session opened with one). `import_data.py`'s cross-test CSV write also switched from a cwd-relative literal to `ct_csv_path`.

**`macro_window_tolerance` removed entirely** (config field, `_scan_macro_windows`'s ±w-day endpoint search, diagnosis.py's mirror loop, the then-unused `get_default_config` import in `data_processor.py`). Windows now require an exact observation at both endpoints; with the dense latent series this keeps ~98% of windows (holiday NaNs account for the rest).

**`import_data.py` always regenerates** — the "existing CSV covers the range → skip download" logic removed. Eliminates the stale-dataset trap when toggling `latent_method`/dates.

**Codebase cleanup — 16 root files deleted** (old synthetic-data era: `example.py`, `generate_data.py`, `Stocks_logret.csv`, `analyze_regime.py`, `train.log`; packaging: `setup.py`, root `__init__.py`; wandb/cluster tooling: `cleanup_wandb.sh`, `PRIVACY.md`, `run_training.sh`, `run_sampling.sh`, `run_sweep_pretrain.sh`; root eval scripts superseded by `diffusion_model_analysis/`+`evaluation/`: `sample_insample.py`, `sample_outsample.py`, `pretrain_and_plot.py`, `compare_train_test_events.py`). Verified nothing living references them.

**diagnosis.py upgrades:**
- Correlation figure now **2×2**: train/test rows × unconditional/conditional (event-window) columns, with event masks from `get_z_windows_train_aligned()`/`get_z_windows_test()`. Finding: conditioning on latent-state spikes raises average off-diagonal correlation ~0.45→0.56 in train, directionally consistent in test; the *unconditional* baseline itself shifts train→test (AAPL–MSFT 0.40→0.67 — regime drift to the mega-cap era), which must be kept in mind when judging out-of-sample generation.
- New **`conditional_series.png`** stationarity figure: level + 252-day rolling mean, rolling std, 120-lag ACF, ADF test in title + console. Finding: latent state is stationary in mean (ADF −8.96, p≈0) but heteroskedastic (rolling std ~3× in 2008-09, ~5× in 2020) with geometric ACF decay matching `b1≈0.92` — events cluster in crisis regimes.

**`cov.py` event-mask bug (the 2026-07-08 bug, resurfaced):** its `get_mask(X)` read `X[:, :, event_asset_idx]` — a *stock* channel (AAPL), since X has no macro column — and compared against the raw fraction `0.1` instead of the percentile-converted cutoff. Result: "Real Train (event windows) n=2120" (~45% of windows) instead of the true 421. Fixed to the main.py pattern (`get_event_threshold_from_percentile` + `get_z_windows_*` + `valid_idx` indexing); also added the missing `start_date`/`end_date`/`train_end_date` to its `DataProcessor` (it was silently using a different data window — 422/229 vs 421/226 until aligned); suptitle now shows the converted threshold and a latent-aware label. Verified exact agreement with diagnosis.py (threshold 1.2070, 421/226). The clarified non-bug: "Conditional Generated (n=2000)" is `n_gen_samples` by design. **The other analysis scripts have not yet been audited for the same X-based-mask pattern.**

**Environment:** installed scikit-learn into `/usr/local/bin/python3` (the VSCode-invoked interpreter), which was missing it; `~/anaconda3/bin/python3` has the full stack.

### 2026-07-14 — Directional event types, window_shift support, threshold-quantile fix, weight_decay exposure

**Event type renamed and extended:**
- `"change"` renamed to `"abs_change"` everywhere; two new one-sided event types added: `"upper_change"` (`Z_end - Z_start >= threshold`) and `"lower_change"` (`Z_end - Z_start <= -threshold`). A sign bug was caught and fixed in every `lower_change` branch (first written as `<= threshold`, true for almost the entire distribution). `config.hfunction.event_type` default changed to `"upper_change"`.

**`get_event_threshold_from_percentile()` quantile computation fixed to depend on `event_type`:** previously always computed the quantile off `|Z_end - Z_start|` — for one-sided types this mixes both tails and selects roughly double the intended fraction. Now branches per type.

**`window_shift` config field added** (default `1`) and threaded through the three independently-coded window-scanning loops (`make_sequences()`, `get_diffusion_data()`, `_scan_macro_windows()`) and every `DataProcessor(...)` call site — previously the field existed but was never passed, so it had zero effect.

**`weight_decay` exposed for `DiffusionModel.train()`** (`config.diffusion.weight_decay`, default 0.0) — previously AdamW's own default (0.01) was silently applied with no way to disable it.

**`cov.py` made resilient to a missing conditional-generation pipeline:** `generated_samples_*.pt` loading optional; subplot grid sizes itself to the actual panel count.

### 2026-07-08 — Event-mask bug fixes, h-function architecture pass, data-scarcity diagnosis, percentage-based event_threshold

**Bug fixes:**
- **Event mask source bug:** masks were read from `X[:, :, event_asset_idx]`, but `X` is stock-returns-only. Fixed by sourcing all masks/labels from `Z_start`/`Z_end` via the new `get_z_windows_train_aligned()`/`get_z_windows_test()`.
- **Off-by-one window indexing:** `get_z_windows()` (aligned with `get_diffusion_data()`) had one more window than `X_train`/`X_test`; added `_sequence_split_idx()` + the aligned variants.
- `b_max`/schedule mismatch between training and sampling reconciled; missing `.eval()` calls added in `ConditionalGenerator._sample_batch()`; stale `h_losses.csv` KeyError fixed by retraining.

**H-function (`hfunction_direct.py`) changes:** sinusoidal positional embedding; `[h_start, h_end, h_end-h_start]` pooling; `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`; real per-epoch shuffling; `h_t_max` cap; `_compute_labels()` branches on `event_type`; dropout tried and reverted (underfitting, not overfitting).

**Root-cause diagnosis — conditional generation over-concentration is data scarcity,** not architecture: at the rare threshold only ~138 positive train windows from ~12 episodes; a top-50% control experiment (~1658 positives) closed most of the std-ratio gap. Untried: soft labels; training `h` on synthetic trajectories.

**`event_threshold` semantics changed to percentage/quantile** ("top X%"), converted once to a raw cutoff from train windows only.

**Sampling-time:** `n_gen_samples` (2000) decoupled from real event count; `stop_early_steps` (20); `pos_weight` guidance correction implemented then removed (matches reference implementation).

### 2026-06-17
- `config`: `start_date`, `ct_csv_path`/`ct_start_date`/`ct_end_date` added; conditioning ticker switched.
- `import_data.py`: cross-time CSV generation added. `cross_time.py`: new OOD script.

### 2026-06-15
- Train-only standardization (no leakage); weekday removal disabled; dynamic shapes in `_sample_batch`; loss CSVs saved to `ckpt_new/`.

### 2026-05-19
**Issue:** Generated std ≈ 2× real std. `corr(score, x)` collapses to ~0 by t=0.59.
**Fix:** Switched to noise parameterization, fixed `adjust = (1+stoch²)/2`.
