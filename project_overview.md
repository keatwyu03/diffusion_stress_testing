# CDG Finance — Project Overview

## What It Does

Conditional Diffusion Generation (CDG) for financial time series. Trains a VP-SDE score model using a **joint spatiotemporal Transformer** (every asset-day token attends to every other in one hop) on financial returns, then conditions the reverse diffusion on a user-defined market event using Doob's h-transform. The conditioning variable is a **daily latent macro state** estimated from monthly growth/inflation panels via a tracking regression or a mixed-frequency Kalman filter (see Latent State Estimation). Returns are standardized **per window** using causal EMA statistics frozen at each window's entry day. Portfolio strategies (min-variance, risk-parity, equal-weight) are evaluated on generated vs. real event windows.

**Current asset universe:** a 10-name tech basket (IBM, CSCO, AAPL, MSFT, ORCL, INTC, TXN, QCOM, AMAT, ADBE), with `seq_len = 10` days and causal event alignment (`event_causal=True`, `event_lag_gap=1`).

---

## Pipeline (top level)

```
latent_state_estimation/macro_importer.py   # step 0 (rare): refresh raw FRED macro panels
explore/import_data.py                      # build dataset — bakes the conditioning series
                                            # (latent state or raw FRED) into column 0
main.py                                     # train diffusion + h-function → ckpt_new/
analysis/                                   # model evaluation scripts (renamed from the
                                            # old diffusion_model_analysis/ + evaluation/ split
                                            # — both merged into one directory)
explore/stationary_diagnosis.py             # macro-conditional stationarity diagnostics
                                            # (replaces the old explore/diagnosis.py, deleted)
explore/bfk_stationarity.py                 # BFK stationarity hypothesis test, consumes
                                            # stationary_diagnosis.py's residual CSV
gan_baseline/cgan_tsgm.py                   # cGAN baseline, trained/evaluated separately
```

Everything downstream of `import_data.py` **trusts that the first column of the CSV
(`tickers[0]`) is the chosen conditioning series** — no other script runs the latent
estimation or swaps columns.

**`diffusion_model_analysis/` and `evaluation/` no longer exist** — both were merged
into a single `analysis/` directory at some point after the sections below were
originally written. **`explore/diagnosis.py` no longer exists either** — it was an
empty file (see the old "Known Issues" list) and has since been deleted outright,
replaced conceptually by `explore/stationary_diagnosis.py`, a different and much more
extensive diagnostic (see its own section below). Any reference to the old paths
elsewhere in this document is describing history, not current structure.

---

## Directory Structure

```
Conditional_diffusion/
├── config/config.py                         # All hyperparameters as dataclasses; paths root-anchored via _ROOT
├── data/data_processor.py                   # Full preprocessing pipeline
├── models/
│   ├── transformer_score.py                 # SpatioTemporalBlock + DualAxisBlock + AdaLN + FinancialTransformerScore
│   ├── diffusion_model.py                   # VP-SDE wrapper: train / sample
│   ├── hfunction_direct.py                  # HFunctionTransformerDirect + HFunctionDirectTrainer (one-step BCE);
│   │                                        # now has EMA-smoothed early stopping — see H-Function Early Stopping
│   ├── hfunction_twostep.py                 # EllTransformer + EllTrainer + HFunctionTransformerTwoStep + HFunctionTwoStepTrainer (two-step MSE)
│   └── conditional_generator.py             # GradientHUNet + ConditionalGenerator — Doob h-transform guided sampler + Q-model
├── utils/
│   ├── helpers.py                           # set_seed + block_interleaved_epoch_order
│   └── portfolio.py                         # PortfolioAnalyzer — strategies + stats/plots
├── main.py                                  # Full end-to-end pipeline; writes ckpt_new/, generated_samples_{train,test}.pt at root
├── latent_state_estimation/
│   ├── macro_importer.py                    # Downloads FRED macro panels → growth/inflation {macro,daily}.csv
│   ├── tracking_regression.py               # PCA monthly factor + tracking regression → daily tracking portfolio u_t
│   ├── state_space.py                       # StateSpace — vector-form daily-state Kalman filter (see below)
│   ├── macro_main.py                        # LatentStateEstimator class ONLY (no script code)
│   ├── growth_macro.csv / growth_daily.csv  # Raw macro panels (inputs to the estimator)
│   └── inflation_macro.csv / inflation_daily.csv
├── analysis/                                 # renamed/merged from the old diffusion_model_analysis/ + evaluation/
│   ├── evaluation_main.py                   # Orchestrator — runs the scripts below as subprocesses, forwards --gan
│   ├── unconditional_gen.py                 # Unconditional diffusion samples vs Real. --gan: no-op (see below)
│   ├── conditional_gen.py                   # Conditional-generation diagnostics (event-conditioned samples vs Real)
│   ├── cov.py                               # Correlation/covariance matrix comparison (real vs uncond vs cond)
│   ├── dependency_metric.py                 # ACF-of-squared-returns dependency test with per-lag p-values
│   ├── distribution_metrics.py              # Wasserstein-distance table, real vs generated marginals
│   ├── h_function_eval.py                   # H-function calibration check — bypasses sampling/guidance entirely.
│   │                                        # Writes to analysis/results/ — a THIRD results dir, distinct from
│   │                                        # diffusion_results/ and gan_results/ below
│   ├── hypothesis_testing.py                # Energy-distance / episode-matched hypothesis tests with bootstrap
│   │                                        # CIs. Standalone — not called by evaluation_main.py; own argparse
│   │                                        # CLI, prints to stdout only, writes no files
│   ├── diffusion_results/                   # Output of the above scripts in diffusion mode (default)
│   └── gan_results/                         # Output of the above scripts under --gan (cGAN comparison plots)
├── gan_baseline/
│   └── cgan_tsgm.py                         # Conditional GAN baseline (TSGM ConditionalGAN). generate_conditional()
│                                            # always requires a binary event label — NO unconditional generation
│                                            # path exists. Writes gan_baseline/gan_results/ (checkpoints, raw
│                                            # samples, losses) when run — not currently present on disk; a run
│                                            # produced analysis/gan_results/'s plots at some point but the raw
│                                            # gan_baseline/gan_results/ outputs are gitignored and gone
├── explore/
│   ├── import_data.py                       # Builds macro_data_new.csv; runs LatentStateEstimator; also writes
│   │                                        # bucket_corr_matrices.png, conditioning_series.png
│   ├── stationary_diagnosis.py              # Macro-conditional stationarity diagnostics — see its own section.
│   │                                        # Imported by bfk_stationarity.py (not standalone-only)
│   ├── bfk_stationarity.py                  # BFK stationarity hypothesis test (rpy2 → R's npcp package).
│   │                                        # CLI: run-asset / combine subcommands. Writes bfk_results/<TICKER>.json
│   ├── bfk_array_job.sh                     # SLURM array job, one task per ticker, calls bfk_stationarity.py run-asset
│   ├── submit_bfk_jobs.sh                   # Clears old bfk_results/*.json then sbatch's the array job
│   ├── bfk_results/                         # Per-ticker JSON results (10 tickers) + logs/
│   ├── macro_data_new.csv                   # conditioning series (col 0) + 10-ticker log-returns
│   └── nn_standardized_macro.csv            # Output of stationary_diagnosis.py's save_standardized_residuals() —
│                                            # the residual CSV bfk_stationarity.py consumes
└── ckpt_new/                                # Active checkpoint directory (created by training)
    ├── diffusion_model.pt
    ├── hfunction.pt                         # h-function checkpoint (one-step or two-step, same path)
    ├── h_losses.csv                         # H-function training loss log (epoch, loss, lr, etc.)
    ├── score_losses.csv                     # Score-network training loss log
    ├── ell_function.pt                      # EllTransformer checkpoint (two-step only)
    └── q_model.pt                           # Q-model checkpoint (only if --train-q-model was used)
```

**`explore/test.py` does not exist.** An earlier, unintegrated Dwivedi–Subba Rao (DFT/trispectrum) portmanteau stationarity test was built and debugged as a standalone file at this path, but it is gone from the current repo state — deleted at some point after being set aside as unvalidated. If you're looking for a DFT-based stationarity test, it isn't there; **the stationarity test actually in use is the BFK test** (`explore/bfk_stationarity.py`), a different method entirely (Bücher–Fermanian–Kojadinov, via R's `npcp` package).

**Deleted 2026-07-17** — 16 stale root-level files from the old synthetic-data/packaging/cluster-tooling era (`example.py`, `generate_data.py`, `Stocks_logret.csv`, `analyze_regime.py`, `train.log`, `setup.py`, root `__init__.py`, `cleanup_wandb.sh`, `PRIVACY.md`, `run_training.sh`, `run_sampling.sh`, `run_sweep_pretrain.sh`, `sample_insample.py`, `sample_outsample.py`, `pretrain_and_plot.py`, `compare_train_test_events.py`). The root sample/analysis scripts were superseded by what was then `diffusion_model_analysis/` + `evaluation/` (now merged into `analysis/`, see above). All recoverable from git.

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
"found existing dataset, skipping download" logic was removed, so switching
`latent_method` (or dates/tickers) takes effect by simply rerunning
`import_data.py`. Contains daily rows with:

| Column | Type | Description |
|---|---|---|
| `cond` (column 0) | Conditioning series | **Content depends on `latent_method` at build time**: latent state (default) or the raw FRED series. NaN on days with no observation; no interpolation. |
| `IBM`, `CSCO`, `AAPL`, `MSFT`, `ORCL`, `INTC`, `TXN`, `QCOM`, `AMAT`, `ADBE` | Log return | Daily stock log-returns from yfinance |

### Tickers and the conditioning column (**changed** — conditioning is no longer `tickers[0]`)

`config.data.tickers` now holds **asset tickers only** — currently the 10-name tech
basket `["IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "TXN", "QCOM", "AMAT", "ADBE"]`.
The conditioning series is a **separate column**, resolved **positionally** by
`DataProcessor.load_returns()` as `self.macro_col = df.columns[0]` (the first column
after `Date`). `import_data.py` writes it under the name `cond`; the name only has to
avoid clashing with a ticker.

Consequences:
- `n_assets = len(config.data.tickers)` (no `- 1`) — see `main.py`, which uses it to
  override `config.diffusion.in_channels`, `config.diffusion.out_channels`, and
  `config.hfunction.asset_dim`.
- `load_returns()` selects `df[[macro_col] + tickers]` and drops rows missing any
  *asset* (`dropna(subset=self.tickers)`) — the macro series is allowed to be sparse.

**Important (unchanged):** the macro series lives only in `self.df`, never in `X`, so
any event-mask logic must read it via `get_z_windows*()`. Indexing into
`X[:, :, event_asset_idx]` reads a stock channel instead. This bug was found in
`main.py` on 2026-07-08 and again in `cov.py` (then in `diffusion_model_analysis/`,
now `analysis/`) on 2026-07-17.

### Paths — root-anchored (new this session)

`config/config.py` computes `_ROOT` from its own file location and builds `csv_path` /
`ct_csv_path` as absolute paths, so **every script works regardless of the current
working directory** (previously, relative paths broke when running from inside a
subdirectory).

### Preprocessing pipeline (`DataProcessor.process_all`) — **rewritten: per-window causal EMA standardization**

Global train-set `mu_seq`/`sigma_seq` standardization is **gone**. The order is now
winsorize-then-standardize (previously the reverse), and standardization is per-window
using causal EMA stats frozen at each window's entry day:

1. **Load CSV** (`load_returns`) — parse Date index, resolve `macro_col` positionally,
   filter to `[start_date, end_date]`, drop rows missing any asset.
2. **Winsorize RAW returns** (`_winsorize_raw_returns`) — clip each stock's raw
   log-return to its `[winsorize_lower, winsorize_upper]` quantiles, computed from
   **train rows only**, applied to all rows. Runs **before** the EMA so the vol
   estimate itself never sees the outliers.
3. **Causal EMA standardizer** (`_compute_ema_stats`) — per-stock
   `MU = r.ewm(span=ema_span, min_periods=20).mean().shift(1)` and the matching `SG`
   for the std. The `.shift(1)` means row `t`'s stats use data through `t-1` only.
   Rows before EMA warm-up are trimmed from `self.df` so windows and the macro series
   keep sharing one row base.
4. **`standardize()`** — per-row EMA z-scores into `self.df_z`. **Diagnostics only**
   (`diagnosis.py`); the model tensors do not use this frame.
5. **`make_sequences()`** — each window is standardized by the EMA mean/vol at its
   **entry day**, `z = (r - MU[s]) / SG[s]`, one vector per stock broadcast across the
   window's days. Stepped by `window_shift`. Also records `start_weekdays`, and
   `mu_entry`/`sig_entry` `(Nw, A)` per window for invertibility.
6. **Train/test split** — by `train_end_date` if set, else last `test_days` rows
   (config default `test_days=1200`, `train_end_date=None`). Also splits
   `start_weekdays`, `mu_entry`, `sig_entry`.

`get_diffusion_data()` returns transposed windows `(N, A, T)` with the **same**
entry-day EMA standardization, training portion only, and stores
`self.diffusion_end_dates_train` 1:1 aligned with the returned windows (consumed by
block sampling).

**New config field:** `ema_span: int = 60` (constructor also takes `ema_min_periods=20`).

### De-standardization (**replaces `invert_samples`**)

`invert_samples()` was **removed** along with global-sigma/weekday inversion —
`sigma_seq`/`mu_seq`/`weekday_mean` are no longer populated at all. Replacements:

- **`destandardize_windows(z, mu_entry, sig_entry)`** — inverts per-window EMA
  standardization back to raw log-returns: `z * sig_entry[:, :, None] + mu_entry[:, :, None]`.
- **`sample_entry_stats(n_draw, split, mask, seed)`** — draws `(mu_entry, sig_entry)`
  pairs with replacement from a split's entry-stat pool, to give generated windows
  realistic vol scales. Pass the **event mask** for conditional samples (events cluster
  in particular vol regimes; drawing from the full pool erases the macro-event ↔ entry
  volatility relationship). Unconditional samples use the full pool.

**Note:** `utils/portfolio.py` still calls the removed `data_processor.invert_samples(...)`
at `analyze_samples()` and `analyze_test_set()`, so `main.py` Step 6 raises
`AttributeError` until it is ported to `destandardize_windows` + `sample_entry_stats`.

**`window_shift`** (`config.data.window_shift`, default `1`): controls the stride between consecutive windows. Threaded through three independently-coded window-scanning loops that all need to agree on it: `make_sequences()`, `get_diffusion_data()`, and `_scan_macro_windows()` (the latter converts a window *index* to a raw row offset via `i = w_idx * window_shift`). `get_z_windows()`'s `n_train_windows` count is likewise `(n_train - seq_len) // window_shift + 1`.

**Causal event alignment (`event_causal` / `event_lag_gap`)** — new `DataConfig` fields
(`event_causal: bool = True`, `event_lag_gap: int = 1`) controlling *which* macro days
`_scan_macro_windows()` reads for a given return window starting at raw row `i`:

| Mode | `Z_start` / `Z_end` read from | Meaning |
|---|---|---|
| `event_causal=False` (original) | `macro[i]` and `macro[i + seq_len - 1]` | the **same** days as the return window — `Z_end` can depend on the last day of the window being generated |
| `event_causal=True` (**current default**) | the `seq_len`-day macro window ending at `i - event_lag_gap - 1` | event is **fully known before** the return window starts; `event_lag_gap=0` means the macro window ends the day immediately before |

Causal mode drops windows whose implied `start_row < 0`. Threaded through every
`DataProcessor(...)` call site (`main.py`, `import_data.py`, `analysis/*` —
formerly split across `evaluation/*` and `diffusion_model_analysis/*`, since
merged, see Directory Structure above).

**Z-window extraction (macro `Z_start`/`Z_end` per window)** — several sibling methods, all built around `_macro_std_values_and_n_train()` + `_scan_macro_windows()`. **`macro_window_tolerance` was removed** — a window is now valid only if the conditioning series has an actual observation at **both** exact endpoints (no ±w-day search). With the (dense) latent series this keeps ~98% of windows; gaps are holiday NaNs.
- `get_z_windows()` — aligned with `get_diffusion_data()` (has one extra trailing window vs. `X_train`).
- `get_z_windows_train_aligned()` — aligned exactly with `X_train.shape[0]`; `valid_idx` indexes `X_train` directly. Use this (not `get_z_windows()`) whenever the mask must line up with `X_train`.
- `get_z_windows_test()` — same idea, aligned exactly with `X_test.shape[0]`.
- `get_event_threshold_from_percentile(top_fraction, event_type)` — converts a "top X%" fraction into the equivalent raw numeric threshold, computed from **train windows only** (no leakage). The quantile computation branches on `event_type` (see Event Condition section).

All three `get_z_windows*` variants return `(Z_start, Z_end, valid_idx)`, where `valid_idx` filters out windows lacking a conditioning-series observation at an endpoint.

---

## Models

### 1. FinancialTransformerScore (`models/transformer_score.py`) — **now joint spatiotemporal attention, not dual-axis**

The primary score network. Stores `self.n_assets` / `self.seq_len` (used by
`ConditionalGenerator._sample_batch`).

- **Input projection:** each scalar return value → `embed_dim` (linear)
- **Positional embeddings (changed):** `nn.Embedding` **lookup tables** — `day_emb(seq_len, D)`
  + `stock_emb(n_assets, D)`, with `day_ids`/`stock_ids` registered as non-persistent
  buffers. (Previously learnable `nn.Parameter` tensors.) Same tokenization style as
  `HFunctionTransformerDirect`.
- **Time conditioning:** Gaussian Fourier features → 2-layer MLP → `cond_dim`
- **Block type (changed):** `FinancialTransformerScore` now stacks **`SpatioTemporalBlock`**,
  not `DualAxisBlock`. Tokens are flattened day-major to `(B, T*A, D)` and every
  (asset, day) token attends to every other token **in a single hop** — "asset i on day 3
  → asset j on day 9" no longer needs two hops. At `A*T` tokens (currently 10×10=100)
  the full attention matrix is tiny, so the factorization bought nothing.
  `DualAxisBlock` still exists and is used by `HFunctionTransformerTwoStep`.
- **Output:** `LayerNorm` → linear → `ScoreOutput.sample` shape `(B, A, T)`

### 2. DiffusionModel (`models/diffusion_model.py`)
- **Architecture:** Transformer (`arch="transformer"`) — config default
- **SDE type:** Variance Preserving (VP). **Config now `b_min=0.1`, `b_max=10.0`** (was 3.25)
- **Parameterization (documented here for the first time):** the network predicts **eps**,
  not the score. `score_from_eps(eps_hat, t) = -eps_hat / sigma(t)` is the single
  conversion point every consumer (loss, reverse-SDE drift, Doob guidance) goes through.
  eps has unit variance at every `t`, unlike the score, which diverges as `t → 0`.
- **Loss:** `sum((eps_hat - z)^2)` + optional correlation penalty. The penalty compares
  the batch's Tweedie one-step reconstruction `x0_hat` last-day correlation against
  `self.real_corr_target` (computed once in `train()` from the training windows'
  last-day returns), off-diagonal only, and applies only to examples with
  `t < cov_t_max`. **`cov_weight` is currently `0.0`, so the penalty is inactive.**
- **Block sampling (new):** `block_sampling=True` (config default) replaces DataLoader's
  flat shuffle with `utils.block_interleaved_epoch_order(end_dates)` — each epoch draws
  minibatches round-robin across calendar-month blocks, so a batch spans many months
  instead of a locally-shuffled run of overlapping windows. Requires `end_dates`
  (`data_processor.diffusion_end_dates_train`); raises `ValueError` without it.
- **Optimizer:** AdamW + `ReduceLROnPlateau` (patience=50, factor=0.5), `weight_decay` from config (default 0.0)
- **Loss logging:** `ckpt_new/score_losses.csv` (epoch, loss, lr)
- **Checkpoint:** `ckpt_new/diffusion_model.pt`

### 3. H-Function Training — controlled by `config.hfunction.one_two_step`

**`"one"` — One-Step BCE (`models/hfunction_direct.py`)** — this is the trainer actually used by `main.py` (`config.hfunction.one_two_step = "one"`)
- **Network:** `HFunctionTransformerDirect` — **`SpatioTemporalBlock`** stack (joint
  attention over all (asset, day) tokens, same block as the score net) with AdaLN time
  conditioning → raw logit. `return_logits=True` feeds `BCEWithLogitsLoss` directly;
  default returns `sigmoid(logits)` for sampling/guidance call sites.
- **Positional embedding:** `day_emb`/`stock_emb` `nn.Embedding` tables (matching the score net)
- **Pooling (changed — now 4 channels, `4*embed_dim` head input):**
  `[h_mean, h_start, h_end, h_end - h_start]`. The `h_mean` channel (global window
  average) was added to summarize window-wide structure — vol level, co-movement —
  alongside the start/end channels that align with the event's change definition.
- **Purpose:** Learn `h(t, y) = P(event | Y_t = y)` directly from real `(X, Z)` pairs
- **Forward noising:** `Y_τ = α(τ)·X + σ(τ)·ε`, `τ ~ Uniform[0, h_t_max]`
- **Label (`_compute_labels`):** branches on `event_type`; **now supports soft labels**
  (see Event Condition below)
- **Loss:** `BCEWithLogitsLoss(pos_weight = n_neg/n_pos, reduction="none")`;
  `pos_weight` persisted in the checkpoint alongside the state dict
  (`{"state_dict":..., "pos_weight":...}`, with backward-compat loading of plain
  state dicts as `pos_weight=1.0`)
- **Block sampling (new):** `cfg.block_sampling=True` uses
  `block_interleaved_epoch_order(end_dates)` instead of `torch.randperm`; requires
  `end_dates` (passed from `main.py` as `diffusion_end_dates_train[valid_idx]`)
- **Episode reweighting (new, `cfg.episode_reweight`, currently `False`):**
  `_episode_weights()` finds maximal runs of consecutive-**date** positive windows (a
  date gap breaks a run even if array indices are adjacent) and weights each positive
  window by `1/sqrt(m_j)` where `m_j` is its episode length; negatives get 1.0. This
  stops the BCE being dominated by however many overlapping windows a single persistent
  macro event happened to span. Stacks multiplicatively with `pos_weight`, giving an
  effective positive weight of `pos_weight / sqrt(m_j)`.
- **Gradient clipping:** `clip_grad_norm_(max_norm=1.0)`
- **Checkpoint:** `ckpt_new/hfunction.pt`
- **`constraint_mode`/`reward_sharpness` (soft labels) ARE now wired into this trainer**
  — the earlier "legacy `hfunction.py` only" note no longer applies.
- **EMA-smoothed early stopping (new):** with only ~253 positive events out of
  ~5,057 windows at the current `event_threshold`, a held-out validation split
  for early stopping was rejected — carving one out would starve both sides and
  the validation loss itself would be too noisy at that sample size to trust.
  Instead, training loss is watched directly, but SMOOTHED (raw per-epoch loss
  on an observed run had epoch-to-epoch std ~0.037 with ~49% of epochs going UP
  vs. the previous one — too noisy for a direct "did this beat the record"
  check). An EMA of the loss (`early_stop_ema_span=20`) is compared against the
  best EMA seen so far; if it hasn't improved by more than
  `early_stop_min_delta` for `early_stop_patience` consecutive epochs, training
  stops and the best-EMA checkpoint is restored. `early_stop_patience` (100) is
  kept larger than `scheduler_patience` (75) so `ReduceLROnPlateau`'s LR decay
  gets a chance to revive progress before early stopping gives up entirely.
  `early_stop_min_delta` was tuned against an actual 425-epoch loss trace: the
  original `5e-3` was too permissive (loss was still declining by ~0.01–0.03
  per 25-epoch window even near the end of that run, so the patience clock kept
  resetting on genuine-but-small progress and never fired); raised to `2e-2`,
  which still lets real decline through without treating any downward drift as
  a new record. `n_epochs` (1000) is now just the upper-bound safety cap, not
  the primary stopping signal.

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
| `event_window` | `10` | Lookback period in days (= `seq_len`) |
| `event_threshold` | `0.075` | **Percentage/quantile semantics** — "top X%" (top 7.5%), **not** a raw standardized-units cutoff |
| `h_t_max` | `0.9` | Cap on τ for both training the classifier and applying guidance at sampling time |
| `constraint_mode` | `"hard"` | `"hard"` = 0/1 labels at the threshold; `"soft"` = graded sigmoid labels — **both implemented in `hfunction_direct.py`** |
| `reward_sharpness` | `5.0` | Sigmoid steepness in soft mode |

**`event_asset_idx` no longer exists** — it was removed along with the
`tickers[0]`-is-the-macro-column design. The conditioning series is `macro_col`
(CSV column 0) and is reached only via `get_z_windows*()`.

**Soft labels (`constraint_mode="soft"`)** — `_compute_labels()` returns
`sigmoid(reward_sharpness * (metric - threshold))` for every window instead of a 0/1
edge, so each window carries graded signal about how event-like it is rather than the
signal coming only from the rare positives. `label >= 0.5` recovers the hard condition
exactly (the sigmoid is centered at the threshold), which is how evaluation code
(`h_function_eval.py`) recovers hard labels from soft ones. Metric/threshold are in
standardized units (threshold ~1.2–1.5 std), so `sharpness=5` spreads the ramp over a
meaningful band; ~50 would be indistinguishable from hard labels.

**Evaluation always uses the HARD event definition** regardless of `constraint_mode` —
soft only changes the h-function's training labels, not what counts as an event.

**Why percentage, not raw standardized units:** `Z_start`/`Z_end` are two points close together in time on a persistent series, so they're highly correlated — `Var(Z_end - Z_start)` is much smaller than 1 and raw-unit thresholds don't correspond to percentile intuition. The fraction is converted **once** to a raw cutoff via `get_event_threshold_from_percentile(top_fraction, event_type)` (train windows only, no leakage). **Current config: `event_threshold=0.05` (top 5%) → threshold ≈ 0.765 std, giving 253 positive events out of 5,057 train windows (5.00%, measured 2026-08-15).** (An older run of this doc reported top 10% → ≈1.207 std, 421/226 events — that was under a different `event_threshold` value; re-measure rather than trust either number blindly, since this is exactly the kind of value that drifts as the config changes.)

The conversion + mask-from-macro-series pattern is used by every consumer: `main.py`, `analysis/{cov,conditional_gen,h_function_eval}.py` (formerly split across the now-deleted `explore/diagnosis.py` and `diffusion_model_analysis/`).

**Event types:**
- `"abs_change"`: `|Z_end - Z_start| >= threshold`
- `"upper_change"` (default): `Z_end - Z_start >= threshold` — one-sided positive
- `"lower_change"`: `Z_end - Z_start <= -threshold` — one-sided negative (note the negated threshold)
- `"absval"`: `|Z_end| >= threshold`
- `"start_upper"` (**new**): `Z_start >= threshold` — depends only on the window's
  *starting* macro value, so it is known at window entry, unlike every other type
  (which all depend on `Z_end`). Quantile taken as `quantile(Z_start, 1 - top_fraction)`.
- `"sum"`: legacy `models/hfunction.py` only

All five are implemented consistently across `_compute_labels()`,
`get_event_threshold_from_percentile()`, `main.py`'s mask blocks, and every analysis
script's `event_mask()`.

**Quantile computation branches on `event_type`** (fixed 2026-07-14): `abs_change`/`absval` take the quantile of the absolute value; `upper_change` takes `quantile(signed_diffs, 1-top_fraction)`; `lower_change` takes `-quantile(signed_diffs, top_fraction)`.

**Mask/label logic must be macro-series-based, not `X`-based** — masks come from `Z_start`/`Z_end` via `get_z_windows_train_aligned()` / `get_z_windows_test()`, never from `X[:, :, event_asset_idx]` (X has no macro channel). Fixed in main.py & co. on 2026-07-08; the same bug was found and fixed in `cov.py` on 2026-07-17.

---

## Pipeline (main.py)

```
Step 0 (separate): explore/import_data.py — build macro_data_new.csv with the
         conditioning series in column 0 (runs LatentStateEstimator per
         config.data.latent_method; always regenerates)
Step 1: DataProcessor.process_all()
         → winsorize raw returns (train-only quantiles) → causal EMA stats →
           per-window entry-day standardization
         → event_threshold: top-X% fraction converted to raw numeric cutoff
           (get_event_threshold_from_percentile), in-place on config.hfunction
         → n_assets = len(tickers); overrides config.diffusion.in/out_channels,
           config.hfunction.asset_dim
Step 2: DiffusionModel.train() → ckpt_new/diffusion_model.pt, ckpt_new/score_losses.csv
         → block_sampling=True uses diffusion_end_dates_train for month-block interleaving
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
latent_method   = "state_space"    # "state_space" | "tracking_regression" | None
growth_vars     = None             # growth group DROPPED from the latent estimate
inflation_vars  = ["cpi"]          # inflation group: CPI only
tickers         = ["IBM","CSCO","AAPL","MSFT","ORCL","INTC","TXN","QCOM","AMAT","ADBE"]
                                   # assets ONLY — conditioning series is CSV column 0 ("cond")
start_date      = "2000-01-01"
end_date        = "2026-07-08"
window_shift    = 1
seq_len         = 10
event_causal    = True             # event known before the return window starts
event_lag_gap   = 1
test_days       = 1200             # used only when train_end_date is None
train_end_date  = None
winsorize_lower = 0.005; winsorize_upper = 0.995
ema_span        = 60               # per-window causal EMA standardizer span

# Diffusion
in_channels=4, out_channels=4      # overridden dynamically in main.py to n_assets (=10)
sample_size=10
arch="transformer"
b_min=0.1, b_max=10.0
embed_dim=256, n_heads=16, n_layers=8, cond_dim=256
n_epochs=20, batch_size=75, lr=1e-4, weight_decay=0.0, num_steps=500
block_sampling=True
cov_weight=0.0, cov_t_max=0.3      # cov penalty currently INACTIVE (weight 0)

# H-Function (HFunctionDirectTrainer — one_two_step="one" is the active path)
asset_dim=4                        # overridden dynamically in main.py (=10)
time_steps=10, embed_dim=256, n_heads=8, n_layers=6, cond_dim=256, dropout=0.0
h_t_max=0.9
train_batch_size=126, train_stoch=0.5, h_mini_batch_size=256
block_sampling=True, episode_reweight=False
event_type="upper_change", event_window=10
event_threshold=0.05               # top 5% — converted to a raw cutoff at startup
                                    # (253 positive events / 5057 train windows, measured this session)
constraint_mode="hard"             # "soft" IS implemented here now
reward_sharpness=5.0
one_two_step="one"
n_epochs=1000, lr=1e-4, weight_decay=5e-4              # n_epochs is now an upper bound, see early stopping below
scheduler_patience=75, scheduler_factor=0.5
early_stop_patience=100, early_stop_min_delta=2e-2, early_stop_ema_span=20  # see H-Function Training section

# Conditional Gen
batch_size=32, num_steps=500, stoch=1.0, eta=1
use_q_model=False
stop_early_steps=5
n_gen_samples=2000                 # decoupled from real event count (generated-side MC noise)
q_embed_dim=64, q_n_heads=4, q_n_layers=4, q_cond_dim=64
q_model_epochs=500, q_model_lr=1e-4, q_model_train_batch_size=2**12

# Portfolio
window_for_cov=54, last_days_sum=5
portfolio_tickers = same 10-name basket as data.tickers
```

---

## Data-Level Diagnostics — historical (`explore/diagnosis.py`, DELETED)

**`explore/diagnosis.py` no longer exists.** The description below is kept for
history only — it describes plots that a past version of the file produced. The
file was tracked as an empty file for a while (see the old "Known Issues" list)
and has since been removed outright. If you need any of this functionality, it
does not currently exist anywhere in the repo; `stationary_diagnosis.py` (next
section) covers different ground (conditional stationarity of the *model's
standardized residuals*, not the raw conditioning series or event/correlation
sanity checks).

Historically wrote to `explore/diagnosis_plots/`:
- `winsorized_standardized_returns.png` — per-stock standardized return series with test-start marker
- `acf_squared_residuals.png` — ACF of squared residuals (volatility clustering)
- `event_detection.png` — ΔZ scatter for train/test windows with event threshold line
- `correlation_matrices.png` — 2×2 last-day-return correlation heatmaps (train/test × unconditional/conditional)
- `conditional_series.png` — stationarity check for the raw conditioning series (rolling mean/std, ACF, ADF test)

## Macro-Conditional Stationarity Diagnostics (`explore/stationary_diagnosis.py`)

Tests whether a **learned, macro-state-conditional location-scale
standardization** removes the nonstationarity in each ticker's returns —
distinct from (and more targeted than) the old `diagnosis.py`'s stationarity
check on the raw conditioning series itself.

**Model (`NLL_Stationarity`):** two small MLPs, `mu_net(m)` and `sigma_net(m)`
(`m` = the macro conditioning state), jointly fit by minimizing the Gaussian
negative log-likelihood of `r ~ N(mu(m), sigma(m)^2)` on the FULL series (no
train/test split for the headline result — the question is representability of
the macro relationship, not forecasting; see the file's own docstrings for the
reasoning, worked out interactively this session). `sigma` uses
`softplus` + a `delta` floor to stay positive and away from the NLL's `log(sigma)`
singularity. `z = (r - mu(m)) / sigma(m)` is the standardized residual under test.

**Two diagnostics per ticker, both run on the full-series model's `z`:**
- **`macro_bins`** — quantile-bins `m` into `n_bins` (default 10), reports
  `E[z | m in bin]` and `Var[z | m in bin]` per bin plus their mean/sd across
  bins. Tests whether the conditional moments are genuinely flat across the
  macro state, not just unconditionally standardized — the stronger, more
  meaningful claim (a plain global z-score trivially passes the unconditional
  version of this test; it says nothing about macro-conditional structure).
- **`time_blocks`** — splits `z` into `n_blocks` (default 10) **consecutive**
  time blocks (never scattered like the macro bins, since autocovariance needs
  temporally adjacent points), computes the lag-1..`max_lag` autocovariance
  within each block, and reports the mean/sd of each lag across blocks — tests
  whether the autocovariance structure drifts over calendar time.

**Validation methodology worked out this session, worth knowing before trusting
a result from this file:** an early "does macro conditioning matter" check used
a permutation control that shuffled `m` and then binned the shuffled model's
residuals **by the shuffled `m`** — this is a null test of "are residuals
similar across random groups," which is trivially true, and gave a false
negative (looked like real macro conditioning was no better than shuffled). The
corrected control evaluates BOTH the real-macro and shuffled-macro models in
bins of the **real** `m` — only then does the real model's advantage show up
(e.g., the bin with the most extreme macro state had raw variance 2.20, fell to
2.05 under a shuffled-macro model, and to 0.81 under the real-macro model).
Network capacity was also tested and found **not** to matter here — a 13-parameter
and a 4,353-parameter version gave nearly identical results, so the earlier
concern that a large network could trivially memorize its way to flat bins was
not the actual confound.

**`save_standardized_residuals()`** writes the full-series `z` for every ticker
to `explore/nn_standardized_macro.csv` (Date + one column per ticker) — this is
the file `bfk_stationarity.py` (next section) consumes, so it must be run first.

Run: `python explore/stationary_diagnosis.py` (add `--print-split-results` for
an optional secondary 70/30 train/test view — not the headline result, see
above).

**A separate, earlier attempt at a DFT/trispectrum-based stationarity test
(Dwivedi–Subba Rao portmanteau) was built, debugged, and set aside as a
standalone file (`explore/test.py`) — that file no longer exists in the repo.**
Real bugs were found and fixed during that work (trispectrum scaling off by a
factor of T², pairing-term subtraction needing to happen *before* smoothing
rather than after, bandwidth instability needing to scale with T), but the
estimator's finite-sample behavior was never fully validated, and it was never
wired into anything. If this test is wanted, it would need to be rebuilt from
scratch.

## BFK Stationarity Test (`explore/bfk_stationarity.py`)

The stationarity hypothesis test actually in use — **Bücher–Fermanian–Kojadinov**
(not Dwivedi–Subba Rao; see above), run via R's `npcp` package through `rpy2`.
Consumes `explore/nn_standardized_macro.csv` (written by
`stationary_diagnosis.py`, above), one ticker column at a time.

CLI: `run-asset <TICKER>` (single ticker) / `combine` (aggregate results) subcommands.

**Cluster submission:**
- `submit_bfk_jobs.sh` — clears old `bfk_results/*.json`, then `sbatch`'s the array job
- `bfk_array_job.sh` — SLURM array job (`--array=0-9%2`), one task per ticker across
  the 10-name basket, each calling `bfk_stationarity.py run-asset <TICKER>`; logs to
  `explore/bfk_results/logs/`

**Results:** `explore/bfk_results/<TICKER>.json`, one per ticker — all 10 present as
of the last run. Multiple job IDs' worth of logs are present in `bfk_results/logs/`,
i.e. this has been run more than once across the session(s) that produced it.

## Analysis Scripts (`analysis/`)

All scripts run from the **project root**. `evaluation_main.py` orchestrates the
rest as subprocesses and forwards `--gan` uniformly; each script can also be run
standalone. Default (no `--gan`): diffusion-model outputs to
`analysis/diffusion_results/`. With `--gan`: cGAN-baseline outputs to
`analysis/gan_results/`.

- **`evaluation_main.py`** — runs `losses.py`, `unconditional_gen.py`,
  `conditional_gen.py`, `cov.py`, `dependency_metric.py`, `distribution_metrics.py`
  in sequence. Writes nothing itself.
- **`losses.py`** — training-loss curves. Default: score/H-function loss CSVs from
  `ckpt_new/`. `--gan`: `gan_baseline/gan_results/cgan_losses.json`.
- **`unconditional_gen.py`** — unconditional diffusion samples vs Real, diagnostics
  table + marginal KDEs. **Under `--gan` this is a no-op** — it prints a message
  and exits immediately, because the cGAN baseline's `generate_conditional()`
  always requires a binary event label; there is no unconditional cGAN generator
  to compare against. (Previously this script existed but was not wired into
  `evaluation_main.py`'s script list at all, and wrote to a stray
  `analysis/results/` directory instead of `diffusion_results/`/`gan_results/` —
  both fixed this session.)
- **`conditional_gen.py`** — conditional-generation diagnostics (event-conditioned
  samples vs Real): KDEs + diagnostics table. Loads pre-generated `.pt` files.
- **`cov.py`** — correlation/covariance heatmaps (real all / real events / uncond
  generated / cond generated). Event mask sourced from the conditioning series via
  `get_z_windows_*` with the percentile-converted threshold, matching `main.py`.
- **`dependency_metric.py`** — ACF of squared residuals (volatility-clustering
  check) for real and generated windows with 95% bands, plus a per-lag two-sample
  Welch t-test rendered as p-value curves. Outputs `acf_squared_real_band.png`,
  `acf_squared_gen_band.png`, `acf_squared_pvalues.png`. Requires `statsmodels`
  and `pmdarima`.
- **`distribution_metrics.py`** — per-asset Wasserstein distance between real
  event-window and generated last-day marginals, rendered to
  `wasserstein_table.png`. Also defines tail-index/exceedance-curve helpers.
- **`h_function_eval.py`** — forward-noises real windows at fixed τ, reports
  `h_model` output split by true label (calibration check, no sampling). **Writes
  to `analysis/results/`** — a third results directory, distinct from
  `diffusion_results/`/`gan_results/`; not currently present on disk (created on
  first run). Not part of `evaluation_main.py`'s script list.
- **`hypothesis_testing.py`** — energy-distance / episode-matched hypothesis
  tests (real-unconditional vs real-conditional; real-conditional vs
  generated-conditional) with bootstrap CIs. **Standalone** — not called by
  `evaluation_main.py`; own `argparse` CLI (`--split`, `--n-reference`,
  `--n-boot`); prints to stdout only, writes no files.

---

## Key Tensor Shapes

| Tensor | Shape | Notes |
|---|---|---|
| `X_train` / `X_test` | `(N, seq_len, A)` | Channels-last; from `make_sequences()`, entry-day EMA standardized; seq_len=10, A=10 |
| Diffusion training data | `(N, A, seq_len)` | Channels-first; from `get_diffusion_data()`, same entry-day EMA standardization (`df_z_wins` no longer exists) |
| `generated_samples_*.pt` | `(N, A, seq_len)` | Output of `ConditionalGenerator` |
| H-function input | `(B, A, seq_len)` | Channels-first (same as diffusion) |
| `mu_entry` / `sig_entry` | `(N, A)` | Per-window entry-day EMA stats; split into `*_train` / `*_test` |
| `MU` / `SG` | `(T, A)` | Per-row trailing EMA mean/vol (shifted 1 day), over the trimmed `self.df` |

---

## Known Issues / Gotchas

1. ~~**Duplicate `DataProcessor` in `data/data_processor.py`**~~ — **FIXED.** Only one
   class definition remains; the old commented-out first version is gone.

2. **`utils/portfolio.py` STILL calls the removed `invert_samples()`** — confirmed still
   live: `analyze_samples()` (line ~76) and `analyze_test_set()` (line ~112) both call
   `self.data_processor.invert_samples(...)`, which does not exist on the current
   `DataProcessor` (removed with the switch to per-window EMA standardization — the
   comment at `data_processor.py:715` explicitly notes it's gone and points at the
   replacement). **`main.py` Step 6 still raises `AttributeError`.** Port to
   `destandardize_windows()` + `sample_entry_stats()` (pass the event mask for
   conditional samples). This supersedes the old issues 2 and 3 about
   `mu_seq`/`weekday_mean`.

3. **`explore/diagnosis.py` no longer exists at all** (previously tracked as an empty
   file; now deleted outright). See "Data-Level Diagnostics — historical" above for what
   it used to produce. `explore/stationary_diagnosis.py` covers different ground (model
   residual stationarity, not raw-series/event/correlation sanity checks) and is not a
   replacement for the deleted plots.

4. **`cov_weight = 0.0`** — the correlation penalty in `DiffusionModel.loss_fn` is fully
   implemented (including the `real_corr_target` and the low-t mask) but inactive at the
   current config value.

5. **Conditional generation is fundamentally data-limited at rare event thresholds**
   (diagnosed 2026-07-08). Soft labels and episode reweighting are now **implemented** as
   levers (`constraint_mode="soft"`, `cfg.episode_reweight`), though both are off by
   default. Remaining untried lever: training `h` on synthetic diffusion-generated
   trajectories (partially addressed by the two-step path).

6. **Score model's own baseline under-dispersion:** unconditional generation is somewhat under-dispersed vs. real (~65-79% std ratio) before any conditioning. Accepted for now.

7. **`n_epochs` currently set low for iteration** — `diffusion.n_epochs=20`. Not a
   converged-training configuration; raise before drawing conclusions from generated samples.

8. **Config `latent_method` vs. CSV content can drift:** the dataset's conditioning column is whatever `import_data.py` last baked in; scripts *label* plots from the config value. After changing `latent_method`, rerun `import_data.py` to keep them in sync. The same applies to `growth_vars`/`inflation_vars`.

9. **yfinance downloads are flaky:** Yahoo intermittently rate-limits and returns an empty column for a random ticker ("possibly delisted; no price data found"), and `import_data.py`'s `dropna(subset=tickers)` then silently writes a 0-row CSV. Always check the printed `total rows` (~6687 for the current 10-ticker basket) after a rebuild; wait a few minutes and rerun if a ticker failed.

10. **`ConditionalGenerator` `pos_weight` inversion is applied in the autograd path but
    NOT the Q-model path** — the autograd branch computes
    `h_val = h_hat / (pos_weight*(1-h_hat) + h_hat)` before differentiating, while the
    Q-model branch builds its own `denom` expression. The two branches are not exactly
    equivalent; verify before relying on `use_q_model=True`.

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

## Architecture Reference

Two block types live in `transformer_score.py`. **The score network and the one-step
h-function both use `SpatioTemporalBlock`**; `DualAxisBlock` is retained and used by
`HFunctionTransformerTwoStep`.

### SpatioTemporalBlock — joint attention (current score / h-function path)

```
Input: (B, A, T)  where A=assets, T=seq_len
  ↓ input_proj (Linear 1→D)
  → (B, A, T, embed_dim)
  + day_emb(day_ids)   (1, 1, T, D)     # nn.Embedding lookup
  + stock_emb(stock_ids) (1, A, 1, D)   # nn.Embedding lookup
  ↓ permute + reshape, day-major → (B, T*A, D)   # joint spatiotemporal tokens

For each SpatioTemporalBlock:
  1. Joint attn: AdaLN(x, t_emb) → MHA over ALL T*A tokens → residual
  2. FFN:        AdaLN → GELU, 4× expansion → residual

  ↓ reshape back to (B, A, T, D)
  ↓ output_norm + output_proj (Linear D→1)
Output: (B, A, T)  — eps prediction (NOT the score; see score_from_eps)
```

Every (asset, day) token attends to every other in **one hop**. At `A*T = 100` tokens
the full attention matrix is small enough that factorizing buys nothing.

### DualAxisBlock — factorized attention (two-step h-function only)

```
For each DualAxisBlock:
  1. Temporal attn:   reshape (B*A, T, D) → MHA → residual
  2. Asset attn:      reshape (B*T, A, D) → MHA → residual
  3. FFN:             GELU, 4× expansion, residual
  (each sub-layer: AdaLN with t_emb conditioning; residual dropout after each)
```

AdaLN: `LayerNorm(x) * (1 + scale(t)) + shift(t)`, scale/shift from a linear projection
of the time embedding, broadcast over any number of intermediate dims.

---

## Experiments

### 2026-08-15 — Macro-conditional stationarity diagnostics, BFK test pipeline, analysis/ merge, unconditional_gen.py wiring fix, H-function early stopping

**`analysis/` supersedes `diffusion_model_analysis/` + `evaluation/`.** The two
directories described throughout the older sections of this document were merged
into a single `analysis/` at some point; `explore/diagnosis.py` was also deleted
outright (previously just empty). All directory-structure references throughout
this doc have been corrected to match; see the "Directory Structure",
"Data-Level Diagnostics — historical", and "Analysis Scripts" sections above.

**`analysis/unconditional_gen.py` existed but was silently not running.** It was
never added to `evaluation_main.py`'s `SCRIPTS` list, and it wrote to a stray
`analysis/results/` directory instead of `diffusion_results/`/`gan_results/` like
every other script. Both fixed: added to the script list, output path corrected,
and given a `--gan` no-op path (prints and exits) since the cGAN baseline has no
unconditional generator to compare against — `generate_conditional()` always
requires a binary event label.

**`explore/stationary_diagnosis.py` — new macro-conditional stationarity
diagnostic**, built and iterated on extensively this session (see its own section
above for the full design). Headline design points worth remembering:
- Fits `mu(m)`/`sigma(m)` (two small MLPs) on the FULL series by joint Gaussian
  NLL — no train/test split for the headline result, since the question is
  representability of the macro relationship, not forecasting.
- Two diagnostics: `macro_bins` (conditional mean/variance flatness across
  quantile bins of the macro state) and `time_blocks` (autocovariance drift
  across consecutive time blocks — deliberately NOT the same partition as the
  macro bins, since autocovariance needs temporally adjacent points).
- **A real methodological bug was caught and fixed during this work:** an initial
  permutation control shuffled the macro state and then binned the shuffled
  model's residuals BY THE SHUFFLED macro state — a null test of "are residuals
  similar across random groups" (trivially yes), which produced a false-negative
  read that real macro conditioning was no better than shuffled. The fix
  evaluates both the real and shuffled models in bins of the REAL macro state;
  only then does the real model's advantage appear (example bin: raw variance
  2.20 → 2.05 under a shuffled-macro model → 0.81 under the real-macro model).
- Network capacity was tested and found not to matter for this result (13 vs.
  4,353 parameters gave near-identical outcomes) — ruling out an earlier
  hypothesis that a large network could trivially memorize its way to flat bins.
- Writes `explore/nn_standardized_macro.csv` (full-series standardized residuals
  per ticker), which feeds directly into `bfk_stationarity.py` below.
- **A separate DFT/trispectrum-based portmanteau test (Dwivedi–Subba Rao) was
  built, debugged (multiple real bugs: trispectrum scaling off by T², pairing
  terms needing to be zeroed before smoothing rather than subtracted after,
  bandwidth needing to scale with T to avoid negative correction denominators at
  realistic sample sizes), and ultimately set aside as unvalidated at
  `explore/test.py`. That file no longer exists in the current repo — if
  wanted, it would need to be rebuilt.**

**`explore/bfk_stationarity.py` — the stationarity test actually in production
use** (Bücher–Fermanian–Kojadinov, via R's `npcp` through `rpy2`; not the
Dwivedi–Subba Rao attempt above — different method). Consumes
`nn_standardized_macro.csv`. Run via SLURM array job across all 10 tickers
(`submit_bfk_jobs.sh` → `bfk_array_job.sh`); results in
`explore/bfk_results/<TICKER>.json`, all 10 present, run more than once across
this project's history (multiple job IDs' worth of logs retained).

**H-function training now has EMA-smoothed early stopping** (`hfunction_direct.py`
/ `HFunctionConfig`). Driven by extreme class imbalance at the current event
threshold (253 positive events / 5,057 train windows, 5.00% — measured this
session, ruling out a held-out validation split as too noisy to trust at that
scale) and by noisy raw per-epoch loss (epoch-to-epoch std ~0.037, ~49% of
epochs upward on an observed 425-epoch run — too noisy for a direct "did loss
improve" check). Design: EMA of the loss (`early_stop_ema_span=20`) compared
against its best-so-far value; `early_stop_patience=100` epochs of no
improvement `> early_stop_min_delta` triggers a stop and restores the best
checkpoint; patience kept above `scheduler_patience=75` so an LR decay gets a
chance to help first. `early_stop_min_delta` was tuned against the actual loss
trace — the initial `5e-3` was too permissive (loss was still declining
~0.01–0.03 per 25 epochs even late in the run, so the clock kept resetting on
genuine small progress and never fired); raised to `2e-2`. `n_epochs` raised
from 425 to 1000 and reframed as an upper-bound safety cap rather than the
primary stopping signal.

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
