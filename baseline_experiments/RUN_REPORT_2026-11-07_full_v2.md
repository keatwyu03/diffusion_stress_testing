# Run report — corrected FULL-macro pipelines (submitted 2026-11-07, ~01:00–01:35)

Scope: **Full macro set only.** No CPI-only or inflation-only runs. Evaluation methodology unchanged.

## 1. What was wrong and what changed

| Branch | Before (wrong) | Now |
|---|---|---|
| Inclusion, rare-event (M1–M4, base M0) | diffusion targets = production residuals standardized on S0 (`m_t`) | **raw daily log returns** (same as conditional-only) |
| Inclusion, conditional-only (M0–M4) | already raw log returns | unchanged — job 18359403 kept running |
| Embedding, rare-event (S0–S4) | every method used the S0 residuals | each S_k uses **its own** NN residuals; fresh base model per method |
| Embedding, conditional-only (S0–S4) | raw log returns | each S_k trains on **its own** NN residuals; generated residuals mapped back to returns with S_k's own μ/σ |

Per-method standardization: `explore/nn_standardize.py` `run_ticker`, **unmodified**, identical hyperparameters for S0–S4
(hidden 64, 2 layers, ≤1000 epochs w/ early stop, batch 128, lr 1e-3, δ 1e-3, seed 0), full-horizon fit (no train/test split, as in
production), on the same 6,319 dates for every method (2000-09-25 … 2026-07-08 = every date where all five states exist).
z^(k)_{t,j} = (r_{t,j} − μ_{k,j}(v^(k)_t)) / σ_{k,j}(v^(k)_t); inverse r = μ + σ·z (max error ~3e-8).

### Files changed / added
- `baseline_experiments/macro_embed_method/standardize_own_macro.py` (new) — per-method standardizer; saves `residuals.csv`, `mu.csv`, `sigma.csv`, `standardizer.pt` (networks + macro normalization), `meta.json`.
- `baseline_experiments/macro_embed_method/diffusion_adapters.py` — targets = own residuals (no raw fallback), inverse transform, saves returns + residuals + μ/σ + dates + conditioning; refuses checkpoints not trained on that method's own residuals.
- `baseline_experiments/macro_embed_method/main_exp2.py` — `--experiment` / `--results-dir` (unchanged metrics).
- `baseline_experiments/macro_inclusion/build_channel_csvs.py` — `--raw-returns`, `--out-dir`.
- `baseline_experiments/macro_inclusion/build_channel_csvs_exp2_ownstd.py` (new) — builds + verifies the per-method rare-event CSVs (exits non-zero on any failed check).
- `baseline_experiments/macro_inclusion/save_rare_event_artifacts.py` (new) — saves generated + real windows, dates, event values, inverse-transform inputs.
- `baseline_experiments/macro_inclusion/run_doobh_exp2_general.py` — `--own-std`: per-channel fresh base model kept in the channel folder, provenance in `channel_meta.json`, artifacts step, **non-zero exit** on any training/generation/evaluation/artifact failure.
- `baseline_experiments/macro_inclusion/run_doobh_channels.py`, `run_doobh_channels_causal.py` — artifacts step; non-zero exit on failure (applies to future runs; the two running inclusion jobs started before this patch and have no dependents).
- `baseline_experiments/macro_inclusion/main_exp1.py` — records/validates `target: raw daily log returns` in checkpoint meta.
- `baseline_experiments/macro_inclusion/window_conditioning.py` — `standardization_dir` path.
- `baseline_experiments/exp_evaluation.py`, `macro_inclusion/tail_evaluation.py` — **not changed tonight** (changed earlier).
- `baseline_experiments/preflight_full_v2.py` (new) — re-runnable preflight suite.
- Not touched: `main.py`, `data/`, `models/`, `config/config.py` (main repo), `explore/nn_standardize.py`.

## 2. Configuration inventory (the "19")

6 branches, **28 method-level runs** (no configurations invented):

| Branch | Method-level runs |
|---|---|
| Inclusion conditional-only (causal) | M0, M1, M2, M3, M4 (5) |
| Inclusion rare-event causal | M1, M2, M3, M4 (4) — M0 = shared unconditional base model |
| Inclusion rare-event non-causal | M1–M4 (4) |
| Embedding conditional-only (causal) | S0–S4 (5) |
| Embedding rare-event causal | S0–S4 (5) |
| Embedding rare-event non-causal | S0–S4 (5) |

"19" = 5 + 4 + 5 + 5, i.e. counting causal and non-causal rare-event once. Conditional-only non-causal is intentionally not run (M1–M3 would leak the target returns).

## 3. Preflight (all passed before the embedding jobs were submitted)

`venv/bin/python3 baseline_experiments/preflight_full_v2.py` → **72/72 checks passed**. Highlights:
- Inclusion conditional-only: eval windows = raw returns; M0 unconditional; one shared scale; checkpoints trained on raw returns (saved scale 55.3608 = 1/std of raw train windows; residual targets would give ≈1).
- Inclusion rare-event (both copies): assets = raw log returns, event column unchanged (382 / 381 events, same as before), window→date alignment, `latent_standardized=False` (run tag `_EWMA`), only checkpoint present was trained by the running job after it started (01:19), base trained before any load.
- Standardizers: same hyperparameters and fit dates for S0–S4, each fit on its own state, saved networks reproduce μ/σ, r = μ + σ·z.
- Embedding rare-event CSVs: column 0 unchanged, residuals finite on every production date ≤ end_date, S1–S4 ≠ S0 residuals, DataProcessor gives 6,299 rows / 5,089 train windows for every channel (same as inclusion).
- Embedding conditional-only (stubbed train of the real script): each S_k trains on exactly its own residual windows (not S0's, not raw), conditioner = its own state's 10 values ending 2 rows before the window, identical windows across S0–S4.
- Smoke runs (tiny config, throwaway copies): embedding launcher end-to-end causal (S1, S4) and non-causal (S2, S3) — fresh, distinct base model per channel, provenance recorded, evaluation + artifacts saved, exit 0; failure path (bad standardizer path) exits 1. Conditional-only embedding: 1-epoch train → sample → inverse (err ~2e-8) → reload with target validation → unchanged evaluator, exit 0.
- No existing unit tests exist in the repo (no test files; pytest not installed).

## 4. Jobs

| Job ID | Name | Branch (Full) | Depends on | Status at ~01:35 |
|---|---|---|---|---|
| 18364610–18364614 | std_full_s0 … s4 | Embedding preprocessing (S0–S4 standardizers) | — | COMPLETED (exit 0) |
| 18359403 | cond_full_inc | Inclusion conditional-only M0–M4 (raw returns) | — | RUNNING since 20:30 Nov 6 (kept; already correct) |
| 18364671 | v2_rare_inc_nc | Inclusion rare-event non-causal (raw returns) | — | RUNNING |
| 18364672 | v2_rare_inc_c | Inclusion rare-event causal (raw returns) | — | RUNNING |
| 18364852 | v2_rare_emb_nc | Embedding rare-event non-causal (own residuals) | afterok 18364610:…:18364614 | RUNNING |
| 18364853 | v2_rare_emb_c | Embedding rare-event causal (own residuals) | afterok 18364610:…:18364614 | RUNNING |
| 18364854 | v2_cond_full_emb | Embedding conditional-only S0–S4 (own residuals) | afterok 18364610:…:18364614 | RUNNING |

Cancelled tonight: 18359404 (Full embedding conditional-only on raw returns — wrong targets), 18359405 / 18359406 (CPI — out of scope).

Stages inside each job run in order and stop on failure: rare-event = base diffusion (M0 / per-S_k) → Doob-h H-function → guided generation → evaluation (unchanged) → artifacts; conditional-only = per-model diffusion training → sampling → inverse transform → evaluation (unchanged). `main.py` runs base → H → generation in one process, so these stages cannot be separate Slurm jobs without modifying `main.py`.

Rough finish times: inclusion rare-event ~05:30; embedding rare-event ~07:00 (5 channels × ~65 min); inclusion conditional-only ~13:00; embedding conditional-only ~17:00–18:00 (5 models × ~3 h sampling).

## 5. Outputs and logs

- Logs: `/scr/kadenwu/Conditional_diffusion/logs/<job name>_<job id>.{out,err}`
- Standardizers: `baseline_experiments/experiment_1 (Full)/extras/standardization_own/<S_k>/`
- Embedding rare-event channel CSVs: `explore/standardization_results/data_for_diffusion_exp2_s{k}_ownstd.csv`; inclusion raw-return CSVs: `…/data_for_diffusion_{channel}_raw.csv`
- Conditional-only: `experiment_1 (Full)/causal/macro_inclusion/conditional_only/` and `experiment_1 (Full)/causal/macro_embed/conditional_only/<S_k>/diffusion_causal/`
  (`generated_samples.npy` returns, `generated_residuals.npy`, `eval_data.npz` = real returns + residuals + μ/σ + dates + conditioning, `checkpoint.pt`, `checkpoint_meta.json`, `loss_history.csv`)
- Rare-event results stay in the isolated run copies until filed:
  - `/scr/kadenwu/Conditional_diffusion_v2_rare_inc_nc/baseline_experiments/main_results/doobh_results/<channel>/`
  - `/scr/kadenwu/Conditional_diffusion_v2_rare_inc_c/…/main_results/doobh_results_causal/<channel>/`
  - `/scr/kadenwu/Conditional_diffusion_v2_rare_emb_nc/…/main_results/doobh_results_exp2_noncausal_ownstd/<channel>/`
  - `/scr/kadenwu/Conditional_diffusion_v2_rare_emb_c/…/main_results/doobh_results_exp2_causal_ownstd/<channel>/`
  Each channel: generated samples, H-function, `doobh_comparison_train.csv`, `rare_event_artifacts.npz/.json` (generated + real windows, dates, event values, entry stats, μ/σ, reconstructed returns), and for embedding the channel's own base model + `channel_meta.json` provenance.
- Old outputs built on the wrong targets were moved (not deleted) to `experiment_1 (Full)/extras/stale_20261107_wrong_targets/`.

## 6. Outstanding items / caveats

1. **Filing rare-event results** into `experiment_1 (Full)/{causal,noncausal}/{macro_inclusion,macro_embed}/rare_event/`: `build_rare_event_tables.py` still points at the old copy names and results folders; update it tomorrow together with the evaluation work.
2. **Units:** rare-event embedding evaluation (`doobh_comparison_train.csv`) is in each method's residual units, unchanged methodology; the returns-unit versions are in `rare_event_artifacts.npz`. Rare-event inclusion is now in raw-return units.
3. **Conditional-only embedding window pool** is 4,719 windows (3,519 train), smaller than inclusion's 5,416 (4,216): residual targets don't exist on the days where the S0 state is missing. Identical across S0–S4.
4. **Standardizer timing:** like production, it maps same-day v_t to μ/σ (contemporaneous). The causal restriction applies to the conditioner/event alignment, not to this transform.
5. The two inclusion rare-event jobs were started before the non-zero-exit patch; they have no dependents, and their success is visible from `rare_event_artifacts.npz` per channel.
6. Throwaway smoke copies to delete when convenient (the automated cleanup was blocked by a safety check): `/scr/kadenwu/Conditional_diffusion_smoke_inc_c`, `…_smoke_emb_c`, `…_smoke2_inc_c`, `…_smoke3_emb_nc`.
7. CPI partial outputs from the cancelled jobs remain in `experiment_2 (CPI Only)/causal/macro_inclusion/conditional_only/` (incomplete).
