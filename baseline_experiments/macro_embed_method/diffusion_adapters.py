"""
Experiment-2-local training/sampling adapters. Confirmed (approved plan +
smoke-tested in common.py) that main_exp1.py's train_model/sample_model/
get_condition_array/conditioning_signature/validate_checkpoint_compatibility
are NOT safely reusable as black boxes for a new S0-S4 key scheme (they
resolve main_exp1's own module-level COND_KEY global, and
conditioning_signature/build_windows's cond_vals dict are hardcoded to
Experiment 1's M1-M4 column names). So these adapters receive conditioning
arrays and metadata EXPLICITLY and call DiffusionModel.train()/.sample()
directly (both confirmed fully generic -- condition_data/condition is an
arbitrary scalar tensor, no model-key awareness at all). Only the genuinely
parameter-free pieces are imported from main_exp1: fit_scale_factor,
build_diffusion_model, architecture_signature.

Group B targets: raw log-returns (same as Experiment 1), with the same
common reversible numerical scale factor convention (fit_scale_factor) --
NOT the standardized residuals from Group A. Group A (stationarity) and
Group B (diffusion) are two independent tests of the same underlying daily
macro state, per the approved plan -- this module never touches
macro_standardization.py's output.

Conditioning feature: the full causal window of each method's own raw daily
state (seq_len values ending event_lag_gap + 1 rows before the return window
starts) -- same conditioner and helper as main_exp1.py's M1-M4.

Evaluation is IN-SAMPLE: train on the train windows, then sample/score on an
evenly spaced subset of those same train windows (--n-eval-windows, same rule
as main_exp1.py), saved to <method>/diffusion_causal/eval_data.npz.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import common

_HERE = os.path.dirname(os.path.abspath(__file__))
_MACRO_INCLUSION = os.path.join(os.path.dirname(_HERE), "macro_inclusion")
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _MACRO_INCLUSION)
sys.path.insert(0, _ROOT)

import main_exp1 as _m1  # noqa: E402
from models import DiffusionModel  # noqa: E402
from utils import set_seed  # noqa: E402
from window_conditioning import (  # noqa: E402
    EXPERIMENTS, causal_value_windows, eval_window_idx, experiment_paths, use_window_conditioning,
)

OUT_DIR = os.path.join(os.path.dirname(_HERE), "main_results", "exp2_results")


def build_cond_feature(method: str, manifest, seq_len: int, states_dir: str) -> pd.Series:
    """The method's raw daily state on the shared trading calendar."""
    state_csv = os.path.join(states_dir, common.METHOD_DIRNAMES[method], "daily_state.csv")
    daily_state = pd.read_csv(state_csv, index_col=0, parse_dates=True)["daily_state"]
    return daily_state.reindex(manifest["df"].index)


def per_window_cond_values(cond_feature: pd.Series, manifest, event_causal: bool, event_lag_gap: int) -> np.ndarray:
    """(N, seq_len): the full causal window of raw state values for each return
    window -- seq_len values ending event_lag_gap + 1 rows before it starts
    (same window and helper as main_exp1.py). Conditional-only is causal-only."""
    if not event_causal:
        raise ValueError("Conditional-only runs are causal-only. Use --causal true.")
    return causal_value_windows(cond_feature.to_numpy(), manifest["starts"], manifest["seq_len"], event_lag_gap)


def common_valid_window_mask(all_cond_values: dict) -> np.ndarray:
    """A window is kept only if every value of its conditioning window is
    defined for EVERY method, so S0-S4 share one identical window pool."""
    n = len(next(iter(all_cond_values.values())))
    valid = np.ones(n, dtype=bool)
    for v in all_cond_values.values():
        valid &= np.isfinite(v).all(axis=1)
    return valid


def conditioning_diagnostics(cond_values: np.ndarray, mask: np.ndarray) -> dict:
    """Per the approved plan: save each method's conditioning-feature
    distribution/std/fraction-of-zero-changes/train-test range BEFORE
    training -- a forward-filled state (S2) is a step function, so its
    windowed change is exactly zero whenever no new monthly observation
    fell in the lookback, which could make its transformed feature
    degenerate across many windows. This is diagnosed, not fixed by
    changing the transform (kept identical across methods) and zero-change
    windows are never dropped."""
    v = cond_values[mask]
    return {
        "n_windows": int(len(v)), "mean": float(np.mean(v)), "std": float(np.std(v)),
        "min": float(np.min(v)), "max": float(np.max(v)),
        "fraction_exactly_zero": float(np.mean(v == 0.0)),
    }


def build_X(df: pd.DataFrame, starts: np.ndarray, seq_len: int) -> np.ndarray:
    R = df[common.TICKERS].to_numpy(dtype=np.float64)
    return np.stack([R[i:i + seq_len].T for i in starts])  # (Nw, A, T)


def train_positions(manifest, mask: np.ndarray, cfg) -> np.ndarray:
    """Manifest positions of the train windows. Same rule as main_exp1.py: the
    train/test cutoff is applied to the common valid pool (not the raw
    calendar), so S0-S4 train on exactly the windows M0-M4 train on."""
    valid_pos = np.flatnonzero(mask)
    split_idx, _ = _m1.train_test_split_idx(manifest["starts"][valid_pos], manifest["window_end_dates"][valid_pos],
                                            manifest["seq_len"], cfg)
    return valid_pos[:split_idx]


def train_one(method: str, manifest, cond_values: np.ndarray, mask: np.ndarray, cfg, args, out_dir):
    set_seed(args.seed)  # same init + shuffle seed for every method (as main_exp1.train_model)
    X = build_X(manifest["df"], manifest["starts"], manifest["seq_len"])

    pos = train_positions(manifest, mask, cfg)
    X_train_raw = X[pos]
    cond_train = cond_values[pos]

    scale = _m1.fit_scale_factor(X_train_raw)
    X_train_scaled = torch.tensor(X_train_raw * scale, dtype=torch.float32)

    c_mu, c_sigma = float(cond_train.mean()), float(cond_train.std())
    cond_tensor = torch.tensor((cond_train - c_mu) / c_sigma, dtype=torch.float32)

    dm = _m1.build_diffusion_model(cfg, seq_len=manifest["seq_len"])
    use_window_conditioning(dm, cond_tensor.shape[1], cfg.diffusion.cond_dim)  # before the optimizer is built
    dm.train(
        train_data=X_train_scaled, condition_data=cond_tensor,
        batch_size=args.batch_size, n_epochs=args.epochs,
        learning_rate=cfg.diffusion.learning_rate, weight_decay=cfg.diffusion.weight_decay,
        scheduler_patience=cfg.diffusion.scheduler_patience, scheduler_factor=cfg.diffusion.scheduler_factor,
        block_sampling=cfg.diffusion.block_sampling, end_dates=manifest["window_end_dates"][pos],  # same as main.py
        loss_csv_path=os.path.join(out_dir, "loss_history.csv"),
    )
    os.makedirs(out_dir, exist_ok=True)
    dm.save(os.path.join(out_dir, "checkpoint.pt"))

    meta = {
        "method": method, "architecture": _m1.architecture_signature(cfg, manifest["seq_len"], len(common.TICKERS)),
        "conditioning": {"transform": "causal_value_window (raw daily_state, seq_len values)", "seq_len": manifest["seq_len"],
                          "event_causal": cfg.data.event_causal, "event_lag_gap": cfg.data.event_lag_gap,
                          "scale_factor": scale, "cond_mu": c_mu, "cond_sigma": c_sigma,
                          "n_train_windows_used": int(len(pos)), "block_sampling": cfg.diffusion.block_sampling},
    }
    with open(os.path.join(out_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return dm, scale, c_mu, c_sigma


def in_sample_eval_windows(manifest, mask: np.ndarray, cfg, n_eval: int) -> np.ndarray:
    """Manifest positions of the in-sample evaluation windows: the same train
    windows train_one fits on, subset by eval_window_idx."""
    train_pos = train_positions(manifest, mask, cfg)
    return train_pos[eval_window_idx(len(train_pos), n_eval)]


def sample_one(method: str, manifest, cond_values: np.ndarray, mask: np.ndarray, dm, scale, c_mu, c_sigma, cfg, args, out_dir):
    X = build_X(manifest["df"], manifest["starts"], manifest["seq_len"])

    pos = in_sample_eval_windows(manifest, mask, cfg, args.n_eval_windows)
    X_test_raw, cond_test = X[pos], cond_values[pos]
    n_test = len(pos)

    K = args.n_samples
    cond_z = (cond_test - c_mu) / c_sigma
    cond_flat = torch.tensor(cond_z, dtype=torch.float32).repeat_interleave(K, dim=0)  # (n_test*K, seq_len)

    A, T = dm.in_channels, dm.sample_size
    total = n_test * K
    flat = np.zeros((total, A, T), dtype=np.float32)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        samp = dm.sample(batch_size=end - start, num_steps=args.num_steps, stoch=1.0,
                          condition=cond_flat[start:end], seed=args.seed + start)
        flat[start:end] = samp.detach().cpu().numpy()
    gen = (flat.reshape(n_test, K, A, T) / scale).astype(np.float32)

    n_nan = int(np.isnan(gen).sum() + np.isinf(gen).sum())
    if n_nan:
        raise ValueError(f"{method}: {n_nan} NaN/Inf in generated samples")

    np.save(os.path.join(out_dir, "generated_samples.npy"), gen)
    np.savez(os.path.join(out_dir, "eval_data.npz"), X_eval=X_test_raw, cond_eval=cond_test, eval_split="train",
             manifest_pos=pos, window_start_dates=manifest["window_start_dates"][pos].strftime("%Y-%m-%d").to_numpy(dtype=str),
             window_end_dates=manifest["window_end_dates"][pos].strftime("%Y-%m-%d").to_numpy(dtype=str), K=K)
    print(f"  {method}: saved generated samples {gen.shape} -> {out_dir}/generated_samples.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default="s0,s1,s2,s3,s4")
    parser.add_argument("--mode", choices=["train", "sample", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-eval-windows", type=int, default=1200,
                         help="In-sample evaluation windows, evenly spaced over the train period (0 = all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default="full",
                         help="Picks the daily-state folder and the default results folder together: "
                              "full -> experiment_1 (Full)/causal/macro_embed/conditional_only, "
                              "cpi -> experiment_2 (CPI Only)/causal/macro_embed/conditional_only.")
    parser.add_argument("--causal", choices=["true", "false"], default="true",
                         help="Conditional-only is causal-only (false raises). Overrides config.data.event_causal "
                              "in memory; event_lag_gap always comes from config.data.event_lag_gap.")
    parser.add_argument("--states-dir", type=str, default=None,
                         help="Override the experiment's <method>/daily_state.csv folder (then --out-dir is required).")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Results go to <out-dir>/<method>/diffusion_causal/. Default: from --experiment.")
    args = parser.parse_args()

    paths = experiment_paths(args.experiment)
    if args.states_dir and not args.out_dir:
        raise SystemExit("--states-dir overrides the experiment's input; also pass --out-dir so the "
                         "results don't land in the wrong experiment folder.")
    args.states_dir = args.states_dir or paths["states_dir"]
    args.out_dir = args.out_dir or paths["macro_embed"]

    cfg = common.get_default_config()
    args.epochs = args.epochs or cfg.diffusion.n_epochs
    args.batch_size = args.batch_size or cfg.diffusion.batch_size
    args.num_steps = args.num_steps or cfg.diffusion.num_steps
    args.seed = args.seed if args.seed is not None else cfg.seed

    manifest = common.get_window_manifest(cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seq_len = manifest["seq_len"]

    if args.causal is not None:
        cfg.data.event_causal = args.causal == "true"
    subdir = f"diffusion_{'causal' if cfg.data.event_causal else 'noncausal'}"
    print(f"Event-window attachment: event_causal={cfg.data.event_causal}, event_lag_gap={cfg.data.event_lag_gap}")
    print(f"States from {args.states_dir}; results to {args.out_dir}/<method>/{subdir}")
    all_cond = {m: per_window_cond_values(build_cond_feature(m, manifest, seq_len, args.states_dir), manifest,
                                          cfg.data.event_causal, cfg.data.event_lag_gap) for m in methods}
    mask = common_valid_window_mask(all_cond)
    print(f"Common valid-window pool across {methods}: {mask.sum()}/{len(mask)} windows")

    for m in methods:
        diag = conditioning_diagnostics(all_cond[m], mask)
        out_dir = os.path.join(args.out_dir, common.METHOD_DIRNAMES[m], subdir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "conditioning_diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)
        print(f"  {m}: n_windows={diag['n_windows']}, std={diag['std']:.4g}, "
              f"frac_zero={diag['fraction_exactly_zero']:.2%}, range=[{diag['min']:.4g},{diag['max']:.4g}]")

    for m in methods:
        print(f"\n=== Diffusion: {m} ({common.METHOD_LABELS[m]}) ===")
        out_dir = os.path.join(args.out_dir, common.METHOD_DIRNAMES[m], subdir)
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")

        if args.mode in ("train", "both"):
            dm, scale, c_mu, c_sigma = train_one(m, manifest, all_cond[m], mask, cfg, args, out_dir)
        else:
            with open(os.path.join(out_dir, "checkpoint_meta.json")) as f:
                meta = json.load(f)
            dm = _m1.build_diffusion_model(cfg, seq_len=seq_len)
            use_window_conditioning(dm, seq_len, cfg.diffusion.cond_dim)  # match trained layer before loading
            dm.load(ckpt_path)
            scale, c_mu, c_sigma = meta["conditioning"]["scale_factor"], meta["conditioning"]["cond_mu"], meta["conditioning"]["cond_sigma"]

        if args.mode in ("sample", "both"):
            sample_one(m, manifest, all_cond[m], mask, dm, scale, c_mu, c_sigma, cfg, args, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
