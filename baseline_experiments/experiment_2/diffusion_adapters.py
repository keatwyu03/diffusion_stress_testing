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

Conditioning feature: windowed absolute change of each method's own daily
state (|state[end] - state[end-window+1]|, window = seq_len), mirroring
Experiment 1's already-validated macro_change design (keeps "changing only
the macro conditioning state" literally true -- same transform, different
underlying series). Saved separately from the stationarity state level
(daily_macro_states.py's daily_state.csv) -- never silently reused as the
standardizer's `m` just because diffusion uses it.
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
_EXP1 = os.path.join(os.path.dirname(_HERE), "experiment_1")
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _EXP1)
sys.path.insert(0, _ROOT)

import main_exp1 as _m1  # noqa: E402
from models import DiffusionModel  # noqa: E402

OUT_DIR = os.path.join(_HERE, "exp2_results")


def windowed_abs_change(daily_state: pd.Series, seq_len: int) -> pd.Series:
    """|state[t] - state[t-(seq_len-1)]| -- same formula as main_exp1.py's
    macro_change, applied here to each method's own daily state."""
    return (daily_state - daily_state.shift(seq_len - 1)).abs()


def build_cond_feature(method: str, manifest, seq_len: int) -> pd.Series:
    state_csv = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[method], "daily_state.csv")
    daily_state = pd.read_csv(state_csv, index_col=0, parse_dates=True)["daily_state"]
    daily_state = daily_state.reindex(manifest["df"].index)
    return windowed_abs_change(daily_state, seq_len)


def per_window_cond_values(cond_feature: pd.Series, manifest) -> np.ndarray:
    """One value per window, at cond_row = window_start - 1 (causal: known
    strictly before the target window), matching main_exp1.py's own
    convention exactly."""
    values = cond_feature.to_numpy()
    cond_row = manifest["starts"] - 1
    valid_row = cond_row >= 0
    out = np.full(len(manifest["starts"]), np.nan)
    out[valid_row] = values[cond_row[valid_row]]
    return out


def common_valid_window_mask(all_cond_values: dict) -> np.ndarray:
    """Intersect validity across ALL methods being compared -- same lesson
    as the Doob-h common-pool fix, applied proactively here instead of
    after the fact."""
    n = len(next(iter(all_cond_values.values())))
    valid = np.ones(n, dtype=bool)
    for v in all_cond_values.values():
        valid &= np.isfinite(v)
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


def train_one(method: str, manifest, cond_values: np.ndarray, mask: np.ndarray, cfg, args, out_dir):
    X = build_X(manifest["df"], manifest["starts"], manifest["seq_len"])
    split_idx, test_start_idx = manifest["split_idx"], manifest["test_start_idx"]

    train_mask = mask[:split_idx]
    X_train_raw = X[:split_idx][train_mask]
    cond_train = cond_values[:split_idx][train_mask]

    scale = _m1.fit_scale_factor(X_train_raw)
    X_train_scaled = torch.tensor(X_train_raw * scale, dtype=torch.float32)

    c_mu, c_sigma = float(cond_train.mean()), float(cond_train.std())
    cond_tensor = torch.tensor((cond_train - c_mu) / c_sigma, dtype=torch.float32)

    dm = _m1.build_diffusion_model(cfg, seq_len=manifest["seq_len"])
    dm.train(
        train_data=X_train_scaled, condition_data=cond_tensor,
        batch_size=args.batch_size, n_epochs=args.epochs,
        learning_rate=cfg.diffusion.learning_rate, weight_decay=cfg.diffusion.weight_decay,
        scheduler_patience=cfg.diffusion.scheduler_patience, scheduler_factor=cfg.diffusion.scheduler_factor,
        block_sampling=False, loss_csv_path=os.path.join(out_dir, "loss_history.csv"),
    )
    os.makedirs(out_dir, exist_ok=True)
    dm.save(os.path.join(out_dir, "checkpoint.pt"))

    meta = {
        "method": method, "architecture": _m1.architecture_signature(cfg, manifest["seq_len"], len(common.TICKERS)),
        "conditioning": {"transform": "windowed_abs_change", "seq_len": manifest["seq_len"],
                          "scale_factor": scale, "cond_mu": c_mu, "cond_sigma": c_sigma,
                          "n_train_windows_used": int(train_mask.sum())},
    }
    with open(os.path.join(out_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return dm, scale, c_mu, c_sigma


def sample_one(method: str, manifest, cond_values: np.ndarray, mask: np.ndarray, dm, scale, c_mu, c_sigma, cfg, args, out_dir):
    split_idx, test_start_idx = manifest["split_idx"], manifest["test_start_idx"]
    X = build_X(manifest["df"], manifest["starts"], manifest["seq_len"])

    test_mask = mask[test_start_idx:]
    X_test_raw = X[test_start_idx:][test_mask]
    cond_test = cond_values[test_start_idx:][test_mask]
    n_test = min(args.n_test_windows, len(X_test_raw)) if args.n_test_windows else len(X_test_raw)
    X_test_raw, cond_test = X_test_raw[:n_test], cond_test[:n_test]

    K = args.n_samples
    cond_z = (cond_test - c_mu) / c_sigma
    cond_flat = torch.tensor(cond_z, dtype=torch.float32).repeat_interleave(K)

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
    np.savez(os.path.join(out_dir, "test_data.npz"), X_test=X_test_raw, cond_test=cond_test, n_test=n_test, K=K)
    print(f"  {method}: saved generated samples {gen.shape} -> {out_dir}/generated_samples.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default="s0,s1,s2,s3,s4")
    parser.add_argument("--mode", choices=["train", "sample", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-test-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = common.get_default_config()
    args.epochs = args.epochs or cfg.diffusion.n_epochs
    args.batch_size = args.batch_size or cfg.diffusion.batch_size
    args.num_steps = args.num_steps or cfg.diffusion.num_steps
    args.seed = args.seed if args.seed is not None else cfg.seed

    manifest = common.get_window_manifest(cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seq_len = manifest["seq_len"]

    print("Building windowed-abs-change conditioning feature for every requested method...")
    all_cond = {m: per_window_cond_values(build_cond_feature(m, manifest, seq_len), manifest) for m in methods}
    mask = common_valid_window_mask(all_cond)
    print(f"Common valid-window pool across {methods}: {mask.sum()}/{len(mask)} windows")

    for m in methods:
        diag = conditioning_diagnostics(all_cond[m], mask)
        out_dir = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[m], "diffusion")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "conditioning_diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)
        print(f"  {m}: n_windows={diag['n_windows']}, std={diag['std']:.4g}, "
              f"frac_zero={diag['fraction_exactly_zero']:.2%}, range=[{diag['min']:.4g},{diag['max']:.4g}]")

    for m in methods:
        print(f"\n=== Diffusion: {m} ({common.METHOD_LABELS[m]}) ===")
        out_dir = os.path.join(OUT_DIR, common.METHOD_DIRNAMES[m], "diffusion")
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")

        if args.mode in ("train", "both"):
            dm, scale, c_mu, c_sigma = train_one(m, manifest, all_cond[m], mask, cfg, args, out_dir)
        else:
            with open(os.path.join(out_dir, "checkpoint_meta.json")) as f:
                meta = json.load(f)
            dm = _m1.build_diffusion_model(cfg, seq_len=seq_len)
            dm.load(ckpt_path)
            scale, c_mu, c_sigma = meta["conditioning"]["scale_factor"], meta["conditioning"]["cond_mu"], meta["conditioning"]["cond_sigma"]

        if args.mode in ("sample", "both"):
            sample_one(m, manifest, all_cond[m], mask, dm, scale, c_mu, c_sigma, cfg, args, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
