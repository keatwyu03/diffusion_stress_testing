"""
Experiment 1: does conditioning the diffusion model on a causal macro latent
state (M4) produce better generated return distributions than conventional
stock-based conditioning (M1-M3) or no conditioning at all (M0)?

Self-contained data pipeline: does NOT use data/data_processor.py. Reads
explore/macro_data_new.csv directly and trains on RAW daily log returns (no
EWMA/winsorize/macro-conditional standardization). Reuses config/config.py's
existing dates, train/test split, and Transformer/VP-SDE hyperparameters
unchanged. Does not rerun explore/import_data.py or refit the SSM -- M4's
conditioning series is read straight from the existing macro_data_new.csv
column (see latent_state_estimation/ for how it was produced).

Models:
    M0  unconditional
    M1  EWMA mean of equal-weight stock returns
    M2  EWMA volatility of equal-weight stock returns
    M3  rolling first-PC factor of stock returns
    M4  existing SSM macro latent state (m_t)

Usage:
    python main_exp1.py --model all --mode both --epochs 3 --n-test-windows 5 --n-samples 4 --num-steps 50
    python main_exp1.py --model m4 --mode train --epochs 200
    python main_exp1.py --model m4 --mode sample --n-samples 500
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # baseline_experiments/experiment_1 -> repo root
sys.path.insert(0, _ROOT)

from config import get_default_config
from models import DiffusionModel
from utils import set_seed

TICKERS = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "TXN", "QCOM", "AMAT", "ADBE",
           "KO", "PG", "CL", "GIS", "KMB"]

MODEL_KEYS = ["m0", "m1", "m2", "m3", "m4"]
MODEL_DIRNAMES = {
    "m0": "M0_unconditional",
    "m1": "M1_ewma_mean",
    "m2": "M2_ewma_volatility",
    "m3": "M3_rolling_pca",
    "m4": "M4_macro_ssm",
}
COND_KEY = {"m1": "ewma_mean", "m2": "ewma_vol", "m3": "pca", "m4": "macro_change"}


# ─────────────────────────────────────────────────────────────────────────
# 1. Data loading — raw returns + macro column, no DataProcessor
# ─────────────────────────────────────────────────────────────────────────

def load_raw_data(cfg):
    csv_path = os.path.join(_ROOT, "explore", "macro_data_new.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")

    macro_col = df.columns[0]  # first column after Date (same convention as DataProcessor)
    missing = [t for t in TICKERS if t not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required tickers {missing}. Experiment 1 "
            "requires the existing 15-asset universe; refusing to silently "
            "fall back to a different asset set."
        )

    df = df[[macro_col] + TICKERS]
    if cfg.data.start_date:
        df = df[df.index >= pd.to_datetime(cfg.data.start_date)]
    if cfg.data.end_date:
        df = df[df.index <= pd.to_datetime(cfg.data.end_date)]
    # Assets must be complete at every row; the macro column may be sparse.
    df = df.dropna(subset=TICKERS)
    print(f"Loaded {csv_path}: {df.shape[0]} rows, macro column = {macro_col!r}, "
          f"{len(TICKERS)} tickers present.")
    return df, macro_col


# ─────────────────────────────────────────────────────────────────────────
# 2. Conditioning variables (all causal: value known at/through day t-1 for
#    a window starting at day t)
# ─────────────────────────────────────────────────────────────────────────

def ewma_finite_mean_vol(R: pd.Series, L: int, alpha: float):
    """Finite-window EWMA mean/vol (not pandas' recursive .ewm()), per spec:
    C_t = (1-a)/(1-a^L) * sum_{j=0}^{L-1} a^j R_{t-j}, and the matching
    weighted-variance formula (not the zero-mean approximation)."""
    weights = alpha ** np.arange(L - 1, -1, -1)  # oldest..newest within window
    norm = (1 - alpha) / (1 - alpha ** L)

    def mean_f(w):
        return norm * np.dot(w, weights)

    def var_f(w):
        mu = norm * np.dot(w, weights)
        return norm * np.dot(weights, (w - mu) ** 2)

    mean_t = R.rolling(L).apply(mean_f, raw=True)
    var_t = R.rolling(L).apply(var_f, raw=True)
    return mean_t, np.sqrt(var_t)


def rolling_pca_factor(returns_df: pd.DataFrame, L: int):
    """C_t = w1_t^T (r_t - mu_t), where mu_t/w1_t (top eigenvector of the
    sample covariance) are estimated from the PRECEDING L days only (not
    including day t). Sign of w1_t is aligned to w1_{t-1} by inner product
    (continuity); the first one is anchored to the equal-weight direction."""
    R = returns_df.to_numpy(dtype=np.float64)
    N, A = R.shape
    out = np.full(N, np.nan)
    market_dir = np.ones(A) / np.sqrt(A)
    prev_w = None
    for t in range(L, N):
        hist = R[t - L:t]
        mu = hist.mean(axis=0)
        X = hist - mu
        sigma = (X.T @ X) / (L - 1)
        eigvals, eigvecs = np.linalg.eigh(sigma)
        w1 = eigvecs[:, -1]  # eigh returns ascending eigenvalues -> last is largest
        ref = prev_w if prev_w is not None else market_dir
        if np.dot(w1, ref) < 0:
            w1 = -w1
        prev_w = w1
        out[t] = w1 @ (R[t] - mu)
    return pd.Series(out, index=returns_df.index)


def compute_conditioning_variables(df: pd.DataFrame, macro_col: str, L_ewma: int, L_pca: int, alpha: float,
                                    macro_change_window: int) -> pd.DataFrame:
    equal_weight_r = df[TICKERS].mean(axis=1)
    ewma_mean, ewma_vol = ewma_finite_mean_vol(equal_weight_r, L=L_ewma, alpha=alpha)
    pca = rolling_pca_factor(df[TICKERS], L=L_pca)
    macro_level = df[macro_col]

    # M4 conditions on the CHANGE in the macro state over a window, not its
    # raw level -- the raw level trends over decades (train/test z-score
    # ranges barely overlap), while a windowed change is far more stable,
    # mirroring the production pipeline's own event metric
    # (event_type="abs_change": |Z_end - Z_start| over event_window days,
    # which equals config.data.seq_len by convention).
    macro_change = (macro_level - macro_level.shift(macro_change_window - 1)).abs()

    return pd.DataFrame({
        "ewma_mean": ewma_mean,
        "ewma_vol": ewma_vol,
        "pca": pca,
        "macro_level": macro_level,
        "macro_change": macro_change,
    }, index=df.index)


# ─────────────────────────────────────────────────────────────────────────
# 3. Window construction (common sample set across all 5 models)
# ─────────────────────────────────────────────────────────────────────────

def build_windows(df: pd.DataFrame, cond_df: pd.DataFrame, seq_len: int):
    R = df[TICKERS].to_numpy(dtype=np.float64)
    N = len(df)
    dates = df.index

    valid_cond = cond_df[["ewma_mean", "pca", "macro_change"]].notna().all(axis=1).to_numpy()

    starts = [i for i in range(1, N - seq_len + 1) if valid_cond[i - 1]]
    starts = np.array(starts, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("No valid windows after applying the common lookback/macro-availability mask.")

    X = np.stack([R[i:i + seq_len].T for i in starts])  # (Nw, A, T) raw log returns
    cond_row = starts - 1
    cond_vals = {k: cond_df[k].to_numpy()[cond_row]
                 for k in ["ewma_mean", "ewma_vol", "pca", "macro_level", "macro_change"]}

    window_start_dates = dates[starts]
    window_end_dates = dates[starts + seq_len - 1]
    conditioning_dates = dates[cond_row]

    print(f"Built {len(starts)} common windows (seq_len={seq_len}) out of "
          f"{N - seq_len} possible start positions "
          f"({(~valid_cond[:N - seq_len]).sum()} dropped for missing lookback/macro data).")
    return X, cond_vals, window_start_dates, window_end_dates, conditioning_dates, starts


def train_test_split_idx(starts: np.ndarray, window_end_dates: pd.DatetimeIndex, seq_len: int, cfg):
    """Train boundary follows DataProcessor.train_test_split()'s rule exactly
    (train_end_date cutoff, else last test_days windows). The TEST start index
    is then pushed forward past any window whose raw-return rows overlap the
    last train window's rows -- sliding windows with shift=1 share seq_len-1
    rows with their neighbors, so without this gap a few boundary windows
    would contain return observations present in both the train and test
    sets. Train windows are left untouched; only boundary test windows are
    dropped, so the config.py test_days/train_end_date setting is preserved
    exactly on the train side and only minimally reduced on the test side."""
    if cfg.data.train_end_date is not None:
        cutoff = pd.to_datetime(cfg.data.train_end_date)
        split_idx = int((window_end_dates <= cutoff).sum())
    else:
        split_idx = len(window_end_dates) - cfg.data.test_days

    if split_idx <= 0 or split_idx >= len(starts):
        return split_idx, split_idx

    last_train_end_row = starts[split_idx - 1] + seq_len - 1
    test_starts = starts[split_idx:]
    n_overlap = int((test_starts <= last_train_end_row).sum())
    test_start_idx = split_idx + n_overlap
    if n_overlap > 0:
        print(f"Dropped {n_overlap} boundary test window(s) that overlapped the last "
              f"train window's return rows (no train/test row overlap).")
    return split_idx, test_start_idx


# ─────────────────────────────────────────────────────────────────────────
# 4. Per-model train / sample
# ─────────────────────────────────────────────────────────────────────────

def fit_scale_factor(X_train: np.ndarray) -> float:
    """One global reversible scalar (not per-asset/per-window) so the raw
    log-return data has roughly unit variance, matching the existing VP-SDE
    schedule (b_min/b_max) which was tuned for that regime. Fit on train only."""
    std = float(X_train.std())
    return 1.0 / std if std > 0 else 1.0


def build_diffusion_model(cfg, seq_len: int) -> DiffusionModel:
    return DiffusionModel(
        in_channels=len(TICKERS),
        out_channels=len(TICKERS),
        sample_size=seq_len,
        b_min=cfg.diffusion.b_min,
        b_max=cfg.diffusion.b_max,
        device=cfg.diffusion.device,
        arch="transformer",
        embed_dim=cfg.diffusion.embed_dim,
        n_heads=cfg.diffusion.n_heads,
        n_layers=cfg.diffusion.n_layers,
        cond_dim=cfg.diffusion.cond_dim,
        cov_weight=0.0,  # no covariance penalty for this experiment (spec sec. 7)
        cov_t_max=cfg.diffusion.cov_t_max,
    )


def get_condition_array(model_key: str, cond_vals: dict):
    if model_key == "m0":
        return None
    return cond_vals[COND_KEY[model_key]]


def architecture_signature(cfg, seq_len: int, n_assets: int) -> dict:
    return {
        "n_assets": n_assets, "seq_len": seq_len,
        "embed_dim": cfg.diffusion.embed_dim, "n_heads": cfg.diffusion.n_heads,
        "n_layers": cfg.diffusion.n_layers, "cond_dim": cfg.diffusion.cond_dim,
        "b_min": cfg.diffusion.b_min, "b_max": cfg.diffusion.b_max,
    }


def conditioning_signature(model_key: str, args) -> dict:
    """Which conditioning-feature knobs affect this model's conditioning
    series, so a checkpoint trained under a different --ewma-lookback or
    --pca-lookback is never silently loaded as if it still matched."""
    if model_key == "m3":
        return {"pca_lookback": args.pca_lookback}
    if model_key in ("m1", "m2"):
        return {"ewma_lookback": args.ewma_lookback}
    if model_key == "m4":
        return {"macro_change_window": args.macro_change_window}
    return {}


def validate_checkpoint_compatibility(out_dir, model_key, seq_len, n_assets, cfg, scale, args) -> dict:
    """Refuses to load a checkpoint whose architecture, conditioning-feature
    config, or target scaling factor doesn't match the current run, instead
    of silently loading something inconsistent. Raises RuntimeError on any
    mismatch; the fix is to pass --retrain."""
    meta_path = os.path.join(out_dir, "checkpoint_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"{model_key}: checkpoint.pt exists at {out_dir} but checkpoint_meta.json "
            "is missing, so compatibility can't be validated. Pass --retrain."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    mismatches = []
    expected_arch = architecture_signature(cfg, seq_len, n_assets)
    for k, v in expected_arch.items():
        if meta.get("architecture", {}).get(k) != v:
            mismatches.append(f"architecture.{k}: saved={meta.get('architecture', {}).get(k)!r} current={v!r}")

    expected_cond = conditioning_signature(model_key, args)
    for k, v in expected_cond.items():
        if meta.get("conditioning_feature", {}).get(k) != v:
            mismatches.append(f"conditioning_feature.{k}: saved={meta.get('conditioning_feature', {}).get(k)!r} current={v!r}")

    saved_scale = meta.get("scale_factor")
    if saved_scale is None or abs(saved_scale - scale) > 1e-6 * max(1.0, abs(scale)):
        mismatches.append(f"scale_factor: saved={saved_scale!r} current={scale!r}")

    if mismatches:
        raise RuntimeError(
            f"{model_key}: existing checkpoint at {out_dir} is INCOMPATIBLE with the "
            f"current run -- {'; '.join(mismatches)}. Pass --retrain to retrain it."
        )
    return meta


def train_model(model_key, X_train_raw, cond_train_all, scale, args, cfg, out_dir):
    set_seed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    raw_cond = get_condition_array(model_key, cond_train_all)
    norm_stats = None
    cond_tensor = None
    if raw_cond is not None:
        mu, sigma = float(raw_cond.mean()), float(raw_cond.std())
        norm_stats = {"mu": mu, "sigma": sigma}
        z = (raw_cond - mu) / sigma
        cond_tensor = torch.tensor(z, dtype=torch.float32)

    X_train_scaled = torch.tensor(X_train_raw * scale, dtype=torch.float32)

    dm = build_diffusion_model(cfg, seq_len=X_train_raw.shape[-1])
    dm.train(
        train_data=X_train_scaled,
        condition_data=cond_tensor,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=cfg.diffusion.learning_rate,
        weight_decay=cfg.diffusion.weight_decay,
        scheduler_patience=cfg.diffusion.scheduler_patience,
        scheduler_factor=cfg.diffusion.scheduler_factor,
        block_sampling=False,
        loss_csv_path=os.path.join(out_dir, "loss_history.csv"),
    )

    dm.save(os.path.join(out_dir, "checkpoint.pt"))
    seq_len, n_assets = X_train_raw.shape[-1], X_train_raw.shape[1]
    meta = {
        "model": model_key,
        "architecture": architecture_signature(cfg, seq_len, n_assets),
        "conditioning_feature": conditioning_signature(model_key, args),
        "scale_factor": scale,
        "condition_norm_stats": norm_stats,
    }
    with open(os.path.join(out_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(vars(args) | {"model": model_key, "seq_len": seq_len}, f, indent=2, default=str)

    return dm, norm_stats


def load_model(model_key, seq_len, n_assets, cfg, scale, args, out_dir):
    meta = validate_checkpoint_compatibility(out_dir, model_key, seq_len, n_assets, cfg, scale, args)
    dm = build_diffusion_model(cfg, seq_len=seq_len)
    dm.load(os.path.join(out_dir, "checkpoint.pt"))
    return dm, meta["condition_norm_stats"], meta["scale_factor"]


def sample_model(model_key, dm, norm_stats, scale, cond_test_all, args, out_dir):
    raw_cond_full = get_condition_array(model_key, cond_test_all)
    n_available = len(next(iter(cond_test_all.values())))
    n_test = min(args.n_test_windows, n_available) if args.n_test_windows else n_available

    K = args.n_samples
    A, T = dm.in_channels, dm.sample_size
    total = n_test * K

    cond_flat = None
    if raw_cond_full is not None:
        z = (raw_cond_full[:n_test] - norm_stats["mu"]) / norm_stats["sigma"]
        cond_per_window = torch.tensor(z, dtype=torch.float32)       # (n_test,)
        cond_flat = cond_per_window.repeat_interleave(K)             # (n_test*K,)

    # Batched by TOTAL trajectories (n_test*K), not by n_test alone -- a large
    # K would otherwise blow args.batch_size up to n_w*K per GPU call. Each
    # chunk here is at most args.batch_size actual trajectories.
    flat_samples = np.zeros((total, A, T), dtype=np.float32)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        chunk_cond = cond_flat[start:end] if cond_flat is not None else None
        samp_scaled = dm.sample(
            batch_size=end - start,
            num_steps=args.num_steps,
            stoch=1.0,
            condition=chunk_cond,
            seed=args.seed + start,
        )
        flat_samples[start:end] = samp_scaled.detach().cpu().numpy()

    all_samples_raw = (flat_samples.reshape(n_test, K, A, T) / scale).astype(np.float32)

    n_nan = np.isnan(all_samples_raw).sum() + np.isinf(all_samples_raw).sum()
    if n_nan > 0:
        raise RuntimeError(f"{model_key}: generated samples contain {n_nan} NaN/Inf values.")

    np.save(os.path.join(out_dir, "generated_samples.npy"), all_samples_raw)
    print(f"{model_key}: saved generated samples {all_samples_raw.shape} -> "
          f"{os.path.join(out_dir, 'generated_samples.npy')}")
    return all_samples_raw, n_test


# ─────────────────────────────────────────────────────────────────────────
# 5. CLI entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    cfg = get_default_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS + ["all"], default="all")
    parser.add_argument("--mode", choices=["train", "sample", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=cfg.diffusion.n_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.diffusion.batch_size)
    parser.add_argument("--num-steps", type=int, default=cfg.diffusion.num_steps)
    parser.add_argument("--n-samples", type=int, default=50, help="K trajectories per test window")
    parser.add_argument("--n-test-windows", type=int, default=0, help="0 = use all test windows")
    parser.add_argument("--ewma-lookback", type=int, default=10)
    parser.add_argument("--pca-lookback", type=int, default=10)
    parser.add_argument("--macro-change-window", type=int, default=cfg.data.seq_len,
                         help="M4 conditions on |m_t[end] - m_t[end-window+1]|, not the raw level "
                              "(default: same window as the return window / production event_window)")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--output-dir", type=str, default=os.path.join(_HERE, "exp1_results"))
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if a checkpoint exists")
    args = parser.parse_args()

    set_seed(args.seed)

    df, macro_col = load_raw_data(cfg)
    cond_df = compute_conditioning_variables(
        df, macro_col, L_ewma=args.ewma_lookback, L_pca=args.pca_lookback, alpha=59 / 61,
        macro_change_window=args.macro_change_window,
    )
    X, cond_vals, w_start, w_end, c_dates, starts = build_windows(df, cond_df, seq_len=cfg.data.seq_len)
    split_idx, test_start_idx = train_test_split_idx(starts, w_end, cfg.data.seq_len, cfg)
    print(f"Train/test split: {split_idx} train windows, {len(X) - test_start_idx} test windows "
          f"(test_days={cfg.data.test_days}, train_end_date={cfg.data.train_end_date}, "
          f"{test_start_idx - split_idx} boundary window(s) excluded to avoid train/test row overlap)")

    X_train, X_test = X[:split_idx], X[test_start_idx:]
    cond_train = {k: v[:split_idx] for k, v in cond_vals.items()}
    cond_test = {k: v[test_start_idx:] for k, v in cond_vals.items()}

    shared_dir = os.path.join(args.output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    scale_path = os.path.join(shared_dir, "scale_factor.json")
    if os.path.exists(scale_path) and not args.retrain:
        with open(scale_path) as f:
            scale = json.load(f)["scale_factor"]
    else:
        scale = fit_scale_factor(X_train)
        with open(scale_path, "w") as f:
            json.dump({"scale_factor": scale, "train_std": float(X_train.std())}, f, indent=2)
    print(f"Numerical scale factor (1/train_std): {scale:.4f}")

    test_data_path = os.path.join(shared_dir, "test_data.npz")
    np.savez(
        test_data_path,
        X_test=X_test,
        window_start_dates=w_start[test_start_idx:].astype(str),
        window_end_dates=w_end[test_start_idx:].astype(str),
        conditioning_dates=c_dates[test_start_idx:].astype(str),
        **{f"cond_{k}": v for k, v in cond_test.items()},
    )
    print(f"Saved shared test dataset ({X_test.shape}) -> {test_data_path}")

    models_to_run = MODEL_KEYS if args.model == "all" else [args.model]

    for model_key in models_to_run:
        out_dir = os.path.join(args.output_dir, MODEL_DIRNAMES[model_key])
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        print(f"\n{'=' * 60}\n{model_key.upper()} ({MODEL_DIRNAMES[model_key]})\n{'=' * 60}")

        seq_len, n_assets = X_train.shape[-1], X_train.shape[1]
        if args.mode in ("train", "both"):
            if args.retrain or not os.path.exists(ckpt_path):
                dm, norm_stats = train_model(model_key, X_train, cond_train, scale, args, cfg, out_dir)
            else:
                print(f"Found existing checkpoint {ckpt_path}, validating and loading...")
                dm, norm_stats, scale = load_model(model_key, seq_len, n_assets, cfg, scale, args, out_dir)
        else:
            dm, norm_stats, scale = load_model(model_key, seq_len, n_assets, cfg, scale, args, out_dir)

        if args.mode in ("sample", "both"):
            sample_model(model_key, dm, norm_stats, scale, cond_test, args, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
