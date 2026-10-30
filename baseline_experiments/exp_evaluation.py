"""
Evaluation for Experiment 1 (and any future baseline_experiments/experiment_N
that saves outputs in the same layout). Consumes SAVED outputs only -- never
retrains or resamples a model. Reads:

    <experiment-dir>/shared/test_data.npz          real test windows + conditioning values
    <experiment-dir>/<MODEL_DIR>/generated_samples.npy   (N_test, K, A, T) raw log returns
    <experiment-dir>/<MODEL_DIR>/loss_history.csv

Metrics (Group A: distributional fidelity, Group B: portfolio-level
financial fidelity), all computed on identical test windows/samples across
models:
    A1  Energy score (lower is better)
    A2  Per-asset Wasserstein-1 of last-day marginals, averaged over 15 assets
    B1  10-day equal-weight portfolio variance: Wasserstein-1(real,gen) + |mean diff|
    B2  10-day equal-weight portfolio max drawdown: Wasserstein-1(real,gen) + |mean diff|
    B3  Portfolio tail quantile pinball loss at q in {0.01, 0.05, 0.95, 0.99}

Usage:
    python exp_evaluation.py --experiment-dir experiment_1/exp1_results
    python exp_evaluation.py --experiment-dir experiment_1/exp1_results --models m0,m4
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

MODEL_DIRNAMES = {
    "m0": "M0_unconditional",
    "m1": "M1_ewma_mean",
    "m2": "M2_ewma_volatility",
    "m3": "M3_rolling_pca",
    "m4": "M4_macro_ssm",
}
MODEL_LABELS = {
    "m0": "M0 Unconditional",
    "m1": "M1 EWMA Mean",
    "m2": "M2 EWMA Volatility",
    "m3": "M3 Rolling PCA",
    "m4": "M4 Macro SSM",
}
QUANTILES = (0.01, 0.05, 0.95, 0.99)


# ─────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────

def load_shared_test_data(experiment_dir):
    path = os.path.join(experiment_dir, "shared", "test_data.npz")
    data = np.load(path, allow_pickle=True)
    return data["X_test"]  # (N_test, A, T) raw log returns


def load_generated(experiment_dir, model_key, model_dirnames=None):
    # Defaults to the module constant -- Experiment 1's existing calls (no
    # new argument) stay byte-identical. Experiment 2 passes its own S0-S4
    # dirname mapping explicitly.
    model_dirnames = model_dirnames or MODEL_DIRNAMES
    out_dir = os.path.join(experiment_dir, model_dirnames[model_key])
    gen_path = os.path.join(out_dir, "generated_samples.npy")
    if not os.path.exists(gen_path):
        return None, out_dir
    return np.load(gen_path), out_dir


# ─────────────────────────────────────────────────────────────────────────
# Group A — distributional fidelity
# ─────────────────────────────────────────────────────────────────────────

def energy_score(real_window: np.ndarray, gen_windows: np.ndarray) -> float:
    """real_window: (A,T). gen_windows: (K,A,T)."""
    K = gen_windows.shape[0]
    real_flat = real_window.reshape(-1)
    gen_flat = gen_windows.reshape(K, -1)

    term1 = np.linalg.norm(gen_flat - real_flat[None, :], axis=1).mean()
    if K > 1:
        diffs = gen_flat[:, None, :] - gen_flat[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        term2 = dists.sum() / (2 * K * (K - 1))
    else:
        term2 = 0.0
    return float(term1 - term2)


def mean_energy_score(X_test: np.ndarray, gen: np.ndarray) -> float:
    n_test = min(len(X_test), len(gen))
    scores = [energy_score(X_test[i], gen[i]) for i in range(n_test)]
    return float(np.mean(scores))


def mean_wasserstein_marginals(X_test: np.ndarray, gen: np.ndarray) -> float:
    """Average over assets of W1(real last-day returns, generated last-day
    returns pooled across K samples). Same last-day-marginal convention as
    analysis/distribution_metrics.py."""
    n_test = min(len(X_test), len(gen))
    A = X_test.shape[1]
    ws = []
    for a in range(A):
        real_vals = X_test[:n_test, a, -1]
        gen_vals = gen[:n_test, :, a, -1].reshape(-1)
        ws.append(wasserstein_distance(real_vals, gen_vals))
    return float(np.mean(ws))


# ─────────────────────────────────────────────────────────────────────────
# Group B — portfolio-level financial fidelity (equal weight, raw log returns)
# ─────────────────────────────────────────────────────────────────────────

def portfolio_log_returns(window: np.ndarray) -> np.ndarray:
    """window: (A,T) raw log returns -> (T,) equal-weight portfolio log returns."""
    return window.mean(axis=0)


def portfolio_variance_legacy(window: np.ndarray) -> float:
    """Original implementation -- CONFIRMED (Experiment 2 review) inconsistent
    with portfolio_wealth_path (and therefore with max_drawdown/
    portfolio_cum_return): this is Var(mean_a log_r[a,t]), the variance of
    the ARITHMETIC MEAN of each asset's log return, while
    portfolio_wealth_path builds the TRUE daily-rebalanced wealth path from
    mean_a(exp(log_r[a,t])-1) -- the log-return of the MEAN simple return.
    By Jensen's inequality these are different series whenever assets'
    daily returns differ (always, in practice). Kept only so already-computed
    Experiment 1 tables stay reproducible under the definition they were
    actually computed with -- use portfolio_variance_v2 for anything new,
    and recompute old tables under v2 before comparing them to new ones."""
    port_r = portfolio_log_returns(window)
    return float(np.var(port_r, ddof=1))


def portfolio_variance_v2(window: np.ndarray) -> float:
    """Corrected: variance of the log-returns of the SAME daily-rebalanced
    wealth path that max_drawdown/portfolio_cum_return already use (via
    portfolio_wealth_path), instead of an independently (and inconsistently)
    defined series."""
    wealth = portfolio_wealth_path(window)
    port_log_r = np.diff(np.log(wealth))
    return float(np.var(port_log_r, ddof=1))


# Default unchanged -- every existing call site (Experiment 1's own CLI,
# with no new arguments) stays byte-identical in behavior.
portfolio_variance = portfolio_variance_legacy


def portfolio_wealth_path(window: np.ndarray) -> np.ndarray:
    """Equal-weight, daily-rebalanced wealth path from SIMPLE returns implied
    by the stocks' log returns. window: (A,T) -> wealth: (T+1,), wealth[0]=1."""
    simple_r = np.exp(window) - 1.0          # (A,T)
    port_simple_r = simple_r.mean(axis=0)    # (T,) equal-weight, daily rebalanced
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + port_simple_r)])
    return wealth


def max_drawdown(window: np.ndarray) -> float:
    wealth = portfolio_wealth_path(window)
    peak = np.maximum.accumulate(wealth)
    dd = (peak - wealth) / peak
    return float(dd.max())


def portfolio_cum_return(window: np.ndarray) -> float:
    wealth = portfolio_wealth_path(window)
    return float(wealth[-1] - 1.0)


def pinball_loss(y: float, q_hat: float, q: float) -> float:
    return q * (y - q_hat) if y >= q_hat else (1 - q) * (q_hat - y)


def group_b_metrics(X_test: np.ndarray, gen: np.ndarray, quantiles=QUANTILES, variance_fn=None):
    # variance_fn defaults to the legacy function -- Experiment 1's existing
    # calls (no new argument) stay byte-identical. Experiment 2 passes
    # portfolio_variance_v2 explicitly.
    variance_fn = variance_fn or portfolio_variance
    n_test = min(len(X_test), len(gen))

    var_real = np.array([variance_fn(X_test[i]) for i in range(n_test)])
    var_gen = np.array([variance_fn(gen[i, k]) for i in range(n_test) for k in range(gen.shape[1])])

    mdd_real = np.array([max_drawdown(X_test[i]) for i in range(n_test)])
    mdd_gen = np.array([max_drawdown(gen[i, k]) for i in range(n_test) for k in range(gen.shape[1])])

    pinball = {q: [] for q in quantiles}
    for i in range(n_test):
        gen_cum_returns = np.array([portfolio_cum_return(gen[i, k]) for k in range(gen.shape[1])])
        y = portfolio_cum_return(X_test[i])
        for q in quantiles:
            q_hat = float(np.quantile(gen_cum_returns, q))
            pinball[q].append(pinball_loss(y, q_hat, q))
    pinball_avg = {q: float(np.mean(v)) for q, v in pinball.items()}

    return {
        "var_real": var_real, "var_gen": var_gen,
        "mdd_real": mdd_real, "mdd_gen": mdd_gen,
        "E_variance_w1": float(wasserstein_distance(var_real, var_gen)),
        "E_variance_meandiff": float(abs(var_gen.mean() - var_real.mean())),
        "E_mdd_w1": float(wasserstein_distance(mdd_real, mdd_gen)),
        "E_mdd_meandiff": float(abs(mdd_gen.mean() - mdd_real.mean())),
        "pinball": pinball_avg,
    }


# ─────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────

def evaluate_model(experiment_dir, model_key, X_test, quantiles, model_dirnames=None, model_labels=None,
                    variance_fn=None):
    # Both default to the module constants -- Experiment 1's existing calls
    # (no new arguments) stay byte-identical. Experiment 2 passes its own
    # S0-S4 dirname/label mappings and portfolio_variance_v2 explicitly.
    model_dirnames = model_dirnames or MODEL_DIRNAMES
    model_labels = model_labels or MODEL_LABELS
    gen, out_dir = load_generated(experiment_dir, model_key, model_dirnames)
    if gen is None:
        print(f"[skip] {model_key}: no generated_samples.npy in {out_dir}")
        return None

    n_nan = np.isnan(gen).sum() + np.isinf(gen).sum()
    if n_nan > 0:
        raise ValueError(f"{model_key}: generated samples contain {n_nan} NaN/Inf values.")

    es = mean_energy_score(X_test, gen)
    w1 = mean_wasserstein_marginals(X_test, gen)
    b = group_b_metrics(X_test, gen, quantiles, variance_fn=variance_fn)

    row = {
        "model": model_labels[model_key],
        "energy_score": es,
        "wasserstein1_marginal": w1,
        "portfolio_variance_w1": b["E_variance_w1"],
        "portfolio_variance_meandiff": b["E_variance_meandiff"],
        "portfolio_mdd_w1": b["E_mdd_w1"],
        "portfolio_mdd_meandiff": b["E_mdd_meandiff"],
    }
    for q in quantiles:
        row[f"pinball_{q}"] = b["pinball"][q]

    return row, gen, b, out_dir


def plot_loss_curves(experiment_dir, model_keys, output_dir):
    plt.figure(figsize=(8, 5))
    any_curve = False
    for mk in model_keys:
        path = os.path.join(experiment_dir, MODEL_DIRNAMES[mk], "loss_history.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        plt.plot(df["epoch"], df["loss"], label=MODEL_LABELS[mk])
        any_curve = True
    if any_curve:
        plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.legend()
        plt.title("Experiment 1 — Training Loss Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()


def plot_distributions(output_dir, results_by_model):
    # Real vs generated portfolio variance / MDD, one panel per available model
    models = list(results_by_model.keys())
    if not models:
        return

    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3 * len(models)), squeeze=False)
    for i, mk in enumerate(models):
        b = results_by_model[mk]["b"]
        axes[i, 0].hist(b["var_real"], bins=20, alpha=0.6, label="Real", density=True)
        axes[i, 0].hist(b["var_gen"], bins=20, alpha=0.6, label="Generated", density=True)
        axes[i, 0].set_title(f"{MODEL_LABELS[mk]} — 10-day portfolio variance")
        axes[i, 0].legend(fontsize=8)

        axes[i, 1].hist(b["mdd_real"], bins=20, alpha=0.6, label="Real", density=True)
        axes[i, 1].hist(b["mdd_gen"], bins=20, alpha=0.6, label="Generated", density=True)
        axes[i, 1].set_title(f"{MODEL_LABELS[mk]} — 10-day max drawdown")
        axes[i, 1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "portfolio_distributions.png"), dpi=150)
    plt.close()


def plot_marginals(output_dir, X_test, results_by_model, asset_idx=0):
    models = list(results_by_model.keys())
    if not models:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(X_test[:, asset_idx, -1], bins=30, alpha=0.5, label="Real", density=True)
    for mk in models:
        gen = results_by_model[mk]["gen"]
        n = min(len(X_test), len(gen))
        plt.hist(gen[:n, :, asset_idx, -1].reshape(-1), bins=30, alpha=0.4,
                 label=MODEL_LABELS[mk], density=True)
    plt.title(f"Real vs Generated last-day return marginal (asset idx {asset_idx})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_comparison.png"), dpi=150)
    plt.close()


def plot_metric_bars(output_dir, table: pd.DataFrame):
    metrics = ["energy_score", "wasserstein1_marginal", "portfolio_variance_w1", "portfolio_mdd_w1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        ax.bar(table["model"], table[m])
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison_bars.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=str, required=True,
                         help="e.g. experiment_1/exp1_results (relative to baseline_experiments/, or absolute)")
    parser.add_argument("--models", type=str, default="m0,m1,m2,m3,m4")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = args.experiment_dir if os.path.isabs(args.experiment_dir) \
        else os.path.join(_here, args.experiment_dir)
    output_dir = args.output_dir or os.path.join(experiment_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    X_test = load_shared_test_data(experiment_dir)
    print(f"Loaded shared test data: {X_test.shape}")

    rows = []
    results_by_model = {}
    for mk in model_keys:
        result = evaluate_model(experiment_dir, mk, X_test, QUANTILES)
        if result is None:
            continue
        row, gen, b, out_dir = result
        rows.append(row)
        results_by_model[mk] = {"gen": gen, "b": b}
        print(f"{mk}: energy_score={row['energy_score']:.4f}  "
              f"W1_marginal={row['wasserstein1_marginal']:.4f}  "
              f"var_W1={row['portfolio_variance_w1']:.6f}  mdd_W1={row['portfolio_mdd_w1']:.4f}")

    if not rows:
        print("No models had generated samples to evaluate. Nothing to save.")
        return

    table = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    table.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table -> {csv_path}")
    print(table.to_string(index=False))

    plot_loss_curves(experiment_dir, list(results_by_model.keys()), output_dir)
    plot_distributions(output_dir, results_by_model)
    plot_marginals(output_dir, X_test, results_by_model)
    plot_metric_bars(output_dir, table)
    print(f"Saved plots -> {output_dir}")


if __name__ == "__main__":
    main()
