"""
Experiment 2 CLI entry point. Dispatches to the already-built, independently
smoke-tested modules (each also runnable standalone):
    daily_macro_states.py    (states)
    macro_standardization.py (standardize)
    stationarity_evaluation.py (stationarity)
    diffusion_adapters.py    (train-diffusion / sample)
and implements `evaluate` (Group B/C metrics, via exp_evaluation.py's
generic functions + S0-S4 overrides added there) and `demo` (a short,
lightweight end-to-end chain through all stages for 2 methods -- NOT a full
run; per the approved plan, this experiment pass stops at smoke-test-level
validation).

Separate modes exist specifically so costly stages (standardizer training,
diffusion training) are never repeated unnecessarily -- each mode loads
already-saved outputs from the previous stage rather than recomputing them.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import common

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.dirname(_HERE)
sys.path.insert(0, _BASELINE)
sys.path.insert(0, os.path.join(_BASELINE, "macro_inclusion"))

from exp_evaluation import (  # noqa: E402
    evaluate_model, portfolio_variance_v2, QUANTILES,
)

PY = sys.executable
OUT_DIR = os.path.join(os.path.dirname(_HERE), "main_results", "exp2_results")  # demo/smoke-test only
from window_conditioning import EXPERIMENTS, experiment_paths  # noqa: E402


def run(script, extra_args):
    cmd = [PY, "-u", os.path.join(_HERE, script)] + extra_args
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_HERE)


def mode_evaluate(methods, results_dir=OUT_DIR):
    """Group B/C: reuses exp_evaluation.py's generic energy_score/wasserstein/
    group_b_metrics (unmodified mathematical definitions, per spec) via the
    additive model_dirnames/model_labels/variance_fn overrides added there --
    Experiment 1's own calls are unaffected (confirmed, regression-checked).
    Uses portfolio_variance_v2 (the corrected, daily-rebalanced-consistent
    variance) for Experiment 2, since this is a fresh evaluation, not a
    reproduction of an already-published Experiment 1 number."""
    # Conditional-only is causal-only; diffusion_adapters.py writes <method>/diffusion_causal/.
    dirnames = {m: os.path.join(common.METHOD_DIRNAMES[m], "diffusion_causal") for m in methods}
    labels = {m: common.METHOD_LABELS[m] for m in methods}

    rows = []
    for m in methods:
        eval_npz = np.load(os.path.join(results_dir, dirnames[m], "eval_data.npz"))
        X_test = eval_npz["X_eval"]  # in-sample train windows the samples were conditioned on
        print(f"  {m}: {len(X_test)} eval windows (split={eval_npz['eval_split']})")
        result = evaluate_model(results_dir, m, X_test, QUANTILES, model_dirnames=dirnames,
                                 model_labels=labels, variance_fn=portfolio_variance_v2)
        if result is None:
            continue
        row, gen, b, out_dir = result
        row["retrospective"] = common.RETROSPECTIVE[m]
        rows.append(row)
        print(f"  {m}: energy_score={row['energy_score']:.4f}  W1_marginal={row['wasserstein1_marginal']:.4f}  "
              f"var_W1={row['portfolio_variance_w1']:.6f}  mdd_W1={row['portfolio_mdd_w1']:.4f}  "
              f"retrospective={row['retrospective']}")

    if not rows:
        print("No methods had generated samples to evaluate.")
        return
    table = pd.DataFrame(rows)

    # Per the approved plan: causal-ish (S0/S2/S4) and retrospective (S1/S3)
    # results are reported SEPARATELY, never merged into one forecasting
    # comparison -- interpolation baselines used a future monthly anchor.
    causal_table = table[~table["retrospective"]]
    retro_table = table[table["retrospective"]]

    causal_path = os.path.join(results_dir, "group_bc_comparison_causal.csv")
    retro_path = os.path.join(results_dir, "group_bc_comparison_retrospective.csv")
    causal_table.to_csv(causal_path, index=False)
    retro_table.to_csv(retro_path, index=False)
    print(f"\nSaved -> {causal_path} (causal-ish: S0/S2/S4, still full-sample-fit preprocessing -- see state_meta.json)")
    print(causal_table.to_string(index=False))
    print(f"\nSaved -> {retro_path} (RETROSPECTIVE, uses future monthly anchors -- not a fair forecast comparison)")
    print(retro_table.to_string(index=False))


def mode_demo():
    """Short, lightweight chain through every stage for 2 methods (one
    causal-ish, one retrospective) -- smoke-test scale only, matching what
    was already manually verified: tiny epochs, K=2, 5 test windows, 2
    assets for the standardizer. NOT a full run."""
    methods = "s0,s1"
    print("=== DEMO: states ===")
    run("daily_macro_states.py", ["--methods", methods])
    print("\n=== DEMO: standardize (2 assets, 5 epochs) ===")
    run("macro_standardization.py", ["--methods", methods, "--assets", "IBM,AAPL", "--epochs", "5"])
    print("\n=== DEMO: stationarity (2 assets) ===")
    run("stationarity_evaluation.py", ["--methods", methods, "--assets", "IBM,AAPL"])
    print("\n=== DEMO: train-diffusion + sample (3 epochs, K=2, 5 test windows) ===")
    run("diffusion_adapters.py", ["--methods", methods, "--mode", "both", "--epochs", "3",
                                   "--n-eval-windows", "5", "--n-samples", "2", "--num-steps", "50",
                                   "--states-dir", OUT_DIR, "--out-dir", OUT_DIR])  # never the real results folders
    print("\n=== DEMO: evaluate ===")
    mode_evaluate(methods.split(","), OUT_DIR)
    print("\nDemo complete -- all numbers above are smoke-test scale only, not real experimental results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                         choices=["states", "standardize", "stationarity", "train-diffusion",
                                  "sample", "evaluate", "demo"])
    parser.add_argument("--methods", type=str, default="s0,s1,s2,s3,s4")
    parser.add_argument("--assets", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-eval-windows", type=int, default=1200)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default=None,
                         help="Optional; must equal config.data.conditioning_set (inputs + results folders come from it).")
    parser.add_argument("--results-dir", type=str, default=None,
                         help="evaluate: folder holding <method>/diffusion_causal/ (diffusion_adapters --out-dir)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.mode == "states":
        run("daily_macro_states.py", ["--methods", args.methods])
    elif args.mode == "standardize":
        extra = ["--methods", args.methods]
        if args.assets: extra += ["--assets", args.assets]
        if args.epochs: extra += ["--epochs", str(args.epochs)]
        if args.batch_size: extra += ["--batch-size", str(args.batch_size)]
        run("macro_standardization.py", extra)
    elif args.mode == "stationarity":
        extra = ["--methods", args.methods]
        if args.assets: extra += ["--assets", args.assets]
        run("stationarity_evaluation.py", extra)
    elif args.mode in ("train-diffusion", "sample"):
        dm_mode = "train" if args.mode == "train-diffusion" else "sample"
        extra = ["--methods", args.methods, "--mode", dm_mode,
                 "--n-samples", str(args.n_samples), "--n-eval-windows", str(args.n_eval_windows)]
        if args.experiment: extra += ["--experiment", args.experiment]
        if args.epochs: extra += ["--epochs", str(args.epochs)]
        if args.batch_size: extra += ["--batch-size", str(args.batch_size)]
        if args.num_steps: extra += ["--num-steps", str(args.num_steps)]
        if args.seed is not None: extra += ["--seed", str(args.seed)]
        run("diffusion_adapters.py", extra)
    elif args.mode == "evaluate":
        mode_evaluate(methods, args.results_dir or experiment_paths(args.experiment)["macro_embed"])
    elif args.mode == "demo":
        mode_demo()


if __name__ == "__main__":
    main()
