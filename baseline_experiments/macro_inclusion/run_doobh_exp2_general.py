"""
Causal Doob-h rare-event runs for the macro-embedding methods (S0 state-space, S1 linear, S2 forward-fill,
S3 cubic spline, S4 tracking regression, S1'/S3' causal extrapolations), each method as its own conditioning
channel, on the conditioning set selected in config (config.data.conditioning_set).

For each channel, main.py is run with
    --data-csv <experiment>/extras/rare_event_channels/data_for_diffusion_<channel>_ownstd.csv
(built by build_channel_csvs_exp2_ownstd.py): column 0 = that method's raw daily state (its windowed change is the
Doob-h event label); the other columns = the 15 stocks' residuals standardized on that method's OWN state.
Every channel trains its OWN base diffusion model (the targets differ per method) under its own checkpoint tag
<run_tag>__rare_embed_<channel>_ownstd, always with --retrain, and the base model is moved into the channel's
results folder -- it can never be reused by another channel. No file is swapped and config.py is never edited.
Requires config.data.event_causal = True (causal only).

Results: <experiment>/causal/macro_embed/rare_event/<run_tag>/<channel>/
(experiment_1 (Inflation Only) or experiment_2 (CPI Only)).

Usage (set the conditioning set in config/config.py first):
    python run_doobh_exp2_general.py --channels exp2_s0,exp2_s1p,exp2_s2,exp2_s3p,exp2_s4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from config import get_default_config, get_run_tag  # noqa: E402
from window_conditioning import experiment_paths  # noqa: E402

ALL_CHANNELS = ["exp2_s0", "exp2_s1", "exp2_s2", "exp2_s3", "exp2_s4", "exp2_s1p", "exp2_s3p"]
RETROSPECTIVE = {"exp2_s0": False, "exp2_s1": True, "exp2_s2": False, "exp2_s3": True, "exp2_s4": False,
                 "exp2_s1p": False, "exp2_s3p": False}
CSV_SUFFIX = "_ownstd"


def clear_stale_outputs(movable, results_root):
    stale_dir = os.path.join(results_root, "_stale_leftovers")
    for path in movable:
        if os.path.exists(path):
            os.makedirs(stale_dir, exist_ok=True)
            dest = os.path.join(stale_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"[warn] found pre-existing {path}, moved to {dest} before starting fresh run")


def run_channel(channel, csv, base_tag, results_root, cfg):
    """Returns True only if THIS run trained a fresh base model and produced generated samples."""
    tag_suffix = f"rare_embed_{channel}{CSV_SUFFIX}"
    run_tag = f"{base_tag}__{tag_suffix}"
    movable = [
        os.path.join(_ROOT, "ckpt_new", f"score_{run_tag}.pt"),
        os.path.join(_ROOT, "ckpt_new", f"score_losses_{run_tag}.csv"),
        os.path.join(_ROOT, "ckpt_new", f"hfunction_{run_tag}.pt"),
        os.path.join(_ROOT, "ckpt_new", f"ell_function_{run_tag}.pt"),
        os.path.join(_ROOT, "ckpt_new", f"h_losses_{run_tag}.csv"),
        os.path.join(_ROOT, "ckpt_new", "q_model.pt"),
        # main.py writes these to results/<run_tag>/ (its per-run output folder)
        os.path.join(_ROOT, "results", run_tag, "generated_samples_train.pt"),
        os.path.join(_ROOT, "results", run_tag, "generated_samples_test.pt"),
        os.path.join(_ROOT, "results", run_tag, "generated_samples_meta.json"),
        os.path.join(_ROOT, "results", run_tag, "portfolio_comparison_insample.png"),
        os.path.join(_ROOT, "results", run_tag, "portfolio_comparison_outsample.png"),
    ]
    t0 = time.time()
    print(f"\n{'=' * 70}\nChannel: {channel}\n  data: {csv}\n  run_tag: {run_tag}\n{'=' * 70}", flush=True)
    clear_stale_outputs(movable, results_root)
    cmd = [sys.executable, "-u", "main.py", "--no-wandb", "--retrain", "--data-csv", csv, "--tag-suffix", tag_suffix]
    result = subprocess.run(cmd, cwd=_ROOT, check=False)
    if result.returncode != 0:
        print(f"[warn] main.py exited {result.returncode} for channel '{channel}' (the known STEP 6 portfolio-"
              f"analysis error happens after generation). Checking generation output...")

    out_dir = os.path.join(results_root, channel)
    os.makedirs(out_dir, exist_ok=True)
    for path in movable:
        if os.path.exists(path):
            shutil.move(path, os.path.join(out_dir, os.path.basename(path)))
            print(f"  moved {path} -> {out_dir}/")
        else:
            print(f"  [note] {path} was not produced this run")
    base = os.path.join(out_dir, f"score_{run_tag}.pt")
    gen = os.path.join(out_dir, "generated_samples_test.pt")
    base_fresh = os.path.exists(base) and os.path.getmtime(base) >= t0
    gen_fresh = os.path.exists(gen) and os.path.getmtime(gen) >= t0
    with open(os.path.join(out_dir, "channel_meta.json"), "w") as f:
        json.dump({"channel": channel, "conditioning_set": cfg.data.conditioning_set, "data_file": csv,
                   "diffusion_target": "own NN residuals of this method", "run_tag": run_tag,
                   "base_model": base, "base_model_trained_this_channel": base_fresh,
                   "event_causal": cfg.data.event_causal, "event_lag_gap": cfg.data.event_lag_gap,
                   "event_type": cfg.hfunction.event_type, "event_top_fraction": cfg.hfunction.event_threshold,
                   "retrospective": RETROSPECTIVE[channel],
                   "note": ("retrospective state construction (needs the next monthly anchor) -- disclosed"
                            if RETROSPECTIVE[channel] else "causal in its own state construction")}, f, indent=2)
    ok = base_fresh and gen_fresh
    print(f"Channel '{channel}' {'done' if ok else 'FAILED (base model or samples not produced in this run)'} -> {out_dir}",
          flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", required=True, help="Comma-separated subset of: " + ",".join(ALL_CHANNELS))
    args = parser.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    for c in channels:
        if c not in ALL_CHANNELS:
            raise SystemExit(f"Unknown channel {c!r}, expected one of {ALL_CHANNELS}")

    cfg = get_default_config()
    if not cfg.data.event_causal:
        raise SystemExit("config.data.event_causal must be True (causal-only)")
    paths = experiment_paths()
    base_tag = get_run_tag(cfg)
    results_root = os.path.join(paths["rare_embed"], base_tag)
    csvs = {}
    for c in channels:   # all channel files must exist before anything runs
        csvs[c] = os.path.join(paths["embed_channels_dir"], f"data_for_diffusion_{c}{CSV_SUFFIX}.csv")
        if not os.path.exists(csvs[c]):
            raise SystemExit(f"{csvs[c]} not found -- run build_channel_csvs_exp2_ownstd.py --channels {args.channels}")
    print(f"conditioning_set = {cfg.data.conditioning_set} | event_causal={cfg.data.event_causal} "
          f"lag={cfg.data.event_lag_gap} {cfg.hfunction.event_type} top {cfg.hfunction.event_threshold:.1%} "
          f"| window {cfg.data.start_date}..{cfg.data.end_date} | seed {cfg.seed}\nresults = {results_root}", flush=True)

    failures = []
    for channel in channels:
        if not run_channel(channel, csvs[channel], base_tag, results_root, cfg):
            failures.append(f"{channel}: training/generation")

    print(f"\n{'=' * 70}\nEvaluation\n{'=' * 70}", flush=True)
    for channel in channels:
        if subprocess.run([sys.executable, "-u", "doobh_evaluation_exp2.py", "--channel", channel,
                           "--results-root", results_root, "--csv-suffix", CSV_SUFFIX], cwd=_HERE).returncode != 0:
            failures.append(f"{channel}: evaluation")

    print(f"\n{'=' * 70}\nSaving rare-event artifacts\n{'=' * 70}", flush=True)
    for channel in channels:
        if subprocess.run([sys.executable, "-u", "save_rare_event_artifacts.py", "--group", "embedding",
                           "--channel", channel, "--results-root", results_root, "--csv-suffix", CSV_SUFFIX,
                           "--standardization-dir", paths["standardization_dir"]], cwd=_HERE).returncode != 0:
            failures.append(f"{channel}: artifacts")

    print(f"\nAll done. Per-channel results under: {results_root}/<channel>/")
    if failures:
        print(f"[FAILED] {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
