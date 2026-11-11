"""
Causal Doob-h rare-event runs for the macro-inclusion channels (macro_change, ewma_mean_change, ewma_vol_change,
pca_change), on the conditioning set selected in config (config.data.conditioning_set).

For each channel, main.py is run with
    --data-csv explore/conditioning_sets/<set>/rare_event_channels/data_for_diffusion_<channel>.csv
(built by build_channel_csvs.py --raw-returns): column 0 = the channel's raw level series, whose windowed change is
the Doob-h event label; the other columns = raw daily log returns (macro inclusion must not depend on
m_t-standardized residuals). main.py reads the channel file directly -- no file is swapped and config.py is never
edited. Requires config.data.event_causal = True (causal only).

Checkpoints: ckpt_new/*_<run_tag>__rare_inclusion_rawret.* -- never shared with the conditioning set's own main.py
runs. The score model (same raw-return targets for every channel) is trained on the first channel and reused; the
H-function is channel-specific and is moved into each channel's results folder after its run.
Results: <experiment folder of the set>/causal/macro_inclusion/rare_event/<run_tag>/<channel>/
(experiment_1 (Inflation Only) or experiment_2 (CPI Only)). Channel files are built automatically if missing.

Usage:
    python run_doobh_channels_causal.py                       # all 4 channels
    python run_doobh_channels_causal.py --channel macro_change # just one
"""
import argparse
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from config import get_default_config, get_run_tag  # noqa: E402
from window_conditioning import experiment_paths  # noqa: E402

CHANNELS = ["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"]
TAG_SUFFIX = "rare_inclusion_rawret"


def channel_csv(cfg, channel):
    path = os.path.join(cfg.data.channels_dir, f"data_for_diffusion_{channel}.csv")
    if not os.path.exists(path):   # built once per conditioning set from its own files (deterministic)
        print(f"{path} missing -- building the channel files: build_channel_csvs.py --raw-returns", flush=True)
        subprocess.run([sys.executable, "-u", "build_channel_csvs.py", "--raw-returns"], cwd=_HERE, check=True)
    if not os.path.exists(path):
        raise SystemExit(f"{path} still not found after build_channel_csvs.py")
    return path


def clear_stale_outputs(movable, results_root):
    stale_dir = os.path.join(results_root, "_stale_leftovers")
    for path in movable:
        if os.path.exists(path):
            os.makedirs(stale_dir, exist_ok=True)
            dest = os.path.join(stale_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"[warn] found pre-existing {path}, moved to {dest} before starting fresh run")


def move_outputs_to_channel_folder(channel, movable, results_root):
    out_dir = os.path.join(results_root, channel)
    os.makedirs(out_dir, exist_ok=True)
    for path in movable:
        if os.path.exists(path):
            dest = os.path.join(out_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"  moved {path} -> {dest}")
        else:
            print(f"  [note] {path} was not produced this run")


def run_channel(channel, csv, movable, results_root, force_retrain):
    print(f"\n{'=' * 70}\nChannel: {channel} (event_causal=True, retrain={force_retrain})\n  data: {csv}\n{'=' * 70}",
          flush=True)
    clear_stale_outputs(movable, results_root)
    cmd = [sys.executable, "-u", "main.py", "--no-wandb", "--data-csv", csv, "--tag-suffix", TAG_SUFFIX]
    if force_retrain:
        cmd.append("--retrain")
    result = subprocess.run(cmd, cwd=_ROOT, check=False)
    if result.returncode != 0:
        print(f"[warn] main.py exited {result.returncode} for channel '{channel}' (the known STEP 6 portfolio-"
              f"analysis error happens after generation). Checking whether generation itself completed...")
    move_outputs_to_channel_folder(channel, movable, results_root)
    ok = os.path.exists(os.path.join(results_root, channel, "generated_samples_test.pt"))
    print(f"Channel '{channel}' {'done' if ok else 'FAILED: no generated_samples_test.pt'} -> "
          f"{os.path.join(results_root, channel)}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=CHANNELS, default=None)
    args = parser.parse_args()

    cfg = get_default_config()
    if not cfg.data.event_causal:
        raise SystemExit("config.data.event_causal must be True (causal-only)")
    base_tag = get_run_tag(cfg)
    run_tag = f"{base_tag}__{TAG_SUFFIX}"
    results_root = os.path.join(experiment_paths()["rare_inclusion"], base_tag)
    results_sub = results_root   # absolute: doobh_evaluation / save_rare_event_artifacts join it unchanged
    channels = [args.channel] if args.channel else CHANNELS
    csvs = {c: channel_csv(cfg, c) for c in channels}   # all channel files must exist before anything runs

    print(f"conditioning_set = {cfg.data.conditioning_set} | event_causal={cfg.data.event_causal} "
          f"lag={cfg.data.event_lag_gap} {cfg.hfunction.event_type} top {cfg.hfunction.event_threshold:.1%} "
          f"| window {cfg.data.start_date}..{cfg.data.end_date} | seed {cfg.seed}")
    print(f"run_tag = {run_tag}\nresults = {results_root}", flush=True)

    movable = [
        os.path.join(_ROOT, "ckpt_new", f"hfunction_{run_tag}.pt"),
        os.path.join(_ROOT, "ckpt_new", f"ell_function_{run_tag}.pt"),
        os.path.join(_ROOT, "ckpt_new", f"h_losses_{run_tag}.csv"),
        os.path.join(_ROOT, "ckpt_new", "q_model.pt"),
        # main.py writes these to <set>/diffusion_runs/<run_tag>/ (its per-run output folder)
        os.path.join(cfg.data.diffusion_runs_dir, run_tag, "generated_samples_train.pt"),
        os.path.join(cfg.data.diffusion_runs_dir, run_tag, "generated_samples_test.pt"),
        os.path.join(cfg.data.diffusion_runs_dir, run_tag, "generated_samples_meta.json"),
        os.path.join(cfg.data.diffusion_runs_dir, run_tag, "portfolio_comparison_insample.png"),
        os.path.join(cfg.data.diffusion_runs_dir, run_tag, "portfolio_comparison_outsample.png"),
    ]

    failed = False
    for i, channel in enumerate(channels):
        # score model trained fresh on the first channel, reused (same raw-return targets) by the others
        failed |= not run_channel(channel, csvs[channel], movable, results_root, force_retrain=(i == 0))

    print(f"\n{'=' * 70}\nEvaluation\n{'=' * 70}", flush=True)
    for channel in channels:
        subprocess.run([sys.executable, "-u", "doobh_evaluation.py", "--channel", channel,
                        "--results-root", results_sub], cwd=_HERE, check=False)

    print(f"\n{'=' * 70}\nSaving rare-event artifacts\n{'=' * 70}", flush=True)
    for channel in channels:
        subprocess.run([sys.executable, "-u", "save_rare_event_artifacts.py", "--group", "inclusion",
                        "--channel", channel, "--results-root", results_sub], cwd=_HERE, check=False)
        if not os.path.exists(os.path.join(results_root, channel, "rare_event_artifacts.npz")):
            failed = True

    print(f"\nAll done. Score checkpoint: ckpt_new/score_{run_tag}.pt\nPer-channel results under: {results_root}/<channel>/")
    if failed:
        sys.exit("[FAILED] a channel produced no generated samples / artifacts")


if __name__ == "__main__":
    main()
