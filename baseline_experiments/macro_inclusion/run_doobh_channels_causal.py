"""
Re-runs the SAME 4-channel Doob-h comparison as doobh_results/ (macro_change,
ewma_mean_change, ewma_vol_change, pca_change), with exactly ONE difference:
config.data.event_causal flipped from False (the default doobh_results used)
to True. This changes how event windows are labeled in data_processor.py's
_scan_macro_windows: at event_causal=False, Z_end can be read from the LAST
DAY of the very return-window being generated (a real, if small, lookahead
in the event label itself); at event_causal=True, the event's Z_start/Z_end
window ends BEFORE the return window starts (config.data.event_lag_gap days
of buffer), so the event is fully known before generation begins.

main.py, data_processor.py, hfunction_direct.py, conditional_generator.py
are NEVER modified -- same external-orchestration-only approach as
run_doobh_channels.py. The ONE line this script temporarily edits is
config/config.py's `event_causal: bool = False` default, restored
unconditionally when the run ends (success, failure, or signal) via the same
backup/self-heal/signal-handler pattern already used for the CSV swapping.
config.get_run_tag() bakes causal/noncausal into the checkpoint filename, so
flipping this does NOT silently reuse/corrupt the existing noncausal score
checkpoint -- main.py's own unmodified checkpoint logic will look for (and,
per ckpt_new/, already find) a separate "..._causal_LS_15a.pt" checkpoint.

Usage:
    python run_doobh_channels_causal.py                       # all 4 channels
    python run_doobh_channels_causal.py --channel macro_change # just one
"""
import argparse
import atexit
import filecmp
import os
import shutil
import signal
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_RESULTS_ROOT = os.path.join(os.path.dirname(_HERE), "main_results")
sys.path.insert(0, _ROOT)

CHANNELS = ["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"]

PROD_CSV = os.path.join(_ROOT, "explore", "standardization_results", "data_for_diffusion.csv")
CSV_BACKUP = PROD_CSV + ".orig_backup"

CONFIG_PY = os.path.join(_ROOT, "config", "config.py")
CONFIG_BACKUP = CONFIG_PY + ".orig_backup_causal_run"
EVENT_CAUSAL_FALSE = "    event_causal: bool = False"
EVENT_CAUSAL_TRUE = "    event_causal: bool = True"

RESULTS_ROOT = os.path.join(_RESULTS_ROOT, "doobh_results_causal")

# ckpt_new/ files written to FIXED, unsuffixed names every run (except the
# score checkpoint itself, which is correctly tagged causal/noncausal by
# get_run_tag() and is deliberately left alone/reused across channels).
MOVABLE = None  # filled in after we can import config (needs RUN_TAG)


def swap_in_channel_csv(channel: str):
    channel_csv = os.path.join(_ROOT, "explore", "standardization_results",
                                f"data_for_diffusion_{channel}.csv")
    if not os.path.exists(channel_csv):
        raise FileNotFoundError(f"{channel_csv} not found -- expected it already built by "
                                 f"build_channel_csvs.py (same CSVs doobh_results/ used).")
    if not os.path.exists(CSV_BACKUP):
        shutil.copy2(PROD_CSV, CSV_BACKUP)
        print(f"Backed up original {PROD_CSV} -> {CSV_BACKUP}")
    shutil.copy2(channel_csv, PROD_CSV)
    print(f"Swapped in {channel_csv} -> {PROD_CSV}")


def restore_csv():
    if os.path.exists(CSV_BACKUP) and not filecmp.cmp(CSV_BACKUP, PROD_CSV, shallow=False):
        shutil.copy2(CSV_BACKUP, PROD_CSV)
        print(f"Restored original {PROD_CSV} from backup")


def flip_event_causal_true():
    if not os.path.exists(CONFIG_BACKUP):
        shutil.copy2(CONFIG_PY, CONFIG_BACKUP)
        print(f"Backed up {CONFIG_PY} -> {CONFIG_BACKUP}")
    with open(CONFIG_PY) as f:
        text = f.read()
    if text.count(EVENT_CAUSAL_FALSE) != 1:
        raise RuntimeError(f"Expected exactly 1 occurrence of {EVENT_CAUSAL_FALSE!r} in "
                            f"{CONFIG_PY}, found {text.count(EVENT_CAUSAL_FALSE)} -- refusing "
                            f"to edit, config.py may have already changed.")
    with open(CONFIG_PY, "w") as f:
        f.write(text.replace(EVENT_CAUSAL_FALSE, EVENT_CAUSAL_TRUE))
    print(f"Flipped event_causal False -> True in {CONFIG_PY}")


def restore_config():
    if os.path.exists(CONFIG_BACKUP) and not filecmp.cmp(CONFIG_BACKUP, CONFIG_PY, shallow=False):
        shutil.copy2(CONFIG_BACKUP, CONFIG_PY)
        print(f"Restored original {CONFIG_PY} from backup")


def self_heal_on_startup():
    """Same uncatchable-SIGKILL protection as run_doobh_channels.py, applied
    to both the swapped CSV and the temporarily-edited config.py."""
    if os.path.exists(CSV_BACKUP) and not filecmp.cmp(CSV_BACKUP, PROD_CSV, shallow=False):
        print(f"[self-heal] {PROD_CSV} does not match backup at startup -- restoring.")
        shutil.copy2(CSV_BACKUP, PROD_CSV)
    if os.path.exists(CONFIG_BACKUP) and not filecmp.cmp(CONFIG_BACKUP, CONFIG_PY, shallow=False):
        print(f"[self-heal] {CONFIG_PY} does not match backup at startup -- restoring.")
        shutil.copy2(CONFIG_BACKUP, CONFIG_PY)


def _install_signal_handlers():
    def _handler(signum, _frame):
        print(f"\n[signal] received signal {signum} -- exiting cleanly so config.py/CSV get restored.")
        sys.exit(1)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def clear_stale_hfunction_outputs(movable):
    stale_dir = os.path.join(RESULTS_ROOT, "_stale_leftovers")
    for path in movable:
        if os.path.exists(path):
            os.makedirs(stale_dir, exist_ok=True)
            dest = os.path.join(stale_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"[warn] found pre-existing {path}, moved to {dest} before starting fresh run")


def move_outputs_to_channel_folder(channel, movable):
    out_dir = os.path.join(RESULTS_ROOT, channel)
    os.makedirs(out_dir, exist_ok=True)
    for path in movable:
        if os.path.exists(path):
            dest = os.path.join(out_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"  moved {path} -> {dest}")
        else:
            print(f"  [note] {path} was not produced this run")


def run_channel(channel, movable, score_ckpt, force_retrain=False):
    print(f"\n{'=' * 70}\nChannel: {channel} (event_causal=True, retrain={force_retrain})\n{'=' * 70}")
    clear_stale_hfunction_outputs(movable)
    swap_in_channel_csv(channel)
    cmd = [sys.executable, "-u", "main.py", "--no-wandb"]
    if force_retrain:
        cmd.append("--retrain")
    try:
        result = subprocess.run(cmd, cwd=_ROOT, check=False)
    finally:
        restore_csv()

    if result.returncode != 0:
        print(f"[warn] main.py exited {result.returncode} for channel '{channel}' "
              f"(expected: the known STEP 6 portfolio-analysis bug, same as the original "
              f"doobh_results runs). Checking whether generation itself still completed...")

    move_outputs_to_channel_folder(channel, movable)
    gen_path = os.path.join(RESULTS_ROOT, channel, "generated_samples_test.pt")
    if os.path.exists(gen_path):
        print(f"Channel '{channel}' done -> {os.path.join(RESULTS_ROOT, channel)}")
    else:
        print(f"[error] Channel '{channel}' did NOT produce generated_samples_test.pt -- a real failure.")
    print(f"(score checkpoint: {score_ckpt}, {'reused/cached' if os.path.exists(score_ckpt) else 'will be trained fresh'})")


def main():
    self_heal_on_startup()
    _install_signal_handlers()
    atexit.register(restore_csv)
    atexit.register(restore_config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=CHANNELS, default=None)
    args = parser.parse_args()

    flip_event_causal_true()
    try:
        # Import AFTER flipping config.py so get_run_tag() reflects event_causal=True
        import importlib
        import config as config_module
        importlib.reload(config_module)
        cfg = config_module.get_default_config()
        if not cfg.data.event_causal:
            raise RuntimeError("config.data.event_causal is still False after the flip -- aborting.")
        run_tag = config_module.get_run_tag(cfg)
        print(f"Confirmed event_causal=True; run_tag={run_tag}")

        score_ckpt = os.path.join(_ROOT, "ckpt_new", f"score_{run_tag}.pt")
        movable = [
            os.path.join(_ROOT, "ckpt_new", f"hfunction_{run_tag}.pt"),
            os.path.join(_ROOT, "ckpt_new", f"ell_function_{run_tag}.pt"),
            os.path.join(_ROOT, "ckpt_new", f"h_losses_{run_tag}.csv"),
            os.path.join(_ROOT, "ckpt_new", "q_model.pt"),
            os.path.join(_ROOT, "generated_samples_train.pt"),
            os.path.join(_ROOT, "generated_samples_test.pt"),
            os.path.join(_ROOT, "results", "portfolio_comparison_insample.png"),
            os.path.join(_ROOT, "results", "portfolio_comparison_outsample.png"),
        ]
        print(f"Score checkpoint for this run: {score_ckpt} "
              f"({'EXISTS -- will be reused, not retrained' if os.path.exists(score_ckpt) else 'does NOT exist -- main.py will train it fresh'})")

        channels = [args.channel] if args.channel else CHANNELS
        for i, channel in enumerate(channels):
            # The cached score checkpoint for event_causal=True is stale (predates
            # the cond_embed module) and fails to load -- force a fresh retrain on
            # the FIRST channel only; channels after that correctly reuse the new,
            # now-valid checkpoint main.py just wrote (same shared-score-model
            # pattern as the original doobh_results run).
            run_channel(channel, movable, score_ckpt, force_retrain=(i == 0))

        # Evaluation must also run while event_causal=True is still active --
        # doobh_evaluation.py rebuilds its own DataProcessor from config.data,
        # and event classification itself depends on this flag.
        print(f"\n{'=' * 70}\nEvaluation (event_causal=True, still active)\n{'=' * 70}")
        for channel in (channels if args.channel else CHANNELS):
            print(f"=== doobh_evaluation.py --channel {channel} ===")
            subprocess.run([sys.executable, "-u", "doobh_evaluation.py",
                             "--channel", channel, "--results-root", "doobh_results_causal"],
                            cwd=_HERE, check=False)

        # still inside event_causal=True: the artifacts rebuild the same causal event mask
        print(f"\n{'=' * 70}\nSaving rare-event artifacts (event_causal=True, still active)\n{'=' * 70}")
        for channel in (channels if args.channel else CHANNELS):
            subprocess.run([sys.executable, "-u", "save_rare_event_artifacts.py", "--group", "inclusion",
                            "--channel", channel, "--results-root", "doobh_results_causal"], cwd=_HERE, check=False)

        print(f"\nAll done. Score checkpoint: {score_ckpt}")
        print(f"Per-channel results under: {RESULTS_ROOT}/<channel>/")
    finally:
        restore_config()


if __name__ == "__main__":
    main()
