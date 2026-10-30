"""
Runs the UNMODIFIED, untouched main.py once per exogenous conditioning
channel (macro_change, ewma_mean_change, ewma_vol_change, pca_change),
to generalize the production Doob-h rare-event pipeline beyond the macro
state -- WITHOUT changing a single line of main.py, data_processor.py,
hfunction_direct.py, or conditional_generator.py.

Mechanism (pure external orchestration, no code changes to the pipeline):
  1. data/data_processor.py picks the conditioning column POSITIONALLY
     (df.columns[0]) from whatever CSV main.py's config points at
     (explore/standardization_results/data_for_diffusion.csv). So swapping
     which channel is used is just a matter of swapping what that file
     contains -- this script backs up the real file, overwrites it with
     one of the 4 pre-built per-channel CSVs (see build_channel_csvs.py),
     runs `python main.py` as a plain subprocess, then restores the
     original file, every single time, even on failure (try/finally).
  2. main.py's own, pre-existing checkpoint-reuse logic
     (`if args.retrain or not os.path.exists(ckpt): train else: load`)
     is left completely alone. score_{run_tag}.pt is channel-independent
     (same assets/architecture), so once channel 1 trains it, it's reused
     automatically for channels 2-4 -- no --retrain is ever passed.
     hfunction_{run_tag}.pt IS channel-specific, but main.py writes it to
     the same unsuffixed path every time; this script makes sure a fresh
     one gets trained each channel by MOVING the previous channel's
     hfunction/ell/h_losses/q_model files out of ckpt_new/ into that
     channel's own results folder immediately after each run completes --
     so the next channel's run naturally finds "no checkpoint yet" and
     trains fresh, with zero changes to main.py's own logic.
  3. The other fixed-path outputs main.py always writes
     (generated_samples_train.pt, generated_samples_test.pt,
     results/portfolio_comparison_{insample,outsample}.png) get moved the
     same way, into baseline_experiments/experiment_1/doobh_results/<channel>/.

Usage:
    python run_doobh_channels.py                       # all 4 channels
    python run_doobh_channels.py --channel macro_change # just one
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
sys.path.insert(0, _ROOT)

from config import get_default_config, get_run_tag  # noqa: E402

CHANNELS = ["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"]

PROD_CSV = os.path.join(_ROOT, "explore", "standardization_results", "data_for_diffusion.csv")
BACKUP_CSV = PROD_CSV + ".orig_backup"
RESULTS_ROOT = os.path.join(_HERE, "doobh_results")

CFG = get_default_config()
RUN_TAG = get_run_tag(CFG)

SCORE_CKPT = os.path.join(_ROOT, "ckpt_new", f"score_{RUN_TAG}.pt")           # never touched
SCORE_LOSSES = os.path.join(_ROOT, "ckpt_new", f"score_losses_{RUN_TAG}.csv")  # never touched

# Files main.py writes under these FIXED, unsuffixed names every run -- these
# are what get moved into each channel's own folder after that channel's run.
MOVABLE = [
    os.path.join(_ROOT, "ckpt_new", f"hfunction_{RUN_TAG}.pt"),
    os.path.join(_ROOT, "ckpt_new", f"ell_function_{RUN_TAG}.pt"),
    os.path.join(_ROOT, "ckpt_new", f"h_losses_{RUN_TAG}.csv"),
    os.path.join(_ROOT, "ckpt_new", "q_model.pt"),
    os.path.join(_ROOT, "generated_samples_train.pt"),
    os.path.join(_ROOT, "generated_samples_test.pt"),
    os.path.join(_ROOT, "results", "portfolio_comparison_insample.png"),
    os.path.join(_ROOT, "results", "portfolio_comparison_outsample.png"),
]


def swap_in_channel_csv(channel: str):
    channel_csv = os.path.join(_ROOT, "explore", "standardization_results",
                                f"data_for_diffusion_{channel}.csv")
    if not os.path.exists(channel_csv):
        raise FileNotFoundError(
            f"{channel_csv} not found -- run build_channel_csvs.py first "
            f"(baseline_experiments/experiment_2/build_channel_csvs.py)."
        )
    if not os.path.exists(BACKUP_CSV):
        shutil.copy2(PROD_CSV, BACKUP_CSV)
        print(f"Backed up original {PROD_CSV} -> {BACKUP_CSV}")
    shutil.copy2(channel_csv, PROD_CSV)
    print(f"Swapped in {channel_csv} -> {PROD_CSV}")


def restore_original_csv():
    if os.path.exists(BACKUP_CSV) and not filecmp.cmp(BACKUP_CSV, PROD_CSV, shallow=False):
        shutil.copy2(BACKUP_CSV, PROD_CSV)
        print(f"Restored original {PROD_CSV} from backup")


def self_heal_on_startup():
    """A prior run could have been killed by an UNCATCHABLE signal (SIGKILL)
    between swapping the CSV in and restoring it -- Python can't run any
    cleanup code in that case, regardless of try/finally or signal handlers.
    So on every startup, if a backup exists and the live file doesn't match
    it, restore it before doing anything else, rather than silently training
    on a leftover swapped file."""
    if os.path.exists(BACKUP_CSV) and not filecmp.cmp(BACKUP_CSV, PROD_CSV, shallow=False):
        print(f"[self-heal] {PROD_CSV} does not match {BACKUP_CSV} at startup -- "
              f"a previous run must have been killed uncleanly. Restoring now.")
        shutil.copy2(BACKUP_CSV, PROD_CSV)


def _install_signal_handlers():
    """SIGTERM/SIGINT (what `scancel`/Ctrl-C/`timeout` send) do NOT run `finally`
    blocks or atexit hooks by default -- they kill the process immediately at
    the OS level. Converting them into a normal SystemExit is what makes the
    try/finally restore in run_channel() actually fire."""
    def _handler(signum, _frame):
        print(f"\n[signal] received signal {signum} -- exiting cleanly so the "
              f"original CSV gets restored.")
        sys.exit(1)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def clear_stale_hfunction_outputs():
    """Called before each channel's run: if a hfunction/ell/h_losses/q_model
    file is still sitting at the unsuffixed path (should only happen if a
    previous channel's run crashed before this script could move it), move
    it into a 'stale' holding folder rather than silently deleting it."""
    stale_dir = os.path.join(RESULTS_ROOT, "_stale_leftovers")
    moved_any = False
    for path in MOVABLE:
        if os.path.exists(path):
            os.makedirs(stale_dir, exist_ok=True)
            dest = os.path.join(stale_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"[warn] found pre-existing {path}, moved to {dest} before starting fresh run")
            moved_any = True
    return moved_any


def move_outputs_to_channel_folder(channel: str):
    out_dir = os.path.join(RESULTS_ROOT, channel)
    os.makedirs(out_dir, exist_ok=True)
    for path in MOVABLE:
        if os.path.exists(path):
            dest = os.path.join(out_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"  moved {path} -> {dest}")
        else:
            print(f"  [note] {path} was not produced this run")


def run_channel(channel: str):
    print(f"\n{'=' * 70}\nChannel: {channel}\n{'=' * 70}")
    clear_stale_hfunction_outputs()
    swap_in_channel_csv(channel)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "main.py", "--no-wandb"],
            cwd=_ROOT, check=False,
        )
    finally:
        restore_original_csv()

    # main.py's STEP 6 (portfolio analysis) has a known, pre-existing bug
    # (DataProcessor has no attribute 'invert_samples' -- unrelated to this
    # generalization, already seen/accepted on the original 15-stock run) that
    # crashes it with exit code 1. By that point STEPS 1-5 -- data processing,
    # score model, h-function training, event extraction, and the actual
    # Doob-h guided generation -- have already completed and saved their
    # outputs (checkpoint.pt / generated_samples_{train,test}.pt), so this is
    # not fatal to what we actually need: move whatever got produced and
    # continue to the next channel rather than aborting the whole job.
    if result.returncode != 0:
        print(f"[warn] main.py exited {result.returncode} for channel '{channel}' "
              f"(expected: the known STEP 6 portfolio-analysis bug, unrelated to "
              f"this run). Checking whether the generation itself still completed...")

    move_outputs_to_channel_folder(channel)
    gen_path = os.path.join(RESULTS_ROOT, channel, "generated_samples_test.pt")
    if os.path.exists(gen_path):
        print(f"Channel '{channel}' done -> {os.path.join(RESULTS_ROOT, channel)}")
    else:
        print(f"[error] Channel '{channel}' did NOT produce generated_samples_test.pt -- "
              f"a real failure, not just the known STEP 6 bug. Inspect the log above.")
    print(f"(score checkpoint untouched/shared at {SCORE_CKPT})")


def main():
    self_heal_on_startup()
    _install_signal_handlers()
    atexit.register(restore_original_csv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=CHANNELS, default=None,
                         help="Run just this one channel. Default: all 4, in sequence.")
    args = parser.parse_args()

    channels = [args.channel] if args.channel else CHANNELS
    for channel in channels:
        run_channel(channel)

    print(f"\nAll done. Shared score checkpoint: {SCORE_CKPT}")
    print(f"Per-channel results under: {RESULTS_ROOT}/<channel>/")


if __name__ == "__main__":
    main()
