"""
Guided Doob-h rare-event generation for Experiment 2's 5 macro-construction
methods (S0 state-space, S1 linear, S2 forward-fill, S3 cubic spline, S4
tracking regression), each as its own conditioning channel -- same mechanism
as the original doobh_results/ (main.py's unmodified production pipeline,
externally orchestrated via CSV-swapping), generalized along TWO axes this
time: (a) which of the 5 Experiment-2 methods is the channel, (b)
config.data.event_causal True or False.

main.py, data_processor.py, hfunction_direct.py, conditional_generator.py
are NEVER modified. The CSV-swap mechanism is identical to
run_doobh_channels.py; the config.py event_causal flip (when --causal is
passed) is identical to run_doobh_channels_causal.py's, including the same
backup/self-heal/signal-handler restore guarantees.

Each channel is evaluated against its OWN valid-window pool (not a pool
shared across all 5 Experiment-2 channels) -- this run is explicitly split
across multiple jobs/channel subsets so results can be inspected before the
full channel set finishes, which rules out a cross-channel common pool for
this particular comparison (same self-contained-per-channel choice already
used for the cheaper tail_event_analysis.py check earlier). Disclosed, not
hidden, in each channel's saved metadata.

Usage:
    python run_doobh_exp2_general.py --channels exp2_s0,exp2_s2 --causal false
    python run_doobh_exp2_general.py --channels exp2_s1,exp2_s3,exp2_s4 --causal true
"""
import argparse
import atexit
import filecmp
import json
import os
import shutil
import signal
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

_EXP2_DIR = os.path.join(os.path.dirname(_HERE), "main_results")

ALL_CHANNELS = ["exp2_s0", "exp2_s1", "exp2_s2", "exp2_s3", "exp2_s4"]
RETROSPECTIVE = {"exp2_s0": False, "exp2_s1": True, "exp2_s2": False, "exp2_s3": True, "exp2_s4": False}

PROD_CSV = os.path.join(_ROOT, "explore", "standardization_results", "data_for_diffusion.csv")
CSV_BACKUP = PROD_CSV + ".orig_backup"

CONFIG_PY = os.path.join(_ROOT, "config", "config.py")
CONFIG_BACKUP = CONFIG_PY + ".orig_backup_exp2_run"
EVENT_CAUSAL_FALSE = "    event_causal: bool = False"
EVENT_CAUSAL_TRUE = "    event_causal: bool = True"
GROWTH_INFLATION_FULL = (
    '    growth_vars: Optional[List[str]] = field(default_factory=lambda: '
    '["indpro", "payems", "pi_transfer", "real_manf_trade", "personal_consump", "capacity_util"])\n'
    '    inflation_vars: Optional[List[str]] = field(default_factory=lambda: '
    '["cpi", "oil_price", "ppi", "hour_earnings"])'
)
GROWTH_INFLATION_CPI = (
    '    growth_vars: Optional[List[str]] = None\n'
    '    inflation_vars: Optional[List[str]] = field(default_factory=lambda: ["cpi"])'
)


def swap_in_channel_csv(channel: str, csv_suffix: str = ""):
    channel_csv = os.path.join(_ROOT, "explore", "standardization_results",
                                f"data_for_diffusion_{channel}{csv_suffix}.csv")
    if not os.path.exists(channel_csv):
        raise FileNotFoundError(f"{channel_csv} not found -- run build_channel_csvs_exp2_all.py first.")
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
    with open(CONFIG_PY) as f:
        text = f.read()
    if text.count(EVENT_CAUSAL_TRUE) == 1 and text.count(EVENT_CAUSAL_FALSE) == 0:
        print(f"{CONFIG_PY} already has event_causal=True -- nothing to flip (no backup needed).")
        return
    if not os.path.exists(CONFIG_BACKUP):
        shutil.copy2(CONFIG_PY, CONFIG_BACKUP)
        print(f"Backed up {CONFIG_PY} -> {CONFIG_BACKUP}")
    if text.count(EVENT_CAUSAL_FALSE) != 1:
        raise RuntimeError(f"Expected exactly 1 occurrence of {EVENT_CAUSAL_FALSE!r} in {CONFIG_PY}, "
                            f"found {text.count(EVENT_CAUSAL_FALSE)} -- refusing to edit.")
    with open(CONFIG_PY, "w") as f:
        f.write(text.replace(EVENT_CAUSAL_FALSE, EVENT_CAUSAL_TRUE))
    print(f"Flipped event_causal False -> True in {CONFIG_PY}")


def flip_cpi_only():
    with open(CONFIG_PY) as f:
        text = f.read()
    if GROWTH_INFLATION_CPI in text:
        print(f"{CONFIG_PY} already has CPI-only growth/inflation vars -- nothing to flip.")
        return
    if not os.path.exists(CONFIG_BACKUP):
        shutil.copy2(CONFIG_PY, CONFIG_BACKUP)
        print(f"Backed up {CONFIG_PY} -> {CONFIG_BACKUP}")
    if text.count(GROWTH_INFLATION_FULL) != 1:
        raise RuntimeError(f"Expected exactly 1 occurrence of the Full growth/inflation block in "
                            f"{CONFIG_PY}, found {text.count(GROWTH_INFLATION_FULL)} -- refusing to edit.")
    with open(CONFIG_PY, "w") as f:
        f.write(text.replace(GROWTH_INFLATION_FULL, GROWTH_INFLATION_CPI))
    print(f"Flipped growth/inflation vars Full -> CPI-only in {CONFIG_PY}")


def restore_config():
    if os.path.exists(CONFIG_BACKUP) and not filecmp.cmp(CONFIG_BACKUP, CONFIG_PY, shallow=False):
        shutil.copy2(CONFIG_BACKUP, CONFIG_PY)
        print(f"Restored original {CONFIG_PY} from backup")


def self_heal_on_startup():
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


def clear_stale_outputs(movable, results_root):
    stale_dir = os.path.join(_EXP2_DIR, results_root, "_stale_leftovers")
    for path in movable:
        if os.path.exists(path):
            os.makedirs(stale_dir, exist_ok=True)
            dest = os.path.join(stale_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"[warn] found pre-existing {path}, moved to {dest} before starting fresh run")


def move_outputs(channel, movable, results_root):
    out_dir = os.path.join(_EXP2_DIR, results_root, channel)
    os.makedirs(out_dir, exist_ok=True)
    for path in movable:
        if os.path.exists(path):
            dest = os.path.join(out_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"  moved {path} -> {dest}")
        else:
            print(f"  [note] {path} was not produced this run")
    with open(os.path.join(out_dir, "channel_meta.json"), "w") as f:
        json.dump({"channel": channel, "retrospective": RETROSPECTIVE[channel],
                    "note": "retrospective methods (S1/S3) were used as a Doob-h guidance channel "
                            "as explicitly requested; their own state construction needs a future "
                            "monthly anchor -- disclosed, not a bug." if RETROSPECTIVE[channel] else
                            "causal in its own state construction."}, f, indent=2)


def run_channel(channel, movable, score_ckpt, results_root, force_retrain=False, csv_suffix=""):
    print(f"\n{'=' * 70}\nChannel: {channel} (retrain={force_retrain}, csv_suffix={csv_suffix!r})\n{'=' * 70}")
    clear_stale_outputs(movable, results_root)
    swap_in_channel_csv(channel, csv_suffix=csv_suffix)
    cmd = [sys.executable, "-u", "main.py", "--no-wandb"]
    if force_retrain:
        cmd.append("--retrain")
    try:
        result = subprocess.run(cmd, cwd=_ROOT, check=False)
    finally:
        restore_csv()

    if result.returncode != 0:
        print(f"[warn] main.py exited {result.returncode} for channel '{channel}' "
              f"(expected: the known STEP 6 portfolio-analysis bug). Checking generation output...")

    move_outputs(channel, movable, results_root)
    gen_path = os.path.join(_EXP2_DIR, results_root, channel, "generated_samples_test.pt")
    if os.path.exists(gen_path):
        print(f"Channel '{channel}' done -> {os.path.join(_EXP2_DIR, results_root, channel)}")
    else:
        print(f"[error] Channel '{channel}' did NOT produce generated_samples_test.pt -- a real failure.")
    print(f"(score checkpoint: {score_ckpt}, {'reused/cached' if os.path.exists(score_ckpt) else 'will be trained fresh'})")


def main():
    self_heal_on_startup()
    _install_signal_handlers()
    atexit.register(restore_csv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", required=True, help="Comma-separated subset of: " + ",".join(ALL_CHANNELS))
    parser.add_argument("--causal", required=True, choices=["true", "false"])
    parser.add_argument("--cpi-only", action="store_true",
                         help="Use the CPI-only growth/inflation var selection (also flipped in config.py, "
                              "same backup/restore pattern as event_causal) and swap in the _cpi-suffixed "
                              "per-channel CSVs instead of the default (Full macro set) ones.")
    parser.add_argument("--own-std", action="store_true",
                         help="Macro embedding with PER-METHOD NN residuals: swap in data_for_diffusion_<channel>_ownstd.csv "
                              "(assets standardized on that method's own state), train a FRESH base diffusion model for "
                              "every channel (data differ per channel) and keep each one in its channel folder.")
    parser.add_argument("--standardization-dir", default=None,
                         help="--own-std: folder with <method>/mu.csv, sigma.csv (for the inverse transform artifacts)")
    args = parser.parse_args()
    if args.own_std and (args.cpi_only or not args.standardization_dir):
        raise ValueError("--own-std needs --standardization-dir and is Full-only here (no --cpi-only).")
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    for c in channels:
        if c not in ALL_CHANNELS:
            raise ValueError(f"Unknown channel {c!r}, expected one of {ALL_CHANNELS}")

    causal = args.causal == "true"
    csv_suffix = "_cpi" if args.cpi_only else ("_ownstd" if args.own_std else "")
    results_root = (f"doobh_results_exp2_{'causal' if causal else 'noncausal'}{'_cpi' if args.cpi_only else ''}"
                    f"{'_ownstd' if args.own_std else ''}")

    if causal:
        atexit.register(restore_config)
        flip_event_causal_true()
    if args.cpi_only:
        atexit.register(restore_config)
        flip_cpi_only()

    try:
        import importlib
        import config as config_module
        importlib.reload(config_module)
        cfg = config_module.get_default_config()
        if cfg.data.event_causal != causal:
            raise RuntimeError(f"config.data.event_causal={cfg.data.event_causal}, expected {causal} -- aborting.")
        run_tag = config_module.get_run_tag(cfg)
        print(f"Confirmed event_causal={causal}; run_tag={run_tag}")

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
        if args.own_std:
            # each channel trains on its own residuals -> its own base model; move it out after the channel
            # so it is kept with that channel and can never be reused by the next one
            movable += [score_ckpt, os.path.join(_ROOT, "ckpt_new", f"score_losses_{run_tag}.csv")]
            if os.path.exists(score_ckpt):
                raise RuntimeError(f"{score_ckpt} already exists -- refusing to reuse a base model in --own-std mode.")
        ckpt_exists = os.path.exists(score_ckpt)
        print(f"Score checkpoint for this run: {score_ckpt} "
              f"({'EXISTS -- will be reused' if ckpt_exists else 'does NOT exist -- training fresh on first channel'})")

        for i, channel in enumerate(channels):
            run_channel(channel, movable, score_ckpt, results_root,
                        force_retrain=True if args.own_std else (not ckpt_exists and i == 0), csv_suffix=csv_suffix)

        print(f"\n{'=' * 70}\nEvaluation\n{'=' * 70}")
        for channel in channels:
            print(f"=== doobh_evaluation.py --channel-override {channel} ===")
            eval_cmd = [sys.executable, "-u", "doobh_evaluation_exp2.py",
                        "--channel", channel, "--results-root", results_root]
            if csv_suffix:
                eval_cmd += ["--csv-suffix", csv_suffix]
            subprocess.run(eval_cmd, cwd=_HERE, check=False)

        print(f"\n{'=' * 70}\nSaving rare-event artifacts (generated + real windows, dates, inverse-transform inputs)\n{'=' * 70}")
        for channel in channels:
            art_cmd = [sys.executable, "-u", "save_rare_event_artifacts.py", "--group", "embedding",
                       "--channel", channel, "--results-root", results_root, "--csv-suffix", csv_suffix]
            if args.own_std:
                art_cmd += ["--standardization-dir", args.standardization_dir]
                subprocess.run(art_cmd, cwd=_HERE, check=False)

        print(f"\nAll done (causal={causal}). Score checkpoint: {score_ckpt}")
        print(f"Per-channel results under: {os.path.join(_EXP2_DIR, results_root)}/<channel>/")
    finally:
        if causal or args.cpi_only:
            restore_config()


if __name__ == "__main__":
    main()
