"""
Where a main.py run's outputs live, for the analysis/ evaluation scripts.

Everything is keyed by the checkpoint tag of the CURRENT config, get_run_tag(config) =
<conditioning_set>_<causal>_lag<gap>_<event_type><top>pct_..., so the analysis always reads the run that was trained
on config.data.csv_path (the conditioning set's data_for_diffusion.csv) with the same event definition:

    main.py writes   results/<run_tag>/generated_samples_{train,test}.pt, generated_samples_meta.json
                     ckpt_new/score_<run_tag>.pt, hfunction_<run_tag>.pt, score_losses_<run_tag>.csv, h_losses_<run_tag>.csv
    analysis writes  results/<run_tag>/unconditional_samples.pt (unconditional_gen.py; reused by cov.py)
                     analysis/diffusion_results/<run_tag>/*.png
"""
import json
import os

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ANALYSIS_DIR)


def run_tag(config):
    from config import get_run_tag
    return get_run_tag(config)


def run_output_dir(config):
    """main.py's per-run output folder (generated samples + their provenance)."""
    return os.path.join(ROOT, "results", run_tag(config))


def analysis_results_dir(config):
    d = os.path.join(ANALYSIS_DIR, "diffusion_results", run_tag(config))
    os.makedirs(d, exist_ok=True)
    return d


def score_ckpt(config):
    return os.path.join(ROOT, "ckpt_new", f"score_{run_tag(config)}.pt")


def hfunction_ckpt(config):
    return os.path.join(ROOT, "ckpt_new", f"hfunction_{run_tag(config)}.pt")


def unconditional_samples_path(config):
    return os.path.join(run_output_dir(config), "unconditional_samples.pt")


def generated_samples_path(config, split="train"):
    """Path of main.py's conditional samples for the current config, after checking their provenance:
    generated_samples_meta.json must name the same data file and run tag as the current config."""
    d = run_output_dir(config)
    path = os.path.join(d, f"generated_samples_{split}.pt")
    meta_path = os.path.join(d, "generated_samples_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run main.py with conditioning_set="
                                f"{config.data.conditioning_set!r} (run_tag {run_tag(config)}) first")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found -- cannot confirm which data produced {path}")
    meta = json.load(open(meta_path))
    if os.path.abspath(meta["data_file"]) != os.path.abspath(config.data.csv_path) or meta["run_tag"] != run_tag(config):
        raise RuntimeError(f"{path} was generated from {meta['data_file']} (run_tag {meta['run_tag']}), not from the "
                           f"current config's {config.data.csv_path} (run_tag {run_tag(config)})")
    print(f"[run] samples {path}\n[run] data    {meta['data_file']}  (conditioning_set={meta['conditioning_set']})")
    return path
