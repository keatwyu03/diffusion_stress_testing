"""
Runs the analysis/ evaluation scripts (losses.py, conditional_gen.py, cov.py,
dependency_metric.py, distribution_metrics.py) against either the diffusion
model's generated samples (default) or the cGAN baseline's (--gan). Under
--gan, losses.py plots the cGAN's joint minimax value V(D,G) = -d_loss
instead of the diffusion model's score/H-function losses.

Each script is a standalone script (not an importable module), so this just
invokes them as subprocesses with the same interpreter and forwards --gan.
Diffusion-mode output goes to analysis/diffusion_results/; GAN-mode output
goes to analysis/gan_results/ (each script creates its own subset of files
there; see each script's own docstring/comments for exactly what it writes).

Usage (from the repo root):
    python analysis/evaluation_main.py            # diffusion mode (generated_samples_{train,test}.pt)
    python analysis/evaluation_main.py --gan      # cGAN mode (gan_baseline/gan_results/gan_generated_samples_{train,test}.pt)
"""
import argparse
import os
import subprocess
import sys

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_ANALYSIS_DIR)

SCRIPTS = [
    "losses.py",
    "conditional_gen.py",
    "cov.py",
    "dependency_metric.py",
    "distribution_metrics.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gan", action="store_true",
                         help="evaluate the cGAN baseline's generated samples instead of "
                              "the diffusion model's")
    args = parser.parse_args()

    results_dir = "gan_results" if args.gan else "diffusion_results"
    print(f"Running evaluation scripts in {'cGAN' if args.gan else 'diffusion'} mode "
          f"-> analysis/{results_dir}/\n")

    failures = []
    for script in SCRIPTS:
        script_path = os.path.join(_ANALYSIS_DIR, script)
        cmd = [sys.executable, script_path]
        if args.gan:
            cmd.append("--gan")

        print("=" * 72)
        print(f"Running: {' '.join(os.path.relpath(c, _ROOT) if os.path.isabs(c) else c for c in cmd)}")
        print("=" * 72)
        result = subprocess.run(cmd, cwd=_ROOT)
        if result.returncode != 0:
            print(f"\n[evaluation_main] {script} FAILED (exit code {result.returncode})")
            failures.append(script)
        print()

    print("=" * 72)
    if failures:
        print(f"Completed with {len(failures)} failure(s): {failures}")
        sys.exit(1)
    else:
        print(f"All {len(SCRIPTS)} evaluation scripts completed successfully.")
        print(f"Outputs saved to analysis/{results_dir}/")


if __name__ == "__main__":
    main()
