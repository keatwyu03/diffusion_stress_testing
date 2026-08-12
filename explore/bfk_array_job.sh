#!/bin/bash
#SBATCH --job-name=bfk_stationarity
#SBATCH --partition=soal
#SBATCH --account=soal
#SBATCH --array=0-9%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/afs/cs.stanford.edu/u/kadenwu/Conditional_diffusion/explore/bfk_results/logs/%x_%A_%a.out
#SBATCH --error=/afs/cs.stanford.edu/u/kadenwu/Conditional_diffusion/explore/bfk_results/logs/%x_%A_%a.err
#
# One SLURM array task per asset for the BFK stationarity test
# (bfk_stationarity.py), capped at 2 concurrently running jobs (`%2`) since
# stDistAutocop is single-threaded and this cluster node has 2 usable cores.
#
# Do not run this directly with `sbatch bfk_array_job.sh` -- use
# `bash submit_bfk_jobs.sh` instead, which clears old results first.
#
# Each task writes explore/bfk_results/<ASSET>.json when done. Once every
# array task has finished (check with: squeue -u $USER), combine results:
#   source venv/bin/activate
#   python explore/bfk_stationarity.py combine

set -euo pipefail

REPO=/afs/cs.stanford.edu/u/kadenwu/Conditional_diffusion
ASSETS=(IBM CSCO AAPL MSFT ORCL INTC TXN QCOM AMAT ADBE)
ASSET="${ASSETS[$SLURM_ARRAY_TASK_ID]}"

cd "$REPO"

source "$REPO/venv/bin/activate"
python "$REPO/explore/bfk_stationarity.py" run-asset "$ASSET"
