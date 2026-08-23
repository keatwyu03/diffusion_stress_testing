#!/bin/bash
# Clear old BFK stationarity results and submit the per-asset array job
# (bfk_array_job.sh) that does the actual work.
#
# Usage:
#   bash explore/submit_bfk_jobs.sh

set -euo pipefail

REPO=/afs/cs.stanford.edu/u/kadenwu/Conditional_diffusion

rm -f "$REPO/explore/bfk_results"/*.json
echo "cleared old results in $REPO/explore/bfk_results/"

sbatch "$REPO/explore/bfk_array_job.sh"
