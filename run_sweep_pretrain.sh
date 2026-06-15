#!/bin/bash

# Pretrain-baseline sweep: compare conditional samples against
# unconditional pretrain samples that satisfy the event condition.
#
# Usage: bash run_sweep_pretrain.sh
#
# Baseline: N_PRETRAIN unconditional samples generated per run,
#           filtered to those satisfying the event condition.
# Sweep:    STOCH_VALUES × ETA_VALUES × Q_FLAGS combinations.

RESULTS_DIR="results/sweep_pretrain"
mkdir -p "$RESULTS_DIR"
mkdir -p logs/sweep_pretrain

# ── Pretrain baseline settings ────────────────────────────────────────────────
# Each script call generates (N_events * PRETRAIN_OVERSAMPLE) unconditional
# samples and filters for events. Increase if too few events are found.
PRETRAIN_OVERSAMPLE=20

# Pretrain events cache: generated once on first run, reused for all sweep combos.
PRETRAIN_CACHE="$RESULTS_DIR/pretrain_events_cache.pt"

# ── Sweep grid ────────────────────────────────────────────────────────────────
STOCH_VALUES=(0 1)
ETA_VALUES=(0 -0.5 -1.0 -1.5)
Q_FLAGS=("" "--use-q-model")

# ── Count total combinations ──────────────────────────────────────────────────
total=0
for Q_FLAG in "${Q_FLAGS[@]}"; do
    for STOCH in "${STOCH_VALUES[@]}"; do
        for ETA in "${ETA_VALUES[@]}"; do
            total=$((total + 1))
        done
    done
done

# ── Run sweep ─────────────────────────────────────────────────────────────────
count=0
for Q_FLAG in "${Q_FLAGS[@]}"; do
    Q_LABEL=$([ -n "$Q_FLAG" ] && echo "with_q" || echo "no_q")

    for STOCH in "${STOCH_VALUES[@]}"; do
        for ETA in "${ETA_VALUES[@]}"; do
            count=$((count + 1))
            TAG="${Q_LABEL}_stoch${STOCH}_eta${ETA}"

            echo ""
            echo "=========================================="
            echo "[$count/$total] $TAG"
            echo "=========================================="

            # In-sample
            python -u sample_insample.py \
                $Q_FLAG \
                --stoch "$STOCH" \
                --eta "$ETA" \
                --results-dir "$RESULTS_DIR" \
                --run-suffix "$TAG" \
                --compare-pretrain \
                --pretrain-oversample "$PRETRAIN_OVERSAMPLE" \
                --pretrain-cache "$PRETRAIN_CACHE" \
                --no-wandb \
                2>&1 | tee "logs/sweep_pretrain/insample_${TAG}.log"

            if [ $? -ne 0 ]; then
                echo "✗ In-sample FAILED: $TAG"
            else
                echo "✓ In-sample done: $TAG"
            fi

            # Out-of-sample (real test data vs conditional)
            python -u sample_outsample.py \
                $Q_FLAG \
                --stoch "$STOCH" \
                --eta "$ETA" \
                --results-dir "$RESULTS_DIR" \
                --run-suffix "$TAG" \
                --no-wandb \
                2>&1 | tee "logs/sweep_pretrain/outsample_${TAG}.log"

            if [ $? -ne 0 ]; then
                echo "✗ Out-of-sample FAILED: $TAG"
            else
                echo "✓ Out-of-sample done: $TAG"
            fi

        done
    done
done

echo ""
echo "=========================================="
echo "Sweep complete! $total combinations done."
echo "Plots and CSVs saved in $RESULTS_DIR/"
echo "=========================================="
ls -lh "$RESULTS_DIR"/*.png 2>/dev/null | awk '{print $NF}' | xargs -I{} basename {}
