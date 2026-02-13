#!/bin/bash

# Sampling and analysis script for both in-sample and out-of-sample
# This script generates conditional samples and performs portfolio analysis

echo "=========================================="
echo "CDG Finance - Sampling & Analysis"
echo "=========================================="

# Configuration
NUM_STEPS=100      # Number of sampling steps
STOCH=0.5          # Stochasticity parameter (0=deterministic, 1=full stochastic)
ETA=2.0            # Conditional guidance strength
BATCH_SIZE=64      # Batch size for generation

# Create necessary directories
mkdir -p logs
mkdir -p results

# Check if models exist
if [ ! -f "checkpoints/diffusion_model.pt" ] || [ ! -f "checkpoints/hfunction.pt" ]; then
    echo "ERROR: Required models not found in checkpoints/"
    echo "Please run ./run_training.sh first"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  num_steps: $NUM_STEPS"
echo "  stoch: $STOCH"
echo "  eta: $ETA"
echo "  batch_size: $BATCH_SIZE"
echo ""

# ==================== Experiment 1: In-Sample WITHOUT Q-model ====================
echo "=========================================="
echo "[1/4] In-Sample Analysis (without Q-model)"
echo "=========================================="
python -u sample_insample.py \
    --num-steps $NUM_STEPS \
    --stoch $STOCH \
    --eta $ETA \
    --batch-size $BATCH_SIZE \
    --run-suffix "baseline" \
    2>&1 | tee logs/insample_no_q.log

if [ $? -eq 0 ]; then
    echo "✓ In-sample (no Q-model) completed"
else
    echo "✗ In-sample (no Q-model) failed"
fi

# ==================== Experiment 2: In-Sample WITH Q-model ====================
echo ""
echo "=========================================="
echo "[2/4] In-Sample Analysis (with Q-model)"
echo "=========================================="

if [ -f "checkpoints/q_model.pt" ]; then
    python -u sample_insample.py \
        --use-q-model \
        --num-steps $NUM_STEPS \
        --stoch $STOCH \
        --eta $ETA \
        --batch-size $BATCH_SIZE \
        --run-suffix "baseline" \
        2>&1 | tee logs/insample_with_q.log

    if [ $? -eq 0 ]; then
        echo "✓ In-sample (with Q-model) completed"
    else
        echo "✗ In-sample (with Q-model) failed"
    fi
else
    echo "⚠ Q-model not found, skipping this experiment"
    echo "Run ./run_training.sh to train Q-model"
fi

# ==================== Experiment 3: Out-of-Sample WITHOUT Q-model ====================
echo ""
echo "=========================================="
echo "[3/4] Out-of-Sample Analysis (without Q-model)"
echo "=========================================="
python -u sample_outsample.py \
    --num-steps $NUM_STEPS \
    --stoch $STOCH \
    --eta $ETA \
    --batch-size $BATCH_SIZE \
    --run-suffix "baseline" \
    2>&1 | tee logs/outsample_no_q.log

if [ $? -eq 0 ]; then
    echo "✓ Out-of-sample (no Q-model) completed"
else
    echo "✗ Out-of-sample (no Q-model) failed"
fi

# ==================== Experiment 4: Out-of-Sample WITH Q-model ====================
echo ""
echo "=========================================="
echo "[4/4] Out-of-Sample Analysis (with Q-model)"
echo "=========================================="

if [ -f "checkpoints/q_model.pt" ]; then
    python -u sample_outsample.py \
        --use-q-model \
        --num-steps $NUM_STEPS \
        --stoch $STOCH \
        --eta $ETA \
        --batch-size $BATCH_SIZE \
        --run-suffix "baseline" \
        2>&1 | tee logs/outsample_with_q.log

    if [ $? -eq 0 ]; then
        echo "✓ Out-of-sample (with Q-model) completed"
    else
        echo "✗ Out-of-sample (with Q-model) failed"
    fi
else
    echo "⚠ Q-model not found, skipping this experiment"
fi

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved in results/:"
ls -lh results/*.png 2>/dev/null || echo "  (no plots generated)"
echo ""
echo "Logs saved in logs/:"
ls -lh logs/*.log 2>/dev/null || echo "  (no logs found)"
echo ""
echo "=========================================="
