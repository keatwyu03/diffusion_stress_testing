#!/bin/bash

# Training script for Conditional Diffusion Generation
# This script trains all models: Diffusion, H-function, and Q-model

echo "=========================================="
echo "CDG Finance - Model Training"
echo "=========================================="

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# ==================== STEP 1: Train Diffusion Model and H-Function ====================
echo ""
echo "[STEP 1/2] Training Diffusion Model and H-Function"
echo "------------------------------------------"
python -u main.py \
    --skip-qmodel-training \
    2>&1 | tee logs/train_diffusion_hfunction.log

# Check if diffusion and h-function training succeeded
if [ ! -f "checkpoints/diffusion_model.pt" ] || [ ! -f "checkpoints/hfunction.pt" ]; then
    echo "ERROR: Diffusion or H-function training failed!"
    exit 1
fi

echo "✓ Diffusion model and H-function training completed"

# ==================== STEP 2: Train Q-Model ====================
echo ""
echo "[STEP 2/2] Training Q-Model"
echo "------------------------------------------"
python -u main.py \
    --skip-diffusion-training \
    --skip-hfunction-training \
    2>&1 | tee logs/train_qmodel.log

# Check if Q-model training succeeded
if [ ! -f "checkpoints/q_model.pt" ]; then
    echo "ERROR: Q-model training failed!"
    exit 1
fi

echo "✓ Q-model training completed"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Trained models saved in checkpoints/:"
echo "  - diffusion_model.pt"
echo "  - hfunction.pt"
echo "  - q_model.pt"
echo ""
echo "Next step: Run ./run_sampling.sh to generate samples and analyze portfolios"
echo "=========================================="
