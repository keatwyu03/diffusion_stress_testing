# Conditional Diffusion Generation for Financial Time Series

A Python framework for generating conditional financial time series using variance-preserving (VP) diffusion models with event-based conditioning.

## Overview

This project implements conditional diffusion models for financial time series generation, enabling generation of synthetic samples that match specific market conditions. The framework supports:

- **Diffusion-based generation** using VP-SDE models
- **Event-based conditioning** via H-function
- **Q-model acceleration** for faster conditional sampling
- **Portfolio analysis** comparing generated vs. real samples
- **In-sample and out-of-sample** evaluation

## Project Structure

```
cdg_finance/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration dataclasses
├── data/
│   ├── __init__.py
│   └── data_processor.py      # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── diffusion_model.py     # VP diffusion model (UNet1D)
│   ├── hfunction.py           # H-function for event conditioning
│   └── conditional_generator.py  # Conditional sampling with optional Q-model
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # Utility functions
│   └── portfolio.py           # Portfolio analysis (Min-Var, Risk-Parity, Equal-Weight)
├── main.py                     # Full pipeline script
├── sample_insample.py          # In-sample analysis script
├── sample_outsample.py         # Out-of-sample analysis script
├── run_training.sh             # Training automation script
├── run_sampling.sh             # Sampling automation script
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd cdg_finance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place `Stocks_logret.csv` in the project root directory
   - CSV format: columns should include `Date`, ticker symbols (e.g., `AAPL`, `AMZN`, `JPM`, `TSLA`), and `weekday`

## Quick Start

### Option 1: Automated Scripts (Recommended)

#### Step 1: Train All Models
```bash
./run_training.sh
```

This will:
1. Train the diffusion model and H-function
2. Train the Q-model for faster sampling
3. Save all checkpoints to `checkpoints/`

#### Step 2: Run Sampling & Analysis
```bash
./run_sampling.sh
```

This will:
1. Generate in-sample samples (with/without Q-model)
2. Generate out-of-sample samples (with/without Q-model)
3. Analyze portfolio performance for all experiments
4. Save results to `results/` and logs to `logs/`

### Option 2: Manual Execution

#### Training

1. **Full training pipeline** (diffusion + H-function + Q-model):
```bash
python main.py
```

2. **Train only diffusion and H-function** (skip Q-model):
```bash
python main.py --skip-qmodel-training
```

3. **Train only Q-model** (requires pre-trained diffusion and H-function):
```bash
python main.py --skip-diffusion-training --skip-hfunction-training
```

#### Sampling & Analysis

1. **In-sample analysis** without Q-model:
```bash
python sample_insample.py --num-steps 100 --stoch 0.5 --eta 2.0
```

2. **In-sample analysis** with Q-model:
```bash
python sample_insample.py --use-q-model --num-steps 100 --stoch 0.5 --eta 2.0
```

3. **Out-of-sample analysis** without Q-model:
```bash
python sample_outsample.py --num-steps 100 --stoch 0.5 --eta 2.0
```

4. **Out-of-sample analysis** with Q-model:
```bash
python sample_outsample.py --use-q-model --num-steps 100 --stoch 0.5 --eta 2.0
```

## Configuration

### Sampling Parameters

Configure sampling behavior via command-line arguments:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--num-steps` | Number of diffusion sampling steps (higher = more accurate, slower) | 200 | 50-500 |
| `--stoch` | Stochasticity (0=deterministic ODE, 1=full SDE) | 0.3 | 0.0-1.0 |
| `--eta` | Conditional guidance strength (higher = stronger conditioning) | 150.0 | 0.5-500.0 |
| `--batch-size` | Batch size for generation | 32 | 16-128 |
| `--use-q-model` | Use Q-model for faster sampling | False | - |
| `--no-wandb` | Disable Weights & Biases logging | False | - |
| `--run-suffix` | Suffix for output filenames | "" | - |

### Model Configuration

Modify `config/config.py` to customize:
- **Data**: tickers, sequence length, train/test split
- **Diffusion model**: architecture, channels, training hyperparameters
- **H-function**: event definition, asset index, threshold
- **Conditional generation**: Q-model settings
- **Portfolio analysis**: covariance window, return aggregation window

## Pipeline Details

### 1. Data Processing

- **Load** financial returns from CSV
- **Remove weekday effect** via day-of-week averaging
- **Standardize** returns to zero mean, unit variance
- **Winsorize** to clip extreme outliers
- **Create sequences** of fixed length (default: 64 days)
- **Split** into train/test (default: last 700 days for testing)

### 2. Model Training

#### Diffusion Model
- **Architecture**: UNet1D from HuggingFace Diffusers
- **Method**: Variance Preserving (VP) SDE
- **Training**: Score matching on winsorized sequences
- **Output**: `checkpoints/diffusion_model.pt`

#### H-Function
- **Purpose**: Predict event probability from noisy samples
- **Training**: Binary classification on synthetic paths
- **Event**: User-defined condition (e.g., sum of last 5 days < threshold)
- **Output**: `checkpoints/hfunction.pt`

#### Q-Model (Optional)
- **Purpose**: Approximate ∇H for faster conditional sampling
- **Training**: Supervised learning on pre-computed gradients
- **Speedup**: Eliminates autograd during sampling
- **Output**: `checkpoints/q_model.pt`

### 3. Conditional Generation

- **Method**: Guided diffusion using H-function gradients
- **Conditioning**: Generate samples matching event criteria
- **Sampling**: Euler-Maruyama with optional Q-model acceleration

### 4. Portfolio Analysis

Three portfolio strategies are evaluated:

1. **Min-Variance**: Minimize portfolio variance given covariance
2. **Risk-Parity**: Equal risk contribution from each asset
3. **Equal-Weight**: Uniform allocation across assets

**Metrics**: Mean, median, std, 5th/10th percentiles of last N-day returns

## Output Files

### Checkpoints
- `checkpoints/diffusion_model.pt` - Trained diffusion model
- `checkpoints/hfunction.pt` - Trained H-function
- `checkpoints/q_model.pt` - Trained Q-model (if enabled)

### Results
- `results/portfolio_insample_*.png` - In-sample comparison plots
- `results/portfolio_outsample_*.png` - Out-of-sample comparison plots

### Logs
- `logs/train_*.log` - Training logs
- `logs/insample_*.log` - In-sample sampling logs
- `logs/outsample_*.log` - Out-of-sample sampling logs

## Weights & Biases Integration

The framework supports automatic experiment tracking via [wandb](https://wandb.ai/).

### Enable wandb
```bash
# Set your wandb entity in config/config.py
python main.py  # wandb enabled by default
```

### Disable wandb
```bash
python main.py --no-wandb
python sample_insample.py --no-wandb
```

### Privacy: Remove wandb Credentials

To remove wandb API keys and cached credentials before sharing:

```bash
# Remove wandb config
rm -rf ~/.netrc
rm -rf ~/.config/wandb/

# Remove wandb cache from project
rm -rf wandb/
rm -rf .wandb/

# Clear wandb settings from config
# Edit config/config.py and set:
#   entity: str = None  # Remove your username/team
```

## Example Output

```
=== In-Sample Portfolio Comparison Statistics ===
[GENERATED (in-sample)] N=450 |
  MV mean=0.0023, median=0.0018, std=0.0312, p5=-0.0421, p10=-0.0289 |
  RP mean=0.0019, median=0.0015, std=0.0298, p5=-0.0398, p10=-0.0276 |
  Avg mean=0.0021, median=0.0016, std=0.0305, p5=-0.0410, p10=-0.0282

[REAL TRAIN] N=450 |
  MV mean=-0.0089, median=-0.0073, std=0.0345, p5=-0.0612, p10=-0.0441 |
  RP mean=-0.0082, median=-0.0068, std=0.0331, p5=-0.0589, p10=-0.0423 |
  Avg mean=-0.0085, median=-0.0070, std=0.0338, p5=-0.0601, p10=-0.0432
```

## Advanced Usage

### Modify Sampling Scripts

Edit `run_sampling.sh` to change default parameters:

```bash
NUM_STEPS=150      # Increase for higher quality
STOCH=0.3          # Lower for more deterministic
ETA=3.0            # Higher for stronger conditioning
BATCH_SIZE=64      # Adjust based on GPU memory
```

### Custom Event Definitions

Edit `config/config.py` to define custom events:

```python
@dataclass
class HFunctionConfig:
    event_asset_idx: int = 0        # Which asset to condition on
    event_window: int = 5           # Last N days
    event_threshold: float = -0.1   # Threshold for sum
```

### Parallel Experiments

Run multiple configurations in parallel (requires multiple GPUs):

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python sample_insample.py --stoch 0.2 --run-suffix "low_stoch" &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python sample_insample.py --stoch 0.8 --run-suffix "high_stoch" &
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python sample_insample.py --batch-size 16
```

### Slow Sampling
```bash
# Train and use Q-model
./run_training.sh
python sample_insample.py --use-q-model
```

### No Space Left on Device
```bash
# Clean temporary files
rm -rf ~/.vscode-tmp/pymp-*
pip cache purge
conda clean --all
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cdg_finance_2025,
  title={Conditional Diffusion Generation for Financial Time Series},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- Variance Preserving SDE formulation based on [Score-Based Generative Models](https://arxiv.org/abs/2011.13456)
