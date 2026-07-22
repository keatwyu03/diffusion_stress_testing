"""Steps
1. Train score model (steins score function for the backward drift correction term)
   - Used for the non conditional diffusion model
   - s'(t,x) = spre + grad log h -> this trains the spre (hard constraint)
   - s'(t, x) = spre + tradoff * fine adjustment (soft constraint)

2. Train H function
3. Train Q function
(Steps 2, 3 above are used to get grad log h = q / h)

#Up to this step we have s'(t,x) and can run the backwards
diffusion model with constraints

dXt = (B(t-t, Xt) + sigma^2(t) * s'(t,xt))dt + sigma(t) dBt
or VP: dXt = (-0.5 * beta Xt + beta(t) * s'(t,xt))dt + sqrt(beta(t))dBt

4. Generate and run the conditional reverse SDE
5. Portfolio Analysis

1. Augment monthlydata to daily data (+flag)
2. unemployment (standardize)
3. in_out sample error (email)
4. Buy&hold sp500
5. Augment
"""


"""
Configuration file for the CDG Financial Time Series project
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

# Project root (parent of this config/ directory) — anchors data paths so
# scripts work regardless of the current working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DataConfig:
    """Data configuration"""
    # Where to load the data. Training/Testing split
    # Outlier handling -> winsorize upper and lower = % bounds to remove
    csv_path: str = os.path.join(_ROOT, "explore", "macro_data_new.csv")
    latent_method: Optional[str] = "state_space"    # Choose between state space, tracking regression, or None

    start_date : str = "2000-01-01"
    end_date: str = "2026-07-08"      # data window end (None = use all)

    window_shift : int = 1
    tickers: List[str] = None
    weekday_col: str = "weekday"
    seq_len: int = 10
    test_days: int = 1200             # used only when train_end_date is None
    train_end_date: str = None        # last day of train set (None = use test_days)
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    ema_span: int = 60                # per-window causal EMA standardizer span
                                       # (z = (r - EMA_mean)/EMA_vol at window entry)

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["T10YFF", "AAPL", "ORCL", "MSFT", "IBM"]


@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    #In and out channels are used to train the neural network for score function
    #This is the part that controls the denosing method for training s_theta

    device: str = field(default_factory=_default_device)
    in_channels: int = 4
    out_channels: int = 4
    sample_size: int = 10

    #number of layers for each CNN and number of parameters in each of those CNN layers
    layers_per_block: int = 3
    block_out_channels: Tuple[int, int, int] = (64, 128, 256)

    # Variance Preserving diffusion parameters
    b_min: float = 0.1
    b_max: float = 10.0

    # Training parameters
    batch_size: int = 75               #Stochastic minibatch gradient descent
    n_epochs: int = 20                #Number of times to loop through the data
    learning_rate: float = 1e-4        #Alpha Stepsize
    weight_decay: float = 0.0          #AdamW weight decay
    scheduler_patience: int = 50       #Check convergence every X number of loops through the data
    scheduler_factor: float = 0.5      #Multiplier for the Learning rate when plateau
    block_sampling: bool = True       #Round Robin SGD sampling for batches 

    # Sampling parameters
    num_steps: int = 500               #Number of noisy elements to add (larger bmax necessitates larger num_steps)
    eps: float = 1e-4                  #Stopping point of when we claim the data are now normal

    # Architecture: "unet" or "transformer"
    arch: str = "transformer"

    # Transformer-specific parameters (used when arch="transformer")
    embed_dim: int = 256
    n_heads: int = 16
    n_layers: int = 8
    cond_dim: int = 256
    
    cov_weight: float = 0.0
    cov_t_max: float = 0.3             # only low-t batch examples (one-step reconstruction
                                        # is trustworthy) are used for the covariance penalty


@dataclass
class HFunctionConfig:
    """H-function training configuration"""
    #Doobs H modification for the terminal constraints
    #Using a dual-axis transformer to learn this

    # Training parameters
    train_batch_size: int = 126        # number of noisy trajectories for unconditional diffusion
    train_stoch: float = 0.5           # stochasticity for generating training paths (0=ODE, 1=full SDE)
    h_mini_batch_size: int = 256       # mini-batch size per gradient step (smaller = more steps/epoch)
    block_sampling: bool = False        
    episode_reweight: bool = True      

    n_epochs: int = 800               # number of times to go through the data
    learning_rate: float = 1e-4        # step size for SGD
    weight_decay: float = 5e-4         # penalty to prevent overfitting
    scheduler_patience: int = 75
    scheduler_factor: float = 0.5
    h_t_max: float = 0.9               # cap on tau during training AND guidance application at

    # Event condition
    event_type: str = "upper_change"         # "absval", "abs_change", "upper_change", or "lower_change"
    event_asset_idx: int = 0           # which asset to watch for the shock
    event_window: int = 10             # lookback period
    event_threshold: float = 0.05       # top X% of |Z_end - Z_start| counts as an event
                                        # (e.g. 0.10 = top 10%), converted to a raw
                                        # numeric cutoff from train data at startup —
                                        # see get_event_threshold_from_percentile()

    # Constraint mode
    constraint_mode: str = "hard"      # "hard": 0/1 labels at the threshold
                                        # "soft": sigmoid(sharpness*(metric - threshold)) labels —
                                        #   every window gets a graded event-likeness label
    reward_sharpness: float = 5.0      # sigmoid steepness in soft mode; metric/threshold are in
                                        # standardized units (threshold ~1.2-1.5 std), so ~5 spreads
                                        # the ramp over a meaningful band of windows (50 would be
                                        # nearly indistinguishable from hard labels)

    # Architecture: "transformer" or "cnn"
    arch: str = "transformer"
    one_two_step: str = "one" #or two step


    device: str = field(default_factory=_default_device)
    asset_dim: int = 4
    time_steps: int = 10
    embed_dim: int = 256

    n_heads: int = 8
    n_layers: int = 6
    cond_dim: int = 256
    dropout: float = 0.0


@dataclass
class ConditionalGenConfig:
    """Conditional generation configuration"""
    #Learning the q function for conditional generation
    #Neural Network to get us grad log h = q / h

    device: str = field(default_factory=_default_device)
    batch_size: int = 32
    num_steps: int = 500
    stoch: float = 1.0
    eta: float = 1
    use_q_model: bool = False
    stop_early_steps: int = 5          # stop this many steps before the reverse SDE
                                        # reaches t=eps, leaving residual noise/diversity
                                        # instead of fully resolving to the sharp end state
    n_gen_samples: int = 5000          # number of samples to generate for train/test each,
                                        # independent of the real event count — reduces
                                        # Monte Carlo noise in the generated-side KDE estimate

    # Q-model training hyperparameters
    q_model_epochs: int = 500
    q_model_lr: float = 1e-4
    q_model_train_batch_size: int = 2**12   # number of diffusion paths for Q-model training
    q_model_mini_batch_size: int = 256      # mini-batch size per gradient step
    q_model_train_stoch: float = 0.5        # stochasticity for Q-model training paths

    # Q-model Transformer architecture
    q_embed_dim: int = 64
    q_n_heads: int = 4
    q_n_layers: int = 4
    q_cond_dim: int = 64

    # Constraint mode
    constraint_mode: str = "hard"      # "hard" or "soft"
    beta: float = 1.0                  # trade-off denominator parameter


@dataclass
class PortfolioConfig:
    """Portfolio analysis configuration"""
    window_for_cov: int = 54           # use first 54 days to compute covariance
    last_days_sum: int = 5             # compute sum of last 5 days for portfolio
    portfolio_tickers: List[str] = None

    def __post_init__(self):
        if self.portfolio_tickers is None:
            self.portfolio_tickers = ["AAPL", "ORCL", "MSFT", "IBM"]


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    enabled: bool = False
    project: str = "cdg-finance"
    entity: str = None
    run_name: str = None
    tags: List[str] = None
    notes: str = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["diffusion", "finance"]


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = None
    diffusion: DiffusionConfig = None
    hfunction: HFunctionConfig = None
    conditional: ConditionalGenConfig = None
    portfolio: PortfolioConfig = None
    wandb: WandbConfig = None

    seed: int = 2025

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.diffusion is None:
            self.diffusion = DiffusionConfig()
        if self.hfunction is None:
            self.hfunction = HFunctionConfig()
        if self.conditional is None:
            self.conditional = ConditionalGenConfig()
        if self.portfolio is None:
            self.portfolio = PortfolioConfig()
        if self.wandb is None:
            self.wandb = WandbConfig()


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
