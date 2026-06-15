"""
Configuration file for the CDG Financial Time Series project
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data configuration"""
    csv_path: str = "Stocks_logret.csv"
    tickers: List[str] = None
    weekday_col: str = "weekday"
    seq_len: int = 64
    test_days: int = 700          # used only when train_end_date is None
    start_date: str = "2014-01-28"   # data window start (None = use all)
    end_date: str = "2025-10-17"         # data window end (None = use all)
    train_end_date: str = "2022-11-07"  # last day of train set (None = use test_days)
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["AAPL", "AMZN", "JPM", "TSLA"]


@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    device: str = "cuda"
    in_channels: int = 4
    out_channels: int = 4
    sample_size: int = 64
    layers_per_block: int = 2
    block_out_channels: Tuple[int, int, int] = (64, 256, 512)

    # VP diffusion parameters
    b_min: float = 0.1
    b_max: float = 10 #3.25

    # Training parameters
    batch_size: int = 256
    n_epochs: int = 600
    learning_rate: float = 1e-4
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5

    # Sampling parameters
    num_steps: int = 100
    eps: float = 1e-4

    # Architecture: "unet" or "transformer"
    arch: str = "transformer"

    # Transformer-specific parameters (used when arch="transformer")
    embed_dim: int = 256
    n_heads: int = 8
    n_layers: int = 8
    cond_dim: int = 128


@dataclass
class HFunctionConfig:
    """H-function training configuration"""
    device: str = "cuda"
    asset_dim: int = 4
    time_steps: int = 64
    embed_dim: int = 128

    # Training parameters
    train_batch_size: int = 2**13  # number of diffusion paths generated for training data
    train_stoch: float = 0.5         # stochasticity for generating training paths (0=ODE, 1=full SDE)
    h_mini_batch_size: int = 512   # mini-batch size per gradient step (keep small for Transformer)
    n_epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5

    # Event condition
    event_asset_idx: int = 3       # TSLA index
    event_window: int = 10         # last 10 days
    event_threshold_raw: float = -0.10   # raw log return sum ≤ -10%; source of truth
    event_threshold: float = -3.0        # computed at runtime: event_threshold_raw / sigma_TSLA; do not set manually

    # Architecture: "transformer" or "cnn"
    arch: str = "transformer"
    n_heads: int = 4
    n_layers: int = 4
    cond_dim: int = 64


@dataclass
class ConditionalGenConfig:
    """Conditional generation configuration"""
    device: str = "cuda"
    batch_size: int = 128
    num_steps: int = 100
    stoch: float = 0 #0.3
    eta: float = 0 #150.0
    use_q_model: bool = False

    # Q-model training hyperparameters
    q_model_epochs: int = 300
    q_model_lr: float = 1e-4
    q_model_train_batch_size: int = 2**12  # number of diffusion paths generated for Q-model training
    q_model_mini_batch_size: int = 256     # mini-batch size per gradient step (keep small for Transformer)
    q_model_train_stoch: float = 0.5 #0.2       # stochasticity for generating Q-model training paths

    # Q-model Transformer architecture (smaller than score network — easier task)
    q_embed_dim: int = 128
    q_n_heads: int = 4
    q_n_layers: int = 4
    q_cond_dim: int = 64


@dataclass
class PortfolioConfig:
    """Portfolio analysis configuration"""
    window_for_cov: int = 54  # Use first 54 days to compute covariance
    last_days_sum: int = 5    # Compute sum of last 5 days for portfolio


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    enabled: bool = True
    project: str = "cdg-finance"
    entity: str = None  # Your wandb username or team name
    run_name: str = None  # Auto-generated if None
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
