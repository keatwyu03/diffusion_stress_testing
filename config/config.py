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
    test_days: int = 700
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
    layers_per_block: int = 3
    block_out_channels: Tuple[int, int, int] = (128, 512, 1024)

    # VP diffusion parameters
    b_min: float = 0.1
    b_max: float = 3.25

    # Training parameters
    batch_size: int = 512
    n_epochs: int = 1500
    learning_rate: float = 2e-4
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5

    # Sampling parameters
    num_steps: int = 200
    eps: float = 1e-4


@dataclass
class HFunctionConfig:
    """H-function training configuration"""
    device: str = "cuda"
    asset_dim: int = 4
    time_steps: int = 64
    embed_dim: int = 512

    # Training parameters
    train_batch_size: int = 2**13
    n_epochs: int = 400
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5

    # Event condition
    event_asset_idx: int = 3  # TSLA index
    event_window: int = 10    # last 10 days
    event_threshold: float = -3.0


@dataclass
class ConditionalGenConfig:
    """Conditional generation configuration"""
    device: str = "cuda"
    batch_size: int = 32
    num_steps: int = 200
    stoch: float = 0
    eta: float = 1.0
    use_q_model: bool = True

    # Q-model parameters (if used)
    q_model_epochs: int = 500
    q_model_lr: float = 1e-4


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
