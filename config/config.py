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
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data configuration"""
    # Where to load the data. Training/Testing split 
    # Outlier handling -> winsorize upper and lower = % bounds to remove
    csv_path: str = "explore/macro_data_new.csv"
    tickers: List[str] = None
    weekday_col: str = "weekday"
    seq_len: int = 64
    test_days: int = 3000
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["unemp", "unemp_flag", "sp500", "baa", "baa_flag"]


@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    #In and out channels are used to train the neural network for score function
    #This is the part that controls the denosing method for training s_theta
    #
    device: str = "cuda"
    in_channels: int = 5
    out_channels: int = 5
    sample_size: int = 64

    #number of layers for each CNN and number of parameters in each of those CNN layers
    layers_per_block: int = 3
    block_out_channels: Tuple[int, int, int] = (64, 128, 256)

    # Variance Preserving diffusion parameters
    b_min: float = 0.1
    b_max: float = 3.25

    # Training parameters
    batch_size: int = 32               #Stochastic minibatch gradient descent
    n_epochs: int = 100                #Number of times to loop through the data
    learning_rate: float = 2e-4        #Alpha Stepsize
    scheduler_patience: int = 50       #Check convergence every X number of loops through the data
    scheduler_factor: float = 0.5      #Multiplier for the Learning rate when plateau

    # Sampling parameters
    num_steps: int = 200               #Number of noisy elements to add
    eps: float = 1e-4                  #Stopping point of when we claim the data are now normal


@dataclass
class HFunctionConfig:
    """H-function training configuration"""
    #Doobs H modification for the terminal constraints
    #Using a Neural Network to learn this

    device: str = "cuda"
    asset_dim: int = 5
    time_steps: int = 64
    embed_dim: int = 128

    # Training parameters
    train_batch_size: int = 256      #Nuber of noisy trajectories to pass through at each stage for unconditional diffusion=
    n_epochs: int = 500              #Numnber of times to go through the data
    learning_rate: float = 1e-4      #Step size (adapted) for the SGD
    weight_decay: float = 1e-4       #Add penalties to prevent overfitting
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5

    # Event condition

    # Type of event to look 
    event_type: str = "absval" #Choose between "sum", "change", or "absval"

    # Column to watch for the shock
    event_asset_idx: int = 0
    
    # Lookback period
    event_window: int = 3

    # Threshold    
    event_threshold: float = 1.2

    # Constraint Mode
    constraint_mode: str = "hard"  #Choose between "hard" or "soft" (exponential)
    reward_sharpness: float = 50.0 #Multiplier for significant sigmoid


@dataclass
class ConditionalGenConfig:
    """Conditional generation configuration"""
    #Learning the q function for conditional generation
    #Neural Network to get us grad log h = q / h

    device: str = "cuda"
    batch_size: int = 32
    num_steps: int = 200
    stoch: float = 0.3
    eta: float = 1.0
    use_q_model: bool = True

    # Q-model parameters (if used)
    q_model_epochs: int = 500
    q_model_lr: float = 1e-4

    # Constraint Mode
    constraint_mode: str = "soft"  #Choose between "hard" and "soft"
    beta: float = 1.0              #Parameter for trade-off denominator



@dataclass
class PortfolioConfig:
    """Portfolio analysis configuration"""
    window_for_cov: int = 54  # Use first 54 days to compute covariance
    last_days_sum: int = 5    # Compute sum of last 5 days for portfolio
    portfolio_tickers: List[str] = None

    def __post_init__(self):
        if self.portfolio_tickers is None:
            self.portfolio_tickers = ["sp500", "baa"]


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    enabled: bool = False
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
