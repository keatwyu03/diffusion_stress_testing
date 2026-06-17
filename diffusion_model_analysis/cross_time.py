import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel, ConditionalGenerator

config = get_default_config()

start_date = "1995-01-01"
end_date = "2005-01-01"


data_processor = DataProcessor(
    csv_path=config.data.csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=config.data.test_days,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)

data_processor.process_all()

tickers = config.data.tickers
n_assets = len(tickers)


