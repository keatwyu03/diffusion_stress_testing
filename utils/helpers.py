"""
Helper utilities
"""
import random
import numpy as np
import pandas as pd
import torch


def block_interleaved_epoch_order(dates, device=None) -> torch.Tensor:
    months = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="M")
    block_ids = pd.factorize(months, sort=True)[0]  # (N,) block id per window, chronological
    n_blocks = block_ids.max() + 1

    # indices belonging to each block, independently shuffled this call
    block_idx = [np.random.permutation(np.where(block_ids == b)[0]) for b in range(n_blocks)]
    max_len = max(len(b) for b in block_idx)

    order = []
    for k in range(max_len):
        for b in block_idx:
            if k < len(b):
                order.append(b[k])

    return torch.tensor(order, dtype=torch.long, device=device)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
