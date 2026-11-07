"""
Conditional-only (non-rare-event) runs condition the diffusion model on the
full causal window of an exogenous variable: the seq_len raw values ending
event_lag_gap + 1 rows before the return window starts (same window as
data_processor.py's causal _scan_macro_windows, e.g. Oct 18..Nov 1 for a
return window starting Nov 5 with gap 1).

The score network (models/transformer_score.py) is NOT edited. Its forward()
calls self.cond_embed(condition.reshape(-1, 1)); a contiguous (B, W) condition
flattens row-major, so WindowCondEmbed views it back as (B, W) and maps the
whole window to one cond_dim embedding. The swap must happen before training
(the optimizer is built inside DiffusionModel.train) and before loading a
checkpoint (so state_dict keys/shapes match).
"""
import os

import numpy as np
import torch.nn as nn

_BASELINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_BASELINE)

# One place that ties each experiment's INPUTS to its OUTPUT folders, so a run
# can't read CPI data and write into the Full folder (or vice versa).
EXPERIMENTS = {
    "full": {"dir": "experiment_1 (Full)", "macro_csv": "explore/macro_data_new.csv",
             "states_dir": "experiment_1 (Full)/extras/exp2_results"},
    "cpi": {"dir": "experiment_2 (CPI Only)", "macro_csv": "explore/macro_data_new_cpi.csv",
            "states_dir": "experiment_2 (CPI Only)/exp2_results_cpi_only"},
}


def experiment_paths(name: str) -> dict:
    """Inputs and conditional-only result folders for one experiment."""
    e = EXPERIMENTS[name]
    causal_dir = os.path.join(_BASELINE, e["dir"], "causal")
    return {
        "macro_csv": os.path.join(_REPO, e["macro_csv"]),
        "states_dir": os.path.join(_BASELINE, e["states_dir"]),
        "macro_inclusion": os.path.join(causal_dir, "macro_inclusion", "conditional_only"),
        "macro_embed": os.path.join(causal_dir, "macro_embed", "conditional_only"),
    }


class WindowCondEmbed(nn.Module):
    def __init__(self, window: int, cond_dim: int):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(nn.Linear(window, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, c):
        return self.net(c.reshape(-1, self.window))


def use_window_conditioning(dm, window: int, cond_dim: int):
    dm.model.cond_embed = WindowCondEmbed(window, cond_dim).to(dm.device)
    return dm


def eval_window_idx(n_train: int, n_eval: int) -> np.ndarray:
    """In-sample evaluation set: n_eval train windows evenly spaced over the
    whole train period (0 or >= n_train = every train window). Same rule in
    main_exp1.py and diffusion_adapters.py, so every model in a group is
    generated and scored on identical windows."""
    if n_eval <= 0 or n_eval >= n_train:
        return np.arange(n_train)
    return np.unique(np.linspace(0, n_train - 1, n_eval).round().astype(np.int64))


def causal_value_windows(values: np.ndarray, starts: np.ndarray, seq_len: int, event_lag_gap: int) -> np.ndarray:
    """(N, seq_len) raw values for each return window starting at row `starts[n]`:
    rows end-seq_len+1 .. end, with end = start - event_lag_gap - 1. NaN where
    any row falls before the series starts."""
    end = np.asarray(starts) - event_lag_gap - 1
    rows = end[:, None] + np.arange(-seq_len + 1, 1)[None, :]
    out = np.full(rows.shape, np.nan)
    ok = (rows >= 0) & (rows < len(values))
    out[ok] = np.asarray(values, dtype=np.float64)[rows[ok]]
    return out
