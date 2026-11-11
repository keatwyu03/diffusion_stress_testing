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

import sys

_BASELINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_BASELINE)
sys.path.insert(0, _REPO)
from config import CONDITIONING_SETS, get_default_config  # noqa: E402

# Every M0-M4 / S0-S4 script takes its inputs AND output folders from config.data.conditioning_set
# (one place: config.CONDITIONING_SETS), so a run can never read one set's data and write into another's
# experiment folder. --experiment, where a script has it, may only repeat the config's set.
EXPERIMENTS = list(CONDITIONING_SETS)


def experiment_paths(name: str = None) -> dict:
    """Inputs and result folders of the conditioning set selected in config."""
    d = get_default_config().data
    if name is not None and name != d.conditioning_set:
        raise SystemExit(f"--experiment {name!r} != config.data.conditioning_set {d.conditioning_set!r}: "
                         f"set the conditioning set in config/config.py instead")
    exp = d.experiment_dir
    causal_dir = os.path.join(exp, "causal")
    return {
        "conditioning_set": d.conditioning_set,
        "experiment_dir": exp,
        "macro_csv": d.macro_csv_path,                                   # m_t + raw daily log returns
        "m_t_csv": os.path.join(os.path.dirname(d.macro_csv_path), "m_t.csv"),
        "states_dir": os.path.join(exp, "extras", "exp2_results"),       # S0-S4 <method>/daily_state.csv
        "macro_inclusion": os.path.join(causal_dir, "macro_inclusion", "conditional_only"),
        "macro_embed": os.path.join(causal_dir, "macro_embed", "conditional_only"),
        # per-method NN standardizers / residuals (standardize_own_macro.py)
        "standardization_dir": os.path.join(exp, "extras", "standardization_own"),
        "rare_inclusion": os.path.join(causal_dir, "macro_inclusion", "rare_event"),
        "rare_embed": os.path.join(causal_dir, "macro_embed", "rare_event"),
        "embed_channels_dir": os.path.join(exp, "extras", "rare_event_channels"),   # S0-S4 own-std channel files
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
