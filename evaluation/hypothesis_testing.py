from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Energy distance
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pairwise_distance(A: np.ndarray, B: np.ndarray, chunk: int = 512) -> float:
    total = 0.0
    for start in range(0, len(A), chunk):
        block = A[start : start + chunk]
        # (chunk, 1, D) - (1, len(B), D) -> (chunk, len(B), D)
        total += np.linalg.norm(block[:, None, :] - B[None, :, :], axis=-1).sum()
    return total / (len(A) * len(B))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"expected 2-D samples, got {X.shape} and {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dimension mismatch: {X.shape[1]} vs {Y.shape[1]}")
    if len(X) < 2 or len(Y) < 2:
        raise ValueError(f"need >=2 rows per sample, got {len(X)} and {len(Y)}")

    cross = _mean_pairwise_distance(X, Y)
    within_x = _mean_pairwise_distance(X, X)
    within_y = _mean_pairwise_distance(Y, Y)
    return 2.0 * cross - within_x - within_y


def energy_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    return len(X) * len(Y) / (len(X) + len(Y)) * energy_distance(X, Y)


def _offdiag_mean(block: np.ndarray) -> float:

    m = len(block)
    if m < 2:
        return 0.0
    return float((block.sum() - np.trace(block)) / (m * (m - 1)))


class FixedReferenceEnergy:
    def __init__(self, X: np.ndarray, chunk: int = 512):
        self.X = np.asarray(X, dtype=np.float64)
        self.n = len(self.X)
        # Full (n, n) pairwise distances, built in row blocks to bound peak
        # memory. At n~5000 this is ~200MB in float64 and is the one big
        # allocation the test makes.
        self._d = np.empty((self.n, self.n), dtype=np.float64)
        for start in range(0, self.n, chunk):
            block = self.X[start : start + chunk]
            self._d[start : start + chunk] = np.linalg.norm(
                block[:, None, :] - self.X[None, :, :], axis=-1
            )
        self._within_x = float(self._d.mean())

    def __call__(self, idx: np.ndarray) -> float:
        """Scaled energy statistic between X and X[idx]."""
        idx = np.asarray(idx)
        m = len(idx)
        if m < 2:
            raise ValueError(f"need >=2 rows in the subset, got {m}")
        cross = float(self._d[:, idx].mean())
        within_y = float(self._d[np.ix_(idx, idx)].mean())
        energy = 2.0 * cross - self._within_x - within_y
        return self.n * m / (self.n + m) * energy


# ─────────────────────────────────────────────────────────────────────────────
# Episode structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeIndex:
    starts: np.ndarray   # (n_episodes,) first window index of each episode
    lengths: np.ndarray  # (n_episodes,) number of windows in each episode

    @property
    def n_episodes(self) -> int:
        return len(self.starts)

    @property
    def total_windows(self) -> int:
        return int(self.lengths.sum())

    def indices(self) -> np.ndarray:
        """All window indices covered, concatenated in chronological order."""
        if self.n_episodes == 0:
            return np.empty(0, dtype=int)
        return np.concatenate([
            np.arange(s, s + n) for s, n in zip(self.starts, self.lengths)
        ])

    @classmethod
    def from_mask(
        cls, mask: np.ndarray, dates: Sequence, max_gap_days: int = 5
    ) -> "EpisodeIndex":
        mask = np.asarray(mask, dtype=bool)
        dates = pd.DatetimeIndex(dates)
        if len(mask) != len(dates):
            raise ValueError(f"mask/dates length mismatch: {len(mask)} vs {len(dates)}")

        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return cls(np.empty(0, dtype=int), np.empty(0, dtype=int))

        # A new episode starts at the first event window, and wherever the
        # previous event window is not the immediately preceding TRADING day.
        # Adjacent window indices are consecutive trading days by construction,
        # so the date check is only a guard against genuine gaps in the rows.
        gap_days = (dates[idx[1:]] - dates[idx[:-1]]).days
        contiguous = (idx[1:] - idx[:-1] == 1) & (gap_days <= max_gap_days)
        breaks = np.r_[True, ~contiguous]

        starts = idx[breaks]
        # episode k runs until the next break (or the end of idx)
        break_positions = np.flatnonzero(breaks)
        ends = np.r_[break_positions[1:], len(idx)]
        lengths = ends - break_positions
        return cls(starts, lengths)


def episode_matched_null(
    episodes: EpisodeIndex,
    n_windows: int,
    forbidden: np.ndarray,
    rng: np.random.Generator,
    locality: int = 250,
    max_tries: int = 100,
) -> Optional[np.ndarray]:
    taken = forbidden.copy()
    chosen: List[np.ndarray] = []

    for start, length in zip(episodes.starts, episodes.lengths):
        low = max(0, start - locality)
        high = min(n_windows - length, start + locality)
        if high < low:
            return None

        for _ in range(max_tries):
            candidate_start = int(rng.integers(low, high + 1))
            span = np.arange(candidate_start, candidate_start + length)
            if not taken[span].any():
                taken[span] = True
                chosen.append(span)
                break
        else:
            return None

    return np.concatenate(chosen)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — real unconditional vs real conditional
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RandomizationResult:
    """Outcome of the episode-matched randomization test."""

    observed: float
    null_values: np.ndarray
    n_event_windows: int
    n_episodes: int
    n_reference: int
    n_failed: int
    locality: int

    @property
    def p_value(self) -> float:
        k = int((self.null_values >= self.observed).sum())
        return (k + 1) / (len(self.null_values) + 1)

    @property
    def null_mean(self) -> float:
        return float(self.null_values.mean())

    @property
    def effect_ratio(self) -> float:
        return self.observed / self.null_mean if self.null_mean else float("nan")

    def summary(self) -> str:
        return (
            f"Episode-matched randomization test\n"
            f"  event windows : {self.n_event_windows} in {self.n_episodes} episodes\n"
            f"  locality      : +/-{self.locality} windows\n"
            f"  reference sets: {self.n_reference}"
            + (f"  ({self.n_failed} failed to place)" if self.n_failed else "")
            + f"\n"
            f"  observed T    : {self.observed:.4f}\n"
            f"  null mean T   : {self.null_mean:.4f}  "
            f"[{np.percentile(self.null_values, 5):.4f}, "
            f"{np.percentile(self.null_values, 95):.4f}] (5-95%)\n"
            f"  ratio         : {self.effect_ratio:.3f}x\n"
            f"  p-value       : {self.p_value:.4f}\n"
        )


def episode_matched_test(
    windows: np.ndarray,
    mask: np.ndarray,
    dates: Sequence,
    n_reference: int = 5000,
    locality: int = 250,
    seed: int = 0,
    verbose: bool = True,
) -> RandomizationResult:
    windows = np.asarray(windows, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    episodes = EpisodeIndex.from_mask(mask, dates)
    if episodes.n_episodes == 0:
        raise ValueError("event mask selects no windows")


    stat = FixedReferenceEnergy(windows)
    observed = stat(np.flatnonzero(mask))

    rng = np.random.default_rng(seed)
    null_values: List[float] = []
    n_failed = 0

    iterator = range(n_reference)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Episode-matched null")

    for _ in iterator:
        idx = episode_matched_null(episodes, len(windows), mask, rng, locality)
        if idx is None:
            n_failed += 1
            continue
        null_values.append(stat(idx))

    if not null_values:
        raise RuntimeError(
            "no reference set could be placed — locality is too tight for the "
            "episode lengths, or the event set covers too much of the sample"
        )

    return RandomizationResult(
        observed=observed,
        null_values=np.array(null_values),
        n_event_windows=int(mask.sum()),
        n_episodes=episodes.n_episodes,
        n_reference=len(null_values),
        n_failed=n_failed,
        locality=locality,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — real conditional vs generated conditional
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EquivalenceResult:
    """Bootstrap CI on E(real events, generated), with a same-distribution reference."""

    point: float
    ci_low: float
    ci_high: float
    boot_values: np.ndarray
    reference: np.ndarray
    baseline: Optional[float]
    n_real: int
    n_gen: int
    alpha: float

    @property
    def reference_high(self) -> float:
        return float(np.percentile(self.reference, 95))

    @property
    def ci_shifted(self) -> bool:
        return not (self.ci_low <= self.point <= self.ci_high)

    @property
    def overlaps_reference(self) -> bool:
        return self.ci_low <= self.reference_high

    def summary(self) -> str:
        pct = int(round((1 - self.alpha) * 100))
        lines = [
            "Bootstrap CI on energy distance (real events vs generated)",
            f"  n real / n gen : {self.n_real} / {self.n_gen}",
            f"  E(real, gen)   : {self.point:.4f}",
            f"  {pct}% CI         : [{self.ci_low:.4f}, {self.ci_high:.4f}]",
            f"  same-dist ref  : median {np.median(self.reference):.4f}, "
            f"95th pct {self.reference_high:.4f}",
        ]
        if self.baseline is not None:
            lines.append(f"  uncond baseline: {self.baseline:.4f}")
            if self.point < self.baseline:
                lines.append(
                    f"  -> conditioning closes the gap "
                    f"({self.baseline:.4f} -> {self.point:.4f})"
                )
            else:
                lines.append(
                    f"  -> WARNING: conditioning does NOT beat the unconditional "
                    f"baseline ({self.baseline:.4f})"
                )
        lines.append(
            "  -> CI overlaps the same-distribution reference band"
            if self.overlaps_reference
            else "  -> CI sits ABOVE the same-distribution reference band"
        )
        if self.ci_shifted:
            lines.append(
                "     (CI sits above the point estimate — expected bootstrap "
                "duplication bias, conservative)"
            )
        return "\n".join(lines) + "\n"


def _split_half_reference(
    real: np.ndarray,
    episodes: Optional[EpisodeIndex],
    n_splits: int,
    rng: np.random.Generator,
    n_match: Optional[int] = None,
) -> np.ndarray:

    use_episodes = episodes is not None and episodes.n_episodes >= 4
    if use_episodes:
        bounds = np.r_[0, np.cumsum(episodes.lengths)]
        slices = [np.arange(bounds[k], bounds[k + 1]) for k in range(episodes.n_episodes)]

    values = []
    for _ in range(n_splits):
        if use_episodes:

            order = rng.permutation(episodes.n_episodes)
            cumulative = np.cumsum(episodes.lengths[order])
            cut = int(np.argmin(np.abs(cumulative - len(real) / 2))) + 1
            cut = min(max(cut, 1), episodes.n_episodes - 1)  # keep both sides non-empty
            left = np.concatenate([slices[k] for k in order[:cut]])
            right = np.concatenate([slices[k] for k in order[cut:]])
        else:
            perm = rng.permutation(len(real))
            left, right = perm[: len(real) // 2], perm[len(real) // 2 :]

        if n_match is not None:
            if len(left) < n_match or len(right) < n_match:
                continue
            left = rng.permutation(left)[:n_match]
            right = rng.permutation(right)[:n_match]

        if len(left) < 2 or len(right) < 2:
            continue
        values.append(energy_distance(real[left], real[right]))

    return np.array(values)


def bootstrap_energy_ci(
    real: np.ndarray,
    generated: np.ndarray,
    episodes: Optional[EpisodeIndex] = None,
    baseline_generated: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    n_splits: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> EquivalenceResult:
    real = np.asarray(real, dtype=np.float64)                     #Conditional real windows -> mask
    generated = np.asarray(generated, dtype=np.float64)           #Conditional generated windows

    rng = np.random.default_rng(seed)

    n_match = min(int(0.9 * (len(real) // 2)), len(generated))
    if n_match < 2:
        raise ValueError(
            f"too few rows to match sample sizes: real={len(real)}, "
            f"generated={len(generated)}"
        )
    generated = generated[rng.choice(len(generated), n_match, replace=False)]


    point = energy_distance(real, generated)
    baseline = (
        energy_distance(
            real,
            (lambda b: b[rng.choice(len(b), min(n_match, len(b)), replace=False)])(
                np.asarray(baseline_generated, dtype=np.float64)
            ),
        )
        if baseline_generated is not None
        else None
    )

    use_episodes = episodes is not None and episodes.n_episodes >= 2
    if use_episodes:
        bounds = np.r_[0, np.cumsum(episodes.lengths)]
        slices = [np.arange(bounds[k], bounds[k + 1]) for k in range(episodes.n_episodes)]

    d_rr = np.linalg.norm(real[:, None, :] - real[None, :, :], axis=-1)
    d_gg = np.linalg.norm(generated[:, None, :] - generated[None, :, :], axis=-1)
    d_rg = np.linalg.norm(real[:, None, :] - generated[None, :, :], axis=-1)

    boot = []
    iterator = range(n_boot)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Bootstrap CI")

    for _ in iterator:
        if use_episodes:
            pick = rng.integers(0, episodes.n_episodes, size=episodes.n_episodes)
            ridx = np.concatenate([slices[k] for k in pick])
            if len(ridx) < 2:
                ridx = np.arange(len(real))
        else:
            ridx = rng.integers(0, len(real), size=len(real))
        ridx = rng.permutation(ridx)[:n_match]
        gidx = rng.integers(0, n_match, size=n_match)

        boot.append(
            2.0 * d_rg[np.ix_(ridx, gidx)].mean()
            - _offdiag_mean(d_rr[np.ix_(ridx, ridx)])
            - _offdiag_mean(d_gg[np.ix_(gidx, gidx)])
        )

    boot = np.array(boot)
    reference = _split_half_reference(real, episodes, n_splits, rng, n_match=n_match)

    return EquivalenceResult(
        point=point,
        ci_low=float(np.percentile(boot, 100 * alpha / 2)),
        ci_high=float(np.percentile(boot, 100 * (1 - alpha / 2))),
        boot_values=boot,
        reference=reference,
        baseline=baseline,
        n_real=n_match,
        n_gen=n_match,
        alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — mirrors main.py's event-mask construction exactly
# ─────────────────────────────────────────────────────────────────────────────

def event_mask_from_z(
    Z_start: torch.Tensor,
    Z_end: torch.Tensor,
    event_type: str,
    threshold: float,
) -> torch.Tensor:
    """Hard event condition, matching main.py / cov.py / conditional_gen.py.

    Evaluation always uses the HARD definition regardless of
    cfg.constraint_mode — "soft" only grades the h-function's training labels,
    it does not change what counts as an event.
    """
    if event_type == "abs_change":
        return (Z_end - Z_start).abs() >= threshold
    if event_type == "absval":
        return Z_end.abs() >= threshold
    if event_type == "upper_change":
        return (Z_end - Z_start) >= threshold
    if event_type == "lower_change":
        return (Z_end - Z_start) <= -threshold
    if event_type == "start_upper":
        return Z_start >= threshold
    raise NotImplementedError(f"event_type={event_type!r} not supported")


def load_split(split: str = "train"):
    """Build (windows, mask, dates) for one split, plus the config in use.

    windows: (N, A) last-day return vector per window — the object every test
             here operates on. Real windows are stored (N, T, A) so the last day
             is [:, -1, :]; generated samples are stored (N, A, T) so it is
             [:, :, -1]. Mixing those up silently compares a time slice against
             an asset slice, so both conversions live here.
    """
    from config import get_default_config
    from data import DataProcessor

    config = get_default_config()
    dp = DataProcessor(
        csv_path=config.data.csv_path,
        tickers=config.data.tickers,
        weekday_col=config.data.weekday_col,
        seq_len=config.data.seq_len,
        test_days=config.data.test_days,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        train_end_date=config.data.train_end_date,
        window_shift=config.data.window_shift,
        winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper,
        ema_span=config.data.ema_span,
        event_causal=config.data.event_causal,
        event_lag_gap=config.data.event_lag_gap,
    )
    dp.process_all()

    event_type = config.hfunction.event_type
    threshold = dp.get_event_threshold_from_percentile(
        config.hfunction.event_threshold, event_type
    )
    print(
        f"Event threshold: top {config.hfunction.event_threshold:.1%} -> "
        f"{threshold:.4f} std ({event_type})"
    )

    if split == "train":
        X = dp.X_train
        Z_start, Z_end, valid_idx = dp.get_z_windows_train_aligned()
        dates = dp.y_dates_train
    else:
        X = dp.X_test
        Z_start, Z_end, valid_idx = dp.get_z_windows_test()
        dates = dp.y_dates_test

    valid_event = event_mask_from_z(Z_start, Z_end, event_type, threshold)
    mask = torch.zeros(X.shape[0], dtype=torch.bool)
    mask[valid_idx] = valid_event

    windows = X[:, -1, :].numpy()
    return windows, mask.numpy(), dates, config


def load_generated(split: str = "train") -> Optional[np.ndarray]:
    """Last-day returns of the generated conditional samples, or None."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, f"generated_samples_{split}.pt")
    if not os.path.exists(path):
        return None
    gen = torch.load(path, map_location="cpu")
    return gen[:, :, -1].numpy()   # (N, A, T) -> (N, A)


# ─────────────────────────────────────────────────────────────────────────────

def main(split: str = "train", n_reference: int = 5000, n_boot: int = 2000):
    windows, mask, dates, config = load_split(split)
    print(f"\n{split}: {mask.sum()} event windows / {len(mask)} total")

    episodes = EpisodeIndex.from_mask(mask, dates)
    print(
        f"Episodes: {episodes.n_episodes} "
        f"(lengths {episodes.lengths.min()}-{episodes.lengths.max()}, "
        f"median {int(np.median(episodes.lengths))})\n"
    )

    print("=" * 70)
    print("TEST 1 — real unconditional vs real conditional")
    print("=" * 70)
    result1 = episode_matched_test(
        windows, mask, dates, n_reference=n_reference, seed=config.seed
    )
    print(result1.summary())

    generated = load_generated(split)
    if generated is None:
        print(f"No generated_samples_{split}.pt — skipping test 2.")
        return result1, None

    print("=" * 70)
    print("TEST 2 — real conditional vs generated conditional")
    print("=" * 70)
    result2 = bootstrap_energy_ci(
        windows[mask], generated, episodes=episodes,
        n_boot=n_boot, seed=config.seed,
    )
    print(result2.summary())
    return result1, result2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-reference", type=int, default=5000)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    main(args.split, args.n_reference, args.n_boot)
