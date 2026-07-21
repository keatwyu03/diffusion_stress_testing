# """
# Data processing module for financial time series
# """
# import numpy as np
# import pandas as pd
# import torch
# from typing import Tuple, List, Optional


# class DataProcessor:
#     """Process financial time series data"""

#     def __init__(
#         self,
#         csv_path: str,
#         tickers: List[str],
#         weekday_col: str = "weekday",
#         seq_len: int = 64,
#         test_days: int = 700,
#         winsorize_lower: float = 0.005,
#         winsorize_upper: float = 0.995,
#     ):
#         self.csv_path = csv_path
#         self.tickers = tickers
#         self.weekday_col = weekday_col
#         self.seq_len = seq_len
#         self.test_days = test_days
#         self.winsorize_lower = winsorize_lower
#         self.winsorize_upper = winsorize_upper

#         # Data containers
#         self.df = None
#         self.r_dw = None
#         self.weekday_mean = None
#         self.sigma_seq = None
#         self.df_z = None
#         self.df_z_wins = None

#         # Sequence data
#         self.X = None
#         self.y = None
#         self.y_dates = None

#         # Train/test splits
#         self.X_train = None
#         self.y_train = None
#         self.y_dates_train = None
#         self.X_test = None
#         self.y_test = None
#         self.y_dates_test = None

        

#     def load_returns(self) -> pd.DataFrame:
#         """Load returns from CSV"""
#         df = pd.read_csv(self.csv_path)
#         if "Date" in df.columns:
#             df["Date"] = pd.to_datetime(df["Date"])
#             df = df.sort_values("Date").set_index("Date")
#         df = df[self.tickers].dropna()
#         self.df = df
#         return df

#     def remove_weekday_effect(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """Remove weekday effect from returns"""
#         # weekday_mean = self.df.groupby(self.weekday_col)[self.tickers].mean()
#         # aligned = self.df[self.weekday_col].map(weekday_mean.to_dict("index"))
#         # aligned_df = pd.DataFrame(list(aligned), index=self.df.index)[self.tickers]
#         weekday_series = pd.Series(self.df.index.dayofweek, index=self.df.index)
#         weekday_mean = self.df.groupby(weekday_series)[self.tickers].mean()
#         aligned_df = weekday_series.map(weekday_mean.to_dict("index"))
#         aligned_df = pd.DataFrame(list(aligned_df), index=self.df.index)[self.tickers]
#         r_dw = self.df[self.tickers] - aligned_df

#         self.r_dw = r_dw
#         self.weekday_mean = weekday_mean
#         return r_dw, weekday_mean

#     def standardize(self) -> pd.DataFrame:
#         """Standardize using train-set std only to avoid data leakage"""
#         data = self.r_dw if self.r_dw is not None else self.df
#         train_data = data.iloc[:-self.test_days]
#         self.mu_seq = train_data.mean()
#         self.sigma_seq = train_data.std()
#         z = (data - self.mu_seq) / self.sigma_seq
#         self.df_z = z.dropna(how="any")
#         return self.df_z

#     def winsorize(self) -> pd.DataFrame:
#         """Winsorize the standardized returns"""
#         df_wins = self.df_z.copy()
#         for col in df_wins.columns:
#             q_low = df_wins[col].quantile(self.winsorize_lower)
#             q_high = df_wins[col].quantile(self.winsorize_upper)
#             df_wins[col] = df_wins[col].clip(lower=q_low, upper=q_high)
#         self.df_z_wins = df_wins
#         return df_wins

#     def make_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Create sequences from standardized returns"""
#         z_values = self.df_z.values
#         dates = self.df_z.index.to_numpy()
#         T, D = z_values.shape
#         X_list, y_list, idx_list = [], [], []

#         for t in range(self.seq_len, T):
#             X_list.append(z_values[t - self.seq_len : t, :])
#             y_list.append(z_values[t, :])
#             idx_list.append(dates[t])

#         X = np.stack(X_list, axis=0)
#         y = np.stack(y_list, axis=0)
#         idx = np.array(idx_list)

#         self.X = X
#         self.y = y
#         self.y_dates = idx
#         return X, y, idx

#     def train_test_split(self) -> None:
#         """Split data into training and test sets"""
#         if self.X is None:
#             raise ValueError("Must call make_sequences() first")

#         # NumPy splits
#         X_train_np = self.X[: -self.test_days]
#         y_train_np = self.y[: -self.test_days]
#         y_dates_train = self.y_dates[: -self.test_days]

#         X_test_np = self.X[-self.test_days :]
#         y_test_np = self.y[-self.test_days :]
#         y_dates_test = self.y_dates[-self.test_days :]

#         # Convert to torch tensors
#         self.X = torch.tensor(self.X, dtype=torch.float32)
#         self.y = torch.tensor(self.y, dtype=torch.float32)
#         self.X_train = torch.tensor(X_train_np, dtype=torch.float32)
#         self.y_train = torch.tensor(y_train_np, dtype=torch.float32)
#         self.X_test = torch.tensor(X_test_np, dtype=torch.float32)
#         self.y_test = torch.tensor(y_test_np, dtype=torch.float32)
#         self.y_dates_train = y_dates_train
#         self.y_dates_test = y_dates_test

#     def process_all(self) -> None:
#         """Run all data processing steps"""
#         print("Loading returns...")
#         self.load_returns()
#         print(f"Loaded data shape: {self.df.shape}")

#         # print("Removing weekday effect...")
#         # self.remove_weekday_effect()
#         self.r_dw = self.df[self.tickers]  # skip weekday removal, use raw data

#         print("Standardizing...")
#         self.standardize()
#         print(f"Standardized data shape: {self.df_z.shape}")

#         print("Winsorizing...")
#         self.winsorize()

#         print("Creating sequences...")
#         self.make_sequences()
#         print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

#         print(f"Splitting into train/test (test_days={self.test_days})...")
#         self.train_test_split()
#         print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

#     def get_diffusion_data(self) -> torch.Tensor:
#         """Get data for diffusion model training (winsorized windows)"""
#         data = torch.tensor(self.df_z_wins.values, dtype=torch.float32)
#         window_size = self.seq_len
#         X_all = []
#         for i in range(len(data) - window_size + 1):
#             X_all.append(data[i : i + window_size].T)
#         X_all = torch.stack(X_all)

#         # Return only training portion
#         X_diffusion_train = X_all[: -self.test_days]
#         return X_diffusion_train

#     def invert_samples(
#         self,
#         samples: torch.Tensor,
#         start_weekday: Optional[int] = None,
#         monthly = False
#     ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
#         """
#         Invert standardized samples back to returns

#         Args:
#             samples: (T, D) tensor of standardized samples
#             start_weekday: Starting weekday (0-4), if None will be random

#         Returns:
#             r_seq: Returns dataframe
#             r_dw_seq: De-weekday returns dataframe
#             sigma_path: Sigma path
#             port_seq: Portfolio returns series
#         """
#         z_seq = (
#             samples.detach().cpu().numpy()
#             if hasattr(samples, "detach")
#             else np.asarray(samples)
#         )

#         if monthly:
#             r_seq = z_seq * self.sigma_seq.to_numpy() + self.mu_seq.to_numpy()
#             return pd.DataFrame(r_seq, columns = self.tickers), None, None, None

#         T, D = z_seq.shape

#         if start_weekday is None:
#             start_weekday = np.random.randint(0, 5)

#         r_seq = np.zeros((T, D))
#         r_dw_seq = np.zeros((T, D))
#         sigma_path = np.tile(self.sigma_seq.to_numpy(), (T + 1, 1))
#         port_seq = np.zeros(T)

#         for t in range(T):
#             r_dw_t = z_seq[t] * self.sigma_seq.to_numpy() + self.mu_seq.to_numpy()
#             w_t = (start_weekday + t) % 5
#             r_t = r_dw_t + self.weekday_mean.loc[w_t, self.tickers].to_numpy()

#             r_dw_seq[t] = r_dw_t
#             r_seq[t] = r_t
#             port_seq[t] = r_t.mean()

#         return (
#             pd.DataFrame(r_seq, columns=self.tickers),
#             pd.DataFrame(r_dw_seq, columns=self.tickers),
#             sigma_path,
#             pd.Series(port_seq, name="Portfolio"),
#         )


"""
Data processing module for financial time series
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataProcessor:
    """Process financial time series data"""

    def __init__(
        self,
        csv_path: str,
        tickers: List[str],
        weekday_col: str = "weekday",
        seq_len: int = 64,
        test_days: int = 700,
        start_date: str = None,
        end_date: str = None,
        train_end_date: str = None,
        window_shift: int = 1,
        winsorize_lower: float = 0.005,
        winsorize_upper: float = 0.995,
        ema_span: int = 60,
        ema_min_periods: int = 20,
    ):
        self.csv_path = csv_path
        self.tickers = tickers
        self.weekday_col = weekday_col
        self.seq_len = seq_len
        self.test_days = test_days
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = train_end_date
        self.window_shift = window_shift
        self.winsorize_lower = winsorize_lower
        self.winsorize_upper = winsorize_upper
        self.ema_span = ema_span
        self.ema_min_periods = ema_min_periods

        # Data containers
        self.df = None
        self.r_dw = None
        self.weekday_mean = None
        self.sigma_seq = None
        self.df_z = None

        # Per-window EMA standardization (causal, entry-day) — see _compute_ema_stats()
        self.MU = None            # (T, A) trailing EMA mean per row (shifted 1 day)
        self.SG = None            # (T, A) trailing EMA vol per row (shifted 1 day)
        self.mu_entry = None      # (Nw, A) entry-day EMA mean per window (invertibility)
        self.sig_entry = None     # (Nw, A) entry-day EMA vol per window

        # Sequence data
        self.X = None
        self.y = None
        self.y_dates = None

        # Train/test splits
        self.X_train = None
        self.y_train = None
        self.y_dates_train = None
        self.X_test = None
        self.y_test = None
        self.y_dates_test = None

    def load_returns(self) -> pd.DataFrame:
        """Load returns from CSV, optionally filtered to [start_date, end of data]"""
        df = pd.read_csv(self.csv_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
        df = df[self.tickers]
        stock_cols = self.tickers[1:]  # tickers[0] is the sparse macro series
        df = df.dropna(subset=stock_cols)
        if self.start_date is not None:
            df = df[df.index >= pd.to_datetime(self.start_date)]
        if self.end_date is not None:
            df = df[df.index <= pd.to_datetime(self.end_date)]
        self.df = df
        return df

    def remove_weekday_effect(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove weekday effect from returns"""
        weekday_series = pd.Series(self.df.index.dayofweek, index=self.df.index)
        weekday_mean = self.df.groupby(weekday_series)[self.tickers].mean()
        aligned = weekday_series.map(weekday_mean.to_dict("index"))
        aligned_df = pd.DataFrame(list(aligned), index=self.df.index)[self.tickers]
        r_dw = self.df[self.tickers] - aligned_df

        self.r_dw = r_dw
        self.weekday_mean = weekday_mean
        return r_dw, weekday_mean

    def _winsorize_raw_returns(self) -> None:
        """Clip each stock's raw log-return to its [winsorize_lower,
        winsorize_upper] quantile range, computed from TRAIN rows only (no
        leakage) and applied to the full self.df (train+test) in place. Runs
        BEFORE _compute_ema_stats(), so the EMA mean/vol themselves are
        computed on the clipped series — outliers are tamed before anything
        downstream (including the trailing vol estimate) sees them."""
        stock_cols = self.tickers[1:]
        r = self.df[stock_cols]
        if self.train_end_date is not None:
            cutoff = pd.to_datetime(self.train_end_date)
            train_r = r[r.index <= cutoff]
        else:
            train_r = r.iloc[:-self.test_days]
        q_low = train_r.quantile(self.winsorize_lower)
        q_high = train_r.quantile(self.winsorize_upper)
        self.df[stock_cols] = r.clip(lower=q_low, upper=q_high, axis=1)
        print(f"Winsorized raw returns to train-only [{self.winsorize_lower:.1%}, "
              f"{self.winsorize_upper:.1%}] quantiles per stock")

    def _compute_ema_stats(self) -> None:
        """Causal per-stock EMA mean/vol for per-window standardization
        (replicates rare_event_CDG): stats at row t use data through t-1 only
        (shift(1)), so the entry-day standardizer never sees the window itself.
        Rows before the EMA warm-up are trimmed from self.df, so windows and
        the macro series keep sharing one row base."""
        r = self.df[self.tickers[1:]]
        MU = r.ewm(span=self.ema_span, min_periods=self.ema_min_periods).mean().shift(1)
        SG = r.ewm(span=self.ema_span, min_periods=self.ema_min_periods).std().shift(1)
        valid = MU.notna().all(axis=1) & SG.notna().all(axis=1) & (SG > 0).all(axis=1)
        first = int(valid.to_numpy().argmax())
        self.df = self.df.iloc[first:]
        self.MU = MU.iloc[first:].to_numpy(np.float64)
        self.SG = SG.iloc[first:].to_numpy(np.float64)
        print(f"EMA standardizer: span={self.ema_span}, warm-up cut {first} rows, {len(self.df)} remain")

    def standardize(self) -> pd.DataFrame:
        """Per-row EMA z-scores, kept for diagnostics (diagnosis.py). The actual
        window tensors are standardized with ENTRY-day stats in make_sequences()
        / get_diffusion_data(), not with this frame."""
        r = self.df[self.tickers[1:]]
        self.df_z = pd.DataFrame(
            (r.to_numpy(np.float64) - self.MU) / self.SG,
            index=self.df.index, columns=r.columns,
        )
        return self.df_z

    def make_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences with PER-WINDOW EMA standardization: each window's
        returns are standardized by the EMA mean/vol at its ENTRY day (causal,
        one vector per stock), z = (r - mu_entry) / sig_entry, broadcast over
        the window's days. Invertible via (mu_entry, sig_entry), stored per
        window."""
        r_values = self.df[self.tickers[1:]].to_numpy(np.float64)
        dates = self.df.index.to_numpy()
        weekday_arr = self.df.index.dayofweek.values
        T, D = r_values.shape
        X_list, y_list, idx_list, sw_list, mu_list, sig_list = [], [], [], [], [], []

        for t in range(self.seq_len, T, self.window_shift):
            s = t - self.seq_len                      # window entry row
            X_list.append((r_values[s:t, :] - self.MU[s]) / self.SG[s])
            y_list.append((r_values[t, :] - self.MU[t]) / self.SG[t])
            idx_list.append(dates[t])
            sw_list.append(int(weekday_arr[s]))       # weekday of window start
            mu_list.append(self.MU[s])
            sig_list.append(self.SG[s])

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        idx = np.array(idx_list)

        self.X = X
        self.y = y
        self.y_dates = idx
        self.start_weekdays = np.array(sw_list, dtype=np.int8)
        self.mu_entry = np.stack(mu_list).astype(np.float32)    # (Nw, A)
        self.sig_entry = np.stack(sig_list).astype(np.float32)  # (Nw, A)
        return X, y, idx

    def train_test_split(self) -> None:
        """Split data into training and test sets"""
        if self.X is None:
            raise ValueError("Must call make_sequences() first")

        if self.train_end_date is not None:
            # Split at the specified date boundary
            cutoff = pd.to_datetime(self.train_end_date)
            # y_dates[i] is the date at position t (one day after the window ends)
            # so sequences where y_dates <= cutoff belong to train
            train_mask = self.y_dates <= cutoff
            split_idx = int(train_mask.sum())
        else:
            split_idx = len(self.X) - self.test_days

        X_train_np = self.X[:split_idx]
        y_train_np = self.y[:split_idx]
        y_dates_train = self.y_dates[:split_idx]

        X_test_np = self.X[split_idx:]
        y_test_np = self.y[split_idx:]
        y_dates_test = self.y_dates[split_idx:]

        self.start_weekdays_train = self.start_weekdays[:split_idx]
        self.start_weekdays_test  = self.start_weekdays[split_idx:]

        # entry-day EMA stats per window, for de-standardizing back to raw returns
        self.mu_entry_train,  self.mu_entry_test  = self.mu_entry[:split_idx],  self.mu_entry[split_idx:]
        self.sig_entry_train, self.sig_entry_test = self.sig_entry[:split_idx], self.sig_entry[split_idx:]

        # Convert to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.X_train = torch.tensor(X_train_np, dtype=torch.float32)
        self.y_train = torch.tensor(y_train_np, dtype=torch.float32)
        self.X_test = torch.tensor(X_test_np, dtype=torch.float32)
        self.y_test = torch.tensor(y_test_np, dtype=torch.float32)
        self.y_dates_train = y_dates_train
        self.y_dates_test = y_dates_test

    def process_all(self) -> None:
        """Run all data processing steps"""
        print("Loading returns...")
        self.load_returns()
        print(f"Loaded data shape: {self.df.shape}")

        print("Winsorizing raw returns (train-only quantiles)...")
        self._winsorize_raw_returns()

        print("Computing causal EMA standardizer...")
        self._compute_ema_stats()
        self.r_dw = self.df[self.tickers[1:]]  # raw stock returns (winsorized, post warm-up trim)

        print("Standardizing (per-row EMA z, diagnostics only)...")
        self.standardize()
        print(f"Standardized data shape: {self.df_z.shape}")

        print("Creating sequences (per-window entry-day EMA standardization)...")
        self.make_sequences()
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

        split_info = f"train_end_date={self.train_end_date}" if self.train_end_date else f"test_days={self.test_days}"
        print(f"Splitting into train/test ({split_info})...")
        self.train_test_split()
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def get_diffusion_data(self) -> torch.Tensor:
        """Get data for diffusion model training: per-window EMA-standardized
        windows (entry-day stats), same standardization as make_sequences().
        Also stores self.diffusion_end_dates_train, 1:1 aligned with the
        returned tensor's window order (window j ends on this date) — used by
        block sampling to group windows into calendar-month blocks."""
        r_values = self.df[self.tickers[1:]].to_numpy(np.float64)
        window_size = self.seq_len
        X_all = []
        for i in range(0, len(r_values) - window_size + 1, self.window_shift):
            z = (r_values[i : i + window_size, :] - self.MU[i]) / self.SG[i]
            X_all.append(torch.tensor(z.T, dtype=torch.float32))
        X_all = torch.stack(X_all)

        # Window j (start = j*window_shift) ends at dates_all[j*window_shift + window_size - 1];
        # step the slice by window_shift so end_dates lines up 1:1 with X_all's windows.
        dates_all = self.df.index
        end_dates = dates_all[window_size - 1 :: self.window_shift]

        # Return only training portion, consistent with train_test_split boundary
        if self.train_end_date is not None:
            cutoff = pd.to_datetime(self.train_end_date)
            n_train = int((end_dates <= cutoff).sum())
            X_diffusion_train = X_all[:n_train]
        else:
            n_train = len(X_all) - self.test_days
            X_diffusion_train = X_all[:n_train]

        self.diffusion_end_dates_train = end_dates[:n_train]
        return X_diffusion_train

    def destandardize_windows(self, z: torch.Tensor, mu_entry: np.ndarray, sig_entry: np.ndarray) -> np.ndarray:
        """Invert per-window EMA standardization back to raw log-returns.
        z: (N, A, T) standardized windows; mu_entry/sig_entry: (N, A) entry stats.
        For GENERATED samples pass entry stats drawn from the comparison set
        (see sample_entry_stats) — that reinjects the vol regime."""
        z = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
        return z * np.asarray(sig_entry)[:, :, None] + np.asarray(mu_entry)[:, :, None]

    def sample_entry_stats(
        self, n_draw: int, split: str = "train", mask: Optional[np.ndarray] = None, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw n_draw (mu_entry, sig_entry) vectors with replacement from a
        split's entry-stat pool, to give generated windows realistic vol scales.

        mask: optional boolean array over the split's windows restricting the
        pool. For CONDITIONAL samples pass the event mask — events cluster in
        particular vol regimes, and drawing from the full pool would erase the
        empirical (macro event <-> entry volatility) relationship when
        reconstructing raw returns. Unconditional samples use the full pool.
        """
        mu_pool = self.mu_entry_train if split == "train" else self.mu_entry_test
        sig_pool = self.sig_entry_train if split == "train" else self.sig_entry_test
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            mu_pool, sig_pool = mu_pool[mask], sig_pool[mask]
        if len(mu_pool) == 0:
            raise ValueError("entry-stat pool is empty (empty mask or split)")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(mu_pool), size=n_draw)
        return mu_pool[idx], sig_pool[idx]

    def _macro_std_values_and_n_train(self) -> Tuple[np.ndarray, int]:
        """Standardize the raw macro column (train-set stats only) and return
        the full standardized array plus the row count belonging to train."""
        macro_col = self.tickers[0]
        macro_raw = self.df[macro_col]

        if self.train_end_date is not None:
            cutoff     = pd.to_datetime(self.train_end_date)
            train_vals = macro_raw[macro_raw.index <= cutoff].dropna()
        else:
            train_vals = macro_raw.iloc[:-self.test_days].dropna()

        z_mean = train_vals.mean()
        z_std  = train_vals.std()
        macro_values = ((macro_raw - z_mean) / z_std).values

        if self.train_end_date is not None:
            cutoff    = pd.to_datetime(self.train_end_date)
            end_dates = self.df.index[self.seq_len - 1:]
            n_train   = int((end_dates <= cutoff).sum())
        else:
            n_train = len(macro_values) - self.test_days

        return macro_values, n_train

    def _scan_macro_windows(
        self, macro_values: np.ndarray, start_i: int, end_i: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scan window indices [start_i, end_i) — window j starts at raw row
        j*self.window_shift — keeping windows with a macro observation at both
        window endpoints.

        valid_idx is 0-indexed relative to this range (i.e. directly indexes
        get_diffusion_data() for the train range, or X_test for the test range).
        """
        Z_start_list, Z_end_list, valid_idx_list = [], [], []

        for pos, w_idx in enumerate(range(start_i, end_i)):
            i = w_idx * self.window_shift
            z_start = macro_values[i]
            z_end   = macro_values[i + self.seq_len - 1]

            if np.isnan(z_start) or np.isnan(z_end):
                continue

            Z_start_list.append(float(z_start))
            Z_end_list.append(float(z_end))
            valid_idx_list.append(pos)

        Z_start   = torch.tensor(Z_start_list, dtype=torch.float32)
        Z_end     = torch.tensor(Z_end_list,   dtype=torch.float32)
        valid_idx = torch.tensor(valid_idx_list, dtype=torch.long)
        return Z_start, Z_end, valid_idx

    def get_z_windows(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract Z_start and Z_end from the first ticker column (macro series)
        for the TRAINING windows. Only keeps windows where an actual macro
        observation exists at both the start and end of the window.

        Returns:
            Z_start   : (M,) standardized macro value near window start
            Z_end     : (M,) standardized macro value near window end
            valid_idx : (M,) indices into get_diffusion_data() for valid windows
        """
        macro_values, n_train = self._macro_std_values_and_n_train()
        n_train_windows = (n_train - self.seq_len) // self.window_shift + 1
        Z_start, Z_end, valid_idx = self._scan_macro_windows(macro_values, 0, n_train_windows)
        print(f"Z windows: {len(valid_idx)} valid out of {n_train_windows} training windows")
        return Z_start, Z_end, valid_idx

    def get_event_threshold_from_percentile(self, top_fraction: float, event_type: str) -> float:
        """
        Convert a desired "top X% of change" fraction into the equivalent raw numeric
        threshold, computed from TRAIN windows only (no leakage into test). Standardized-
        units thresholds are misleading here because Z_start/Z_end are highly correlated
        (slow-moving macro series over a short window), so a raw cutoff doesn't correspond
        to the percentile you'd expect from a single normal variable — computing directly
        off the empirical train distribution avoids that confusion.

        The quantile is taken over the metric matching event_type, since folding to an
        absolute value before quantiling (e.g. using |Z_end-Z_start| for a one-sided
        event_type like upper_change/lower_change) mixes both tails together and no
        longer corresponds to "top X%" of that one-sided metric:
          - "abs_change": top X% of |Z_end - Z_start|
          - "absval":     top X% of |Z_end|
          - "upper_change": top X% largest (most positive) Z_end - Z_start
          - "lower_change": top X% smallest (most negative) Z_end - Z_start
        """
        Z_start, Z_end, _ = self.get_z_windows()
        diffs = Z_end - Z_start
        if event_type == "abs_change":
            return diffs.abs().quantile(1.0 - top_fraction).item()
        elif event_type == "absval":
            return Z_end.abs().quantile(1.0 - top_fraction).item()
        elif event_type == "upper_change":
            return torch.quantile(diffs, 1.0 - top_fraction).item()
        elif event_type == "lower_change":
            return -torch.quantile(diffs, top_fraction).item()
        else:
            raise NotImplementedError(
                f"event_type={event_type!r} not supported by get_event_threshold_from_percentile; "
                "only 'abs_change', 'absval', 'upper_change', and 'lower_change' are implemented."
            )

    def _sequence_split_idx(self) -> int:
        """Train/test window-count boundary matching X_train.shape[0] exactly
        (same formula as train_test_split())."""
        if self.train_end_date is not None:
            cutoff = pd.to_datetime(self.train_end_date)
            return int((self.y_dates <= cutoff).sum())
        return len(self.X) - self.test_days

    def get_z_windows_test(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Z_start/Z_end/valid_idx for the real macro event, aligned with X_train/X_test
        sizes exactly (unlike get_z_windows(), which aligns with get_diffusion_data()
        and has one extra trailing window). valid_idx directly indexes X_test.
        """
        macro_values, _ = self._macro_std_values_and_n_train()
        split_idx = self._sequence_split_idx()
        Z_start, Z_end, valid_idx = self._scan_macro_windows(macro_values, split_idx, len(self.X))
        print(f"Event windows: {len(valid_idx)} valid out of {len(self.X) - split_idx} test windows")
        return Z_start, Z_end, valid_idx

    def get_z_windows_train_aligned(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Z_start/Z_end/valid_idx for the real macro event, aligned with X_train's
        actual size exactly (unlike get_z_windows(), which has one extra trailing
        window). valid_idx directly indexes X_train.
        """
        macro_values, _ = self._macro_std_values_and_n_train()
        split_idx = self._sequence_split_idx()
        Z_start, Z_end, valid_idx = self._scan_macro_windows(macro_values, 0, split_idx)
        print(f"Event windows: {len(valid_idx)} valid out of {split_idx} train windows")
        return Z_start, Z_end, valid_idx

    # NOTE: the legacy invert_samples() (global sigma_seq + weekday_mean inversion)
    # was removed with the switch to per-window EMA standardization — those
    # attributes are no longer populated. Use destandardize_windows() with
    # entry stats (own stats for real windows, sample_entry_stats() draws for
    # generated windows) instead.
