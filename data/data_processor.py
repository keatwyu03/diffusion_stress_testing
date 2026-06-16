"""
Data processing module for financial time series
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Optional


class DataProcessor:
    """Process financial time series data"""

    def __init__(
        self,
        csv_path: str,
        tickers: List[str],
        weekday_col: str = "weekday",
        seq_len: int = 64,
        test_days: int = 700,
        winsorize_lower: float = 0.005,
        winsorize_upper: float = 0.995,
    ):
        self.csv_path = csv_path
        self.tickers = tickers
        self.weekday_col = weekday_col
        self.seq_len = seq_len
        self.test_days = test_days
        self.winsorize_lower = winsorize_lower
        self.winsorize_upper = winsorize_upper

        # Data containers
        self.df = None
        self.r_dw = None
        self.weekday_mean = None
        self.sigma_seq = None
        self.df_z = None
        self.df_z_wins = None

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
        """Load returns from CSV"""
        df = pd.read_csv(self.csv_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
        df = df[self.tickers].dropna()
        self.df = df
        return df

    def remove_weekday_effect(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove weekday effect from returns"""
        # weekday_mean = self.df.groupby(self.weekday_col)[self.tickers].mean()
        # aligned = self.df[self.weekday_col].map(weekday_mean.to_dict("index"))
        # aligned_df = pd.DataFrame(list(aligned), index=self.df.index)[self.tickers]
        weekday_series = pd.Series(self.df.index.dayofweek, index=self.df.index)
        weekday_mean = self.df.groupby(weekday_series)[self.tickers].mean()
        aligned_df = weekday_series.map(weekday_mean.to_dict("index"))
        aligned_df = pd.DataFrame(list(aligned_df), index=self.df.index)[self.tickers]
        r_dw = self.df[self.tickers] - aligned_df

        self.r_dw = r_dw
        self.weekday_mean = weekday_mean
        return r_dw, weekday_mean

    def standardize(self) -> pd.DataFrame:
        """Standardize using train-set std only to avoid data leakage"""
        data = self.r_dw if self.r_dw is not None else self.df
        train_data = data.iloc[:-self.test_days]
        self.mu_seq = train_data.mean()
        self.sigma_seq = train_data.std()
        z = data / self.sigma_seq
        self.df_z = z.dropna(how="any")
        return self.df_z

    def winsorize(self) -> pd.DataFrame:
        """Winsorize the standardized returns"""
        df_wins = self.df_z.copy()
        for col in df_wins.columns:
            q_low = df_wins[col].quantile(self.winsorize_lower)
            q_high = df_wins[col].quantile(self.winsorize_upper)
            df_wins[col] = df_wins[col].clip(lower=q_low, upper=q_high)
        self.df_z_wins = df_wins
        return df_wins

    def make_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences from standardized returns"""
        z_values = self.df_z.values
        dates = self.df_z.index.to_numpy()
        T, D = z_values.shape
        X_list, y_list, idx_list = [], [], []

        for t in range(self.seq_len, T):
            X_list.append(z_values[t - self.seq_len : t, :])
            y_list.append(z_values[t, :])
            idx_list.append(dates[t])

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
        idx = np.array(idx_list)

        self.X = X
        self.y = y
        self.y_dates = idx
        return X, y, idx

    def train_test_split(self) -> None:
        """Split data into training and test sets"""
        if self.X is None:
            raise ValueError("Must call make_sequences() first")

        # NumPy splits
        X_train_np = self.X[: -self.test_days]
        y_train_np = self.y[: -self.test_days]
        y_dates_train = self.y_dates[: -self.test_days]

        X_test_np = self.X[-self.test_days :]
        y_test_np = self.y[-self.test_days :]
        y_dates_test = self.y_dates[-self.test_days :]

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

        # print("Removing weekday effect...")
        # self.remove_weekday_effect()
        self.r_dw = self.df[self.tickers]  # skip weekday removal, use raw data

        print("Standardizing...")
        self.standardize()
        print(f"Standardized data shape: {self.df_z.shape}")

        print("Winsorizing...")
        self.winsorize()

        print("Creating sequences...")
        self.make_sequences()
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

        print(f"Splitting into train/test (test_days={self.test_days})...")
        self.train_test_split()
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def get_diffusion_data(self) -> torch.Tensor:
        """Get data for diffusion model training (winsorized windows)"""
        data = torch.tensor(self.df_z_wins.values, dtype=torch.float32)
        window_size = self.seq_len
        X_all = []
        for i in range(len(data) - window_size + 1):
            X_all.append(data[i : i + window_size].T)
        X_all = torch.stack(X_all)

        # Return only training portion
        X_diffusion_train = X_all[: -self.test_days]
        return X_diffusion_train

    def invert_samples(
        self,
        samples: torch.Tensor,
        start_weekday: Optional[int] = None,
        monthly = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
        """
        Invert standardized samples back to returns

        Args:
            samples: (T, D) tensor of standardized samples
            start_weekday: Starting weekday (0-4), if None will be random

        Returns:
            r_seq: Returns dataframe
            r_dw_seq: De-weekday returns dataframe
            sigma_path: Sigma path
            port_seq: Portfolio returns series
        """
        z_seq = (
            samples.detach().cpu().numpy()
            if hasattr(samples, "detach")
            else np.asarray(samples)
        )

        if monthly:
            r_seq = z_seq * self.sigma_seq.to_numpy() + self.mu_seq.to_numpy()
            return pd.DataFrame(r_seq, columns = self.tickers), None, None, None

        T, D = z_seq.shape

        if start_weekday is None:
            start_weekday = np.random.randint(0, 5)

        r_seq = np.zeros((T, D))
        r_dw_seq = np.zeros((T, D))
        sigma_path = np.tile(self.sigma_seq.to_numpy(), (T + 1, 1))
        port_seq = np.zeros(T)

        for t in range(T):
            r_dw_t = z_seq[t] * self.sigma_seq.to_numpy() + self.mu_seq.to_numpy()
            w_t = (start_weekday + t) % 5
            r_t = r_dw_t + self.weekday_mean.loc[w_t, self.tickers].to_numpy()

            r_dw_seq[t] = r_dw_t
            r_seq[t] = r_t
            port_seq[t] = r_t.mean()

        return (
            pd.DataFrame(r_seq, columns=self.tickers),
            pd.DataFrame(r_dw_seq, columns=self.tickers),
            sigma_path,
            pd.Series(port_seq, name="Portfolio"),
        )


"""
Data processing module for financial time series
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Optional


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
        winsorize_lower: float = 0.005,
        winsorize_upper: float = 0.995,
    ):
        self.csv_path = csv_path
        self.tickers = tickers
        self.weekday_col = weekday_col
        self.seq_len = seq_len
        self.test_days = test_days
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = train_end_date
        self.winsorize_lower = winsorize_lower
        self.winsorize_upper = winsorize_upper

        # Data containers
        self.df = None
        self.r_dw = None
        self.weekday_mean = None
        self.sigma_seq = None
        self.df_z = None
        self.df_z_wins = None

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
        df = df[self.tickers].dropna()
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

    def standardize(self) -> pd.DataFrame:
        """Standardize using train-set std only to avoid data leakage"""
        if self.train_end_date is not None:
            cutoff = pd.to_datetime(self.train_end_date)
            train_data = self.r_dw[self.r_dw.index <= cutoff]
        else:
            train_data = self.r_dw.iloc[:-self.test_days]
        self.mu_seq = train_data.mean()
        self.sigma_seq = train_data.std()
        z = self.r_dw / self.sigma_seq
        self.df_z = z.dropna(how="any")
        return self.df_z

    def winsorize(self) -> pd.DataFrame:
        """Winsorize the standardized returns"""
        df_wins = self.df_z.copy()
        for col in df_wins.columns:
            q_low = df_wins[col].quantile(self.winsorize_lower)
            q_high = df_wins[col].quantile(self.winsorize_upper)
            df_wins[col] = df_wins[col].clip(lower=q_low, upper=q_high)
        self.df_z_wins = df_wins
        return df_wins

    def make_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences from standardized returns"""
        z_values = self.df_z.values
        dates = self.df_z.index.to_numpy()
        weekday_arr = self.df_z.index.dayofweek.values
        T, D = z_values.shape
        X_list, y_list, idx_list, sw_list = [], [], [], []

        for t in range(self.seq_len, T):
            X_list.append(z_values[t - self.seq_len : t, :])
            y_list.append(z_values[t, :])
            idx_list.append(dates[t])
            sw_list.append(int(weekday_arr[t - self.seq_len]))  # weekday of window start

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
        idx = np.array(idx_list)

        self.X = X
        self.y = y
        self.y_dates = idx
        self.start_weekdays = np.array(sw_list, dtype=np.int8)
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

        # print("Removing weekday effect...")
        # self.remove_weekday_effect()
        self.r_dw = self.df[self.tickers]  # skip weekday removal, use raw data

        print("Standardizing...")
        self.standardize()
        print(f"Standardized data shape: {self.df_z.shape}")

        print("Winsorizing...")
        self.winsorize()

        print("Creating sequences...")
        self.make_sequences()
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

        split_info = f"train_end_date={self.train_end_date}" if self.train_end_date else f"test_days={self.test_days}"
        print(f"Splitting into train/test ({split_info})...")
        self.train_test_split()
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def get_diffusion_data(self) -> torch.Tensor:
        """Get data for diffusion model training (winsorized windows)"""
        data = torch.tensor(self.df_z_wins.values, dtype=torch.float32)
        window_size = self.seq_len
        X_all = []
        for i in range(len(data) - window_size + 1):
            X_all.append(data[i : i + window_size].T)
        X_all = torch.stack(X_all)

        # Return only training portion, consistent with train_test_split boundary
        if self.train_end_date is not None:
            cutoff = pd.to_datetime(self.train_end_date)
            dates_all = self.df_z_wins.index
            # Each window i ends at dates_all[i + window_size - 1]
            end_dates = dates_all[window_size - 1:]
            n_train = int((end_dates <= cutoff).sum())
            X_diffusion_train = X_all[:n_train]
        else:
            X_diffusion_train = X_all[: -self.test_days]
        return X_diffusion_train

    def invert_samples(
        self,
        samples: torch.Tensor,
        start_weekday: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
        """
        Invert standardized samples back to returns

        Args:
            samples: (T, D) tensor of standardized samples
            start_weekday: Starting weekday (0-4), if None will be random

        Returns:
            r_seq: Returns dataframe
            r_dw_seq: De-weekday returns dataframe
            sigma_path: Sigma path
            port_seq: Portfolio returns series
        """
        z_seq = (
            samples.detach().cpu().numpy()
            if hasattr(samples, "detach")
            else np.asarray(samples)
        )
        T, D = z_seq.shape

        if start_weekday is None:
            start_weekday = np.random.randint(0, 5)

        r_seq = np.zeros((T, D))
        r_dw_seq = np.zeros((T, D))
        sigma_path = np.tile(self.sigma_seq.to_numpy(), (T + 1, 1))
        port_seq = np.zeros(T)

        for t in range(T):
            r_dw_t = z_seq[t] * self.sigma_seq.to_numpy()
            w_t = (start_weekday + t) % 5
            r_t = r_dw_t + self.weekday_mean.loc[w_t, self.tickers].to_numpy()

            r_dw_seq[t] = r_dw_t
            r_seq[t] = r_t
            port_seq[t] = r_t.mean()

        return (
            pd.DataFrame(r_seq, columns=self.tickers),
            pd.DataFrame(r_dw_seq, columns=self.tickers),
            sigma_path,
            pd.Series(port_seq, name="Portfolio"),
        )
