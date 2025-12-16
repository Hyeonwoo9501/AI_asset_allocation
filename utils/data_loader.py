"""
Data Loader for Sector ETFs and Macro Indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class MarketDataLoader:
    """Load and preprocess market data (ETF returns and macro indicators)"""

    def __init__(self, config: dict):
        self.config = config
        self.sector_etfs = config['data']['sector_etfs']
        self.macro_indicators = config['data']['macro_indicators']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.freq = config['data']['freq']
        self.lookback = config['data']['lookback_window']

        # Initialize FRED API (you need to set your API key)
        self.fred = None
        try:
            self.fred = Fred(api_key='YOUR_FRED_API_KEY')  # Replace with your key
        except:
            print("Warning: FRED API key not set. Set it in environment or here.")

    def fetch_etf_data(self) -> pd.DataFrame:
        """Fetch ETF price data and calculate returns"""
        print("Fetching ETF data...")

        # Download ETF prices
        data = yf.download(
            self.sector_etfs,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )['Adj Close']

        # Calculate returns
        if self.freq == 'D':
            returns = data.pct_change()
        elif self.freq == 'M':
            data = data.resample('M').last()
            returns = data.pct_change()
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        returns = returns.dropna()
        print(f"ETF data shape: {returns.shape}")

        return returns

    def fetch_macro_data(self) -> pd.DataFrame:
        """Fetch macro indicators from FRED"""
        print("Fetching macro data...")

        if self.fred is None:
            print("Warning: Using dummy macro data. Please set FRED API key.")
            # Create dummy data for testing
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
            dummy_data = pd.DataFrame(
                np.random.randn(len(dates), len(self.macro_indicators)),
                index=dates,
                columns=self.macro_indicators
            )
            return dummy_data

        macro_data = {}
        for indicator in self.macro_indicators:
            try:
                series = self.fred.get_series(
                    indicator,
                    observation_start=self.start_date,
                    observation_end=self.end_date
                )
                macro_data[indicator] = series
            except Exception as e:
                print(f"Error fetching {indicator}: {e}")

        macro_df = pd.DataFrame(macro_data)

        # Resample to match frequency
        if self.freq == 'D':
            macro_df = macro_df.resample('D').ffill()
        elif self.freq == 'M':
            macro_df = macro_df.resample('M').last()

        # Fill missing values
        macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')

        # Normalize macro data (z-score)
        macro_df = (macro_df - macro_df.mean()) / macro_df.std()

        print(f"Macro data shape: {macro_df.shape}")

        return macro_df

    def align_data(self, etf_returns: pd.DataFrame, macro_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align ETF returns and macro data on common dates"""
        common_dates = etf_returns.index.intersection(macro_data.index)

        etf_aligned = etf_returns.loc[common_dates]
        macro_aligned = macro_data.loc[common_dates]

        print(f"Aligned data shape: {etf_aligned.shape}, {macro_aligned.shape}")

        return etf_aligned, macro_aligned

    def create_sequences(
        self,
        etf_returns: pd.DataFrame,
        macro_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for time-series modeling

        Returns:
            X_sector: (N, lookback, n_sectors) - sector returns
            X_macro: (N, lookback, n_macros) - macro indicators
            y: (N, n_sectors) - next period returns (target)
        """
        n_samples = len(etf_returns) - self.lookback
        n_sectors = etf_returns.shape[1]
        n_macros = macro_data.shape[1]

        X_sector = np.zeros((n_samples, self.lookback, n_sectors))
        X_macro = np.zeros((n_samples, self.lookback, n_macros))
        y = np.zeros((n_samples, n_sectors))

        for i in range(n_samples):
            X_sector[i] = etf_returns.iloc[i:i+self.lookback].values
            X_macro[i] = macro_data.iloc[i:i+self.lookback].values
            y[i] = etf_returns.iloc[i+self.lookback].values

        print(f"Sequence shapes - X_sector: {X_sector.shape}, X_macro: {X_macro.shape}, y: {y.shape}")

        return X_sector, X_macro, y

    def split_data(
        self,
        X_sector: np.ndarray,
        X_macro: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Split data into train/val/test sets (time-series split)"""
        n_samples = len(X_sector)

        train_size = int(n_samples * self.config['data']['train_ratio'])
        val_size = int(n_samples * self.config['data']['val_ratio'])

        train_data = (
            X_sector[:train_size],
            X_macro[:train_size],
            y[:train_size]
        )

        val_data = (
            X_sector[train_size:train_size+val_size],
            X_macro[train_size:train_size+val_size],
            y[train_size:train_size+val_size]
        )

        test_data = (
            X_sector[train_size+val_size:],
            X_macro[train_size+val_size:],
            y[train_size+val_size:]
        )

        print(f"Train samples: {len(train_data[0])}")
        print(f"Val samples: {len(val_data[0])}")
        print(f"Test samples: {len(test_data[0])}")

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    def load_all_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Main method to load and prepare all data"""
        # Fetch data
        etf_returns = self.fetch_etf_data()
        macro_data = self.fetch_macro_data()

        # Align data
        etf_returns, macro_data = self.align_data(etf_returns, macro_data)

        # Create sequences
        X_sector, X_macro, y = self.create_sequences(etf_returns, macro_data)

        # Split data
        data_splits = self.split_data(X_sector, X_macro, y)

        return data_splits


if __name__ == "__main__":
    # Test data loader
    import yaml

    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    loader = MarketDataLoader(config)
    data = loader.load_all_data()

    print("\nData loading complete!")
    print(f"Train: {data['train'][0].shape}")
    print(f"Val: {data['val'][0].shape}")
    print(f"Test: {data['test'][0].shape}")
