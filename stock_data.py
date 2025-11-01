from typing import Optional

import numpy as np
import pandas as pd


class StockDataStore:
    def __init__(self, ticker: str, outlier_threshold: float = 5.0):
        self.ticker = ticker
        self.outlier_threshold = outlier_threshold / 100.0
        self._raw: Optional[pd.DataFrame] = None
        self._outliers: Optional[pd.DataFrame] = None
        self._store: Optional[pd.DataFrame] = None

    @property
    def data(self):
        """Get the stock data DataFrame."""
        return self._store

    @property
    def outliers(self):
        """Get the outliers DataFrame."""
        return self._outliers

    def load_from_csv(self, file_path):
        """Load stock data from a CSV file. CSV is coming from Nasdaq historical data export."""

        self._raw = pd.read_csv(file_path, parse_dates=["Date"])
        self._raw["Ticker"] = self.ticker

        self._raw.sort_values(by="Date", inplace=True)
        self._raw.reset_index(drop=True, inplace=True)

        # Convert currency columns to float
        for col in ["Open", "High", "Low", "Close"]:
            self._convert_currency_to_float(col)

        self._preprocess_data()

        self._store = self._raw.copy()

    def _preprocess_data(self):
        self._add_simple_features()
        self._detect_and_handle_outliers()

    def _add_simple_features(self):
        """Add very basic helper columns."""

        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        # 1) Log return of Close (safer than raw %)
        ret = np.log(self._raw["Close"] / self._raw["Close"].shift(1)).fillna(0.0)
        self._raw["return"] = ret

        # 2) Intraday range as a simple quality signal
        hl_range = (self._raw["High"] - self._raw["Low"]) / self._raw["Close"].shift(
            1
        ).fillna(0.0)
        self._raw["hl_range"] = hl_range

    def _detect_and_handle_outliers(self):
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._raw["is_outlier"] = False
        self._raw["outlier_desc"] = ""

        # 1) Make sure data is consistent: Open|High|Close|Low -> Making sure no impossible values exist
        self._detect_inconsistencies()

        # 2) Univariate IQR for all numerical columns -> flag as outlier for comparison with large sentiment moves
        self._detect_univariate_outliers()

        # 3) Simple multivariate outlier detection: big move + low volume -> quarantine
        self._detect_and_quarantine_multivariate_outliers()

    def _detect_inconsistencies(self):
        """Drop rows where OHLC relationship is impossible."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        ok = (
            (self._raw["Low"] <= self._raw[["Open", "Close"]].min(axis=1))
            & (self._raw["High"] >= self._raw[["Open", "Close"]].max(axis=1))
            & (self._raw["Low"] <= self._raw["High"])
        )

        bad_idx = ~ok

        if bad_idx.any():
            removed = self._raw.loc[bad_idx].copy()
            removed["is_outlier"] = True
            removed["outlier_desc"] += "inconsistent_data;"
            self._outliers = (
                removed
                if self._outliers is None
                else pd.concat([self._outliers, removed], ignore_index=True)
            )
            self._raw = self._raw.loc[~bad_idx].reset_index(drop=True)

    def _detect_univariate_outliers(self):
        """Your original IQR method (unchanged)."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        numerical_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numerical_cols:
            Q1 = self._raw[col].quantile(0.25)
            Q3 = self._raw[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            idx = self._raw.index[(self._raw[col] < lower) | (self._raw[col] > upper)]

            if len(idx) > 0:
                self._raw.loc[idx, "is_outlier"] = True
                self._raw.loc[idx, "outlier_desc"] += f"{col};"

    def _detect_and_quarantine_multivariate_outliers(self):
        """Quarantine 'big move + low volume' using simple percentiles, but on returns."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        # Absolute log return size
        abs_ret = self._raw["return"].abs()

        # High move = top (1 - threshold) quantile
        high_move_thr = abs_ret.quantile(1 - self.outlier_threshold)

        # Low volume = bottom threshold quantile
        low_vol_thr = self._raw["Volume"].quantile(self.outlier_threshold)

        high_move = abs_ret > high_move_thr
        low_volume = self._raw["Volume"] < low_vol_thr
        unusual_pattern = high_move & low_volume

        if unusual_pattern.any():
            removed = self._raw.loc[unusual_pattern].copy()
            removed["is_outlier"] = True
            removed["outlier_desc"] += "big_move_low_volume;"

            # Append to Outliers DataFrame for quarantined data
            self._outliers = (
                removed
                if self._outliers is None
                else pd.concat([self._outliers, removed], ignore_index=True)
            )

            # Use boolean mask to filter out the unusual patterns from the main DataFrame
            self._raw = self._raw.loc[~unusual_pattern].reset_index(drop=True)

    def _convert_currency_to_float(self, column: str):
        """Convert currency formatted strings to float in the specified column."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._raw[column] = (
            self._raw[column].replace("[\\$,]", "", regex=True).astype(float)
        )

    def _detect_and_handle_nulls(self):
        """
        Handle null values in the stock data.
        Handled by another teammate.
        """
        ...


class DailyIndicatorCalculator:
    def __init__(self, stock_data: StockDataStore):
        self._stock_raw = stock_data.data
        self._indicators_raw = self._calculate_indicators()

    @property
    def data(self):
        """Get the DataFrame with technical indicators."""
        return self._indicators_raw

    def _calculate_indicators(self): ...
