from typing import Optional

import pandas as pd


class StockDataStore:
    def __init__(self, ticker: str):
        self.ticker = ticker
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

        # self._store = self._raw.copy()

    def _convert_currency_to_float(self, column: str):
        """Convert currency formatted strings to float in the specified column."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._raw[column] = (
            self._raw[column].replace("[\\$,]", "", regex=True).astype(float)
        )

    def _preprocess_data(self):
        """Process the stock data by detecting outliers and scaling features."""

        # self._detect_and_handle_nulls()
        self._detect_and_handle_outliers()

        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._store = self._raw.copy()

    def _detect_and_handle_nulls(self):
        """
        Handle null values in the stock data.
        Handled by another teammate.
        """
        ...

    def _detect_and_handle_outliers(self):
        """Detect outliers and remove multivariate ones (bad data), flag univariate ones (real extremes)."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._raw["is_outlier"] = False
        self._raw["outlier_location"] = ""

        # Detect and flag univariate outliers (keep them for comparing with sentiment data)
        self._detect_univariate_outliers()

        # Detect and remove multivariate outliers (assuming bad data)
        self._detect_and_remove_multivariate_outliers()

    def _detect_univariate_outliers(self):
        """Detect univariate outliers in the stock data using IQR method."""

        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        numerical_cols = ["Open", "High", "Low", "Close", "Volume"]

        outlier_indices = set()

        for col in numerical_cols:
            if col not in self._raw.columns:
                continue

            # Calculate IQR
            Q1 = self._raw[col].quantile(0.25)
            Q3 = self._raw[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Find outliers
            col_outliers = self._raw[
                (self._raw[col] < lower_bound) | (self._raw[col] > upper_bound)
            ]

            if not col_outliers.empty:
                outlier_indices.update(col_outliers.index.tolist())
                self._raw.loc[col_outliers.index, "is_outlier"] = True
                self._raw.loc[col_outliers.index, "outlier_location"] += f"{col};"

    def _detect_and_remove_multivariate_outliers(self):
        """Detect and remove unusual patterns: large price moves with low volume."""

        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        # Calculate daily return (absolute price movement %)
        daily_return = self._raw["Close"].pct_change().abs() * 100

        # High movement = top 5% of price changes
        high_movement_threshold = daily_return.quantile(0.95)

        # Low volume = bottom 5% of volume
        low_volume_threshold = self._raw["Volume"].quantile(0.5)

        # Find rows with unusual pattern: big move + low volume
        high_movement = daily_return > high_movement_threshold
        low_volume = self._raw["Volume"] < low_volume_threshold

        unusual_pattern = high_movement & low_volume

        outlier_indices = self._raw[unusual_pattern].index.tolist()

        if len(outlier_indices) > 0:
            # Store removed outliers
            removed_outliers = self._raw.loc[outlier_indices].copy()

            if self._outliers is None:
                self._outliers = removed_outliers
            else:
                self._outliers = pd.concat(
                    [self._outliers, removed_outliers], ignore_index=True
                )

            # Remove them from raw data
            self._raw = self._raw.drop(outlier_indices).reset_index(drop=True)
            print(f"Removed {len(outlier_indices)} rows with unusual patterns")
        else:
            print("No multivariate outliers detected")


class DailyIndicatorCalculator:
    def __init__(self, stock_data: StockDataStore):
        self._stock_raw = stock_data.data
        self._indicators_raw = self._calculate_indicators()

    @property
    def data(self):
        """Get the DataFrame with technical indicators."""
        return self._indicators_raw

    def _calculate_indicators(self): ...
