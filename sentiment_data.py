from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from outliers import DetectionMethodEnum, SentimentDataOutliers


class SentimentkDataStore:
    def __init__(self):
        self._store: Optional[pd.DataFrame] = None
        self._outliers: List[SentimentDataOutliers] = []

        self._preprocess_data()

        if self._outliers is not None:
            print("Outliers detected in the sentiment data. Actions may be required.")
            self._process_outliers()

    @property
    def data(self):
        """Get the stock data DataFrame."""
        return self._store

    def load_data_from_csv(self, file_path):
        """Load stock data from a CSV file."""
        self._df = pd.read_csv(file_path, parse_dates=["Date"])
        self._df.sort_values(by="Date", inplace=True)
        self._df.reset_index(drop=True, inplace=True)

    def _process_outliers(self):
        """Handle outliers in the stock data."""

    def process_nulls(self): ...

    def _preprocess_data(self):
        """Process the stock data by detecting outliers and scaling features."""
        if self._df is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._detect_nulls()
        self._detect_outliers()

    def _detect_nulls(self):
        """
        Handle null values in the stock data.
        Handled by another teammate.
        """
        ...

    def _detect_outliers(self):
        """Detect outliers in the stock data."""
        ...


class SentimentAggregateCalculator:
    def __init__(self, stock_data: SentimentkDataStore):
        self._stock_df = stock_data.data
        self._sentiment_aggregate_df = self._aggregate_sentiment_data()

    @property
    def data(self):
        """Get the DataFrame with technical indicators."""
        return self._sentiment_aggregate_df

    def _aggregate_sentiment_data(self) -> pd.DataFrame:
        """Aggregate sentiment data on a daily basis."""
        ...
