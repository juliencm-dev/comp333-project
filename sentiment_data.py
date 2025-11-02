from pathlib import Path
from typing import Optional, cast

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentDataStore:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self._store: Optional[pd.DataFrame] = None

    @property
    def data(self):
        """Get the stock data DataFrame."""
        return self._store

    def load_reddit(self):
        """Placeholder for Reddit data loading method."""
        raise NotImplementedError("Reddit data loading not implemented yet.")

    def load_news(self):
        """Placeholder for News data loading method."""
        raise NotImplementedError("News data loading not implemented yet.")

    def load_tweets(self):
        """
        Load all CSV files from a directory, append them into a single DataFrame,
        parse dates to YYYY-MM-DD, keep a whitelist of columns, then preprocess.
        """
        dir_path = Path("raw_tweets")
        csv_files = sorted(dir_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dir_path}")

        dfs: list[pd.DataFrame] = []
        for fp in csv_files:
            df = pd.read_csv(fp)

            # Twitter format: "Mon Dec 21 15:46:05 +0000 2020" → use strict format first
            dt1 = pd.to_datetime(
                df["Date"],
                format="%a %b %d %H:%M:%S %z %Y",
                utc=True,
                errors="coerce",
            )

            # Fallback parse for any rows not matching the strict Twitter pattern
            mask = dt1.isna()
            if mask.any():
                dt2 = pd.to_datetime(df.loc[mask, "Date"], utc=True, errors="coerce")
                dt1[mask] = dt2

            df["Date"] = dt1.dt.strftime("%Y-%m-%d")  # or use .dt.date
            df["Ticker"] = self.ticker
            dfs.append(df)

        all_df = pd.concat(dfs, ignore_index=True)

        keep_cols = ["ID", "Date", "Ticker", "Username", "Text", "Retweets", "Likes"]

        if self._store is not None:
            self._store = cast(
                pd.DataFrame,
                pd.concat([self._store, all_df[keep_cols].copy()], ignore_index=True),
            )
        else:
            self._store = cast(pd.DataFrame, all_df[keep_cols].copy())

        # Sort and drop dupes by ID if present
        self._store.sort_values("Date", inplace=True)
        self._store.drop_duplicates(subset=["ID"], inplace=True, keep="first")
        self._store.reset_index(drop=True, inplace=True)

        self._preprocess_data()

    def _process_sentiment_scores(self):
        """Calculate sentiment scores for the stock data."""
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")

        analyzer = SentimentIntensityAnalyzer()

        self._store["Sentiment_Score"] = self._store["Text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"]
        )

    def _preprocess_data(self):
        """Process the stock data by detecting outliers and scaling features."""
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._detect_nulls()
        self._process_sentiment_scores()
        self._detect_outliers()

    def _detect_nulls(self):
        """
        Handle null values in the stock data.
        Handled by another teammate.
        """
        ...

    def _detect_outliers(self):
        """
        Detect and handle outliers in the stock data.

        Use Z-Score method to identify outliers in the 'Sentiment_Score' column.
        Remove rows where the absolute Z-Score is greater than 3.

        """
        ...


class SentimentAggregateCalculator:
    def __init__(self, stock_data: SentimentDataStore):
        self._stock_df = stock_data.data
        self._sentiment_aggregate_df = self._aggregate_sentiment_data()

    @property
    def data(self):
        """Get the DataFrame with technical indicators."""
        return self._sentiment_aggregate_df

    def _aggregate_sentiment_data(self) -> pd.DataFrame:
        """Aggregate sentiment data on a daily basis."""
        ...
