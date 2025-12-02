from pathlib import Path
from typing import Optional, cast

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentDataStore:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self._store: Optional[pd.DataFrame] = None
        self._outliers: Optional[pd.DataFrame] = None

    @property
    def data(self):
        """Get the stock data DataFrame."""
        return self._store

    @property
    def outliers(self):
        """Get the outliers DataFrame."""
        return self._outliers

    def load_reddit(self):
        """Placeholder for Reddit data loading method."""
        raise NotImplementedError("Reddit data loading not implemented yet.")

    def load_news(self):
        """Placeholder for News data loading method."""
        raise NotImplementedError("News data loading not implemented yet.")

    def load_tweets(self):
        """
        Load all tweets from the raw_tweets directory. Format Date and Keep desiered columns.
        """
        dir_path = Path("raw_tweets")
        csv_files = sorted(dir_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dir_path}")

        dfs: list[pd.DataFrame] = []
        count = 0
        for fp in csv_files:
            df = pd.read_csv(fp)
            count += 1

            # Twitter format: "Mon Dec 21 15:46:05 +0000 2020" -> From self scrapped tweets
            dt1 = pd.to_datetime(
                df["Date"],
                format="%a %b %d %H:%M:%S %z %Y",
                utc=True,
                errors="coerce",
            )

            # Look for NaN and let Pandas infer the datetime conversion (Kaggle Data Set)
            mask = dt1.isna()
            if mask.any():
                dt2 = pd.to_datetime(df.loc[mask, "Date"], utc=True, errors="coerce")
                dt1[mask] = dt2

            df["Date"] = dt1.dt.strftime("%Y-%m-%d")
            df["Ticker"] = self.ticker
            df["Platform"] = "Social Media" if count % 2 == 0 else "News"
            dfs.append(df)

        all_df = pd.concat(dfs, ignore_index=True)

        keep_cols = [
            "ID",
            "Date",
            "Ticker",
            "Username",
            "Text",
            "Retweets",
            "Likes",
            "Platform",
        ]

        if self._store is not None:
            self._store = cast(
                pd.DataFrame,
                pd.concat([self._store, all_df[keep_cols].copy()], ignore_index=True),
            )
        else:
            self._store = cast(pd.DataFrame, all_df[keep_cols].copy())

        # Sort by date and drop duplicates tweets with tweet ID
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
        Daily z-score filter on Sentiment_Score.
        For each Date, compute z-scores and drop rows where |z| > 3.
        Assumes Sentiment_Score already exists.
        """
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")
        if "Sentiment_Score" not in self._store.columns:
            raise ValueError("Sentiment_Score not found.")

        # Compute within-day z-scores
        def _zscore(s: pd.Series) -> pd.Series:
            mean = s.mean()
            std_deviation = s.std(ddof=0)
            if std_deviation == 0:
                return pd.Series(0.0, index=s.index)
            return (s - mean) / std_deviation

        self._store["sentiment_z"] = self._store.groupby("Date")[
            "Sentiment_Score"
        ].transform(_zscore)

        # Identify outliers |z| > 3
        mask = self._store["sentiment_z"].abs() > 2.5

        if mask.any():
            # Initialize/append to self._outliers for review
            removed = self._store.loc[mask].copy()
            removed["outlier_desc"] = "daily_sentiment_z>2.5"

            # Append to outliers DataFrame for auditing
            if self._outliers is None:
                self._outliers = removed
            else:
                self._outliers = pd.concat([self._outliers, removed], ignore_index=True)

            # Keep non-outliers in main store
            self._store = self._store.loc[~mask].reset_index(drop=True)


class SentimentAggregateDataStore:
    def __init__(self, sentiment_data: SentimentDataStore):
        self._sentiment_df = sentiment_data.data
        self._sentiment_aggregate_df = self._aggregate_sentiment_data()

    @property
    def data(self):
        """Get the DataFrame with technical indicators."""
        return self._sentiment_aggregate_df

    def _aggregate_sentiment_data(self) -> pd.DataFrame:
        """
        Aggregate sentiment daily by platform ("Social Media" | "News") after outlier filtering.
        Produces:
          - avg_news_sentiment, avg_social_sentiment
          - news_volume, social_volume
          - sentiment_variance_news, sentiment_variance_social
          - extreme_sentiment_flag_news, extreme_sentiment_flag_social
        """
        if self._sentiment_df is None:
            raise ValueError("No sentiment data to aggregate.")

        df = self._sentiment_df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].copy()

        # Daily, per-platform aggregates
        grouped = (
            df.groupby([df["Date"].dt.date, "Platform"])["Sentiment_Score"]
            .agg(avg_sentiment="mean", volume="count", variance=lambda s: s.var(ddof=0))
            .reset_index()
            .rename(columns={"Date": "aggregate_date"})
        )

        # Extreme flag per day/platform based on mean magnitude (tunable threshold)
        grouped["extreme_sentiment_flag"] = grouped["avg_sentiment"].abs() > 0.8

        # Merge grouped data into single row per date combining platforms as columns
        merged = grouped.pivot(index="aggregate_date", columns="Platform")

        # Combine columns that are a combination of metric and platform into a single column name
        merged.columns = [
            f"{metric}_{platform}".lower().replace(" ", "_")
            for metric, platform in merged.columns
        ]

        merged = merged.reset_index()

        return merged
