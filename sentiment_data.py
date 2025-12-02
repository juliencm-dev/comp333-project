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
    def raw(self):
        """Get the raw stock data DataFrame."""
        return self._store

    @property
    def data(self):
        """Get the stock data DataFrame."""
        return self._store

    @property
    def outliers(self):
        """Get the outliers DataFrame."""
        return self._outliers

    def load_reddit(self):
        """
        Load all reddit posts from the data/reddit directory.
        Concatenates Title and Content, standardizes dates, and filters for Tesla relevance.
        """
        dir_path = Path("data/reddit")
        csv_files = sorted(dir_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dir_path}")

        # Keywords specific to Tesla Motors
        tesla_keywords = [
            "Tesla",
            "TSLA",
            "Elon Musk",
            "Musk",
            "Cybertruck",
            "Model S",
            "Model 3",
            "Model X",
            "Model Y",
            "Gigafactory",
            "Starlink",
        ]

        # Create a regex pattern: (Tesla|TSLA|Elon Musk|...)
        # case=False in str.contains handles the case insensitivity
        keyword_pattern = "|".join(tesla_keywords)

        dfs: list[pd.DataFrame] = []

        for fp in csv_files:
            df = pd.read_csv(fp)

            # 1. Standardize Date
            df["date"] = pd.to_datetime(df["post_date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )

            # 2. Combine Title and Content
            df["title"] = df["title"].fillna("")
            df["content"] = df["content"].fillna("")
            df["text"] = df["title"] + " " + df["content"]

            # We filter immediately after creating the 'text' column to save processing time
            # case=False ensures 'tesla', 'TESLA', and 'Tesla' are all caught
            df = df[df["text"].str.contains(keyword_pattern, case=False, na=False)]

            # If the dataframe is empty after filtering, skip to the next file
            if df.empty:
                continue

            # 3. Create Interaction Feature
            df["interaction"] = df["upvotes"] + df["comments"]

            # 4. Standardize Columns
            df["ticker"] = self.ticker
            df["platform"] = "REDDIT"

            # Rename specific columns
            df = cast(pd.DataFrame, df).rename(columns={"post_id": "id"})

            dfs.append(df)

        if not dfs:
            print("No relevant Tesla Reddit posts found.")
            return

        all_df = pd.concat(dfs, ignore_index=True)

        # Select only the columns we need for the ML task
        keep_cols = ["date", "ticker", "text", "interaction", "platform", "id"]

        # Filter columns
        all_df = all_df[keep_cols]

        # Merge into main store
        if self._store is not None:
            self._store = cast(
                pd.DataFrame, pd.concat([self._store, all_df], ignore_index=True)
            )
        else:
            self._store = cast(pd.DataFrame, all_df)

        # Deduplicate
        self._store.sort_values("date", inplace=True)
        self._store.drop_duplicates(
            subset=["id", "platform"], inplace=True, keep="first"
        )
        self._store.reset_index(drop=True, inplace=True)

    def load_tweets(self):
        """
        Load all tweets from the raw_tweets directory. Format Date and Keep desired columns.
        """
        dir_path = Path("data/tweets")
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
            df["ticker"] = self.ticker
            df["platform"] = "TWITTER"

            # Create Interaction Feature
            df["interaction"] = df["Retweets"] + df["Likes"]

            df.rename(columns={"ID": "id"}, inplace=True)
            df.rename(columns={"Date": "date"}, inplace=True)
            df.rename(columns={"Text": "text"}, inplace=True)

            dfs.append(df)

        all_df = pd.concat(dfs, ignore_index=True)

        keep_cols = [
            "id",
            "ticker",
            "date",
            "text",
            "interaction",
            "platform",
        ]

        if self._store is not None:
            self._store = cast(
                pd.DataFrame,
                pd.concat([self._store, all_df[keep_cols].copy()], ignore_index=True),
            )
        else:
            self._store = cast(pd.DataFrame, all_df[keep_cols].copy())

        # Sort by date and drop duplicates tweets with tweet ID
        self._store.sort_values("date", inplace=True)
        self._store.drop_duplicates(subset=["id"], inplace=True, keep="first")
        self._store.reset_index(drop=True, inplace=True)

    def process_sentiment_scores(self):
        """Calculate sentiment scores for the stock data."""
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")

        analyzer = SentimentIntensityAnalyzer()

        self._store["Sentiment_Score"] = self._store["Text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"]
        )

    def detect_nulls(self):
        """
        Handle null values in the stock data.
        Handled by another teammate.
        """
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")

        # Example: Drop rows with null text or date
        self._store.dropna(subset=["text", "date"], inplace=True)
        self._store.reset_index(drop=True, inplace=True)

    def detect_outliers(self):
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
        self._sentiment_aggregate_df: Optional[pd.DataFrame] = None

    @property
    def data(self):
        """Get the DataFrame with technical indicators."""
        return self._sentiment_aggregate_df

    def aggregate_sentiment_data(self):
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

        self._sentiment_aggregate_df = merged
