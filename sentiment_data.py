from pathlib import Path
from typing import Optional, cast

import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
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
        print(f"--- Loading Reddit Data for {self.ticker} ---")
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
        keyword_pattern = "|".join(tesla_keywords)

        dfs: list[pd.DataFrame] = []

        # Using tqdm to show progress bar for file loading
        for fp in tqdm(csv_files, desc="Processing Reddit CSVs", unit="file"):
            try:
                df = pd.read_csv(fp)

                # 1. Standardize Date
                df["date"] = pd.to_datetime(
                    df["post_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")

                # 2. Combine Title and Content
                df["title"] = df["title"].fillna("")
                df["content"] = df["content"].fillna("")
                df["text"] = df["title"] + " " + df["content"]

                # Filter immediately
                df = df[df["text"].str.contains(keyword_pattern, case=False, na=False)]

                if df.empty:
                    continue

                # 3. Create Interaction Feature
                df["interaction"] = df["upvotes"] + df["comments"]

                # 4. Standardize Columns
                df["ticker"] = self.ticker
                df["platform"] = "REDDIT"

                df = cast(pd.DataFrame, df).rename(columns={"post_id": "id"})
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to process file {fp.name}: {e}")

        if not dfs:
            print("No relevant Tesla Reddit posts found.")
            return

        all_df = pd.concat(dfs, ignore_index=True)

        keep_cols = ["date", "ticker", "text", "interaction", "platform", "id"]
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

        # --- Statistics Output ---
        reddit_count = self._store[self._store["platform"] == "REDDIT"].shape[0]
        print(f"\n✅ Reddit Data Loaded. Total Reddit Rows: {reddit_count}")

    def load_tweets(self):
        """
        Load all tweets from the raw_tweets directory. Format Date and Keep desired columns.
        """
        print(f"--- Loading Twitter Data for {self.ticker} ---")
        dir_path = Path("data/tweets")
        csv_files = sorted(dir_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dir_path}")

        dfs: list[pd.DataFrame] = []

        # Using tqdm for Twitter files
        for fp in tqdm(csv_files, desc="Processing Twitter CSVs", unit="file"):
            try:
                df = pd.read_csv(fp)

                # Twitter format handling
                dt1 = pd.to_datetime(
                    df["Date"],
                    format="%a %b %d %H:%M:%S %z %Y",
                    utc=True,
                    errors="coerce",
                )

                mask = dt1.isna()
                if mask.any():
                    dt2 = pd.to_datetime(
                        df.loc[mask, "Date"], utc=True, errors="coerce"
                    )
                    dt1[mask] = dt2

                df["Date"] = dt1.dt.strftime("%Y-%m-%d")
                df["ticker"] = self.ticker
                df["platform"] = "TWITTER"

                df["interaction"] = df["Retweets"] + df["Likes"]

                df = df.rename(columns={"ID": "id", "Date": "date", "Text": "text"})
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to process file {fp.name}: {e}")

        all_df = pd.concat(dfs, ignore_index=True)

        keep_cols = ["id", "ticker", "date", "text", "interaction", "platform"]

        if self._store is not None:
            self._store = cast(
                pd.DataFrame,
                pd.concat([self._store, all_df[keep_cols].copy()], ignore_index=True),
            )
        else:
            self._store = cast(pd.DataFrame, all_df[keep_cols].copy())

        self._store.sort_values("date", inplace=True)
        self._store.drop_duplicates(subset=["id"], inplace=True, keep="first")
        self._store.reset_index(drop=True, inplace=True)

        # --- Statistics Output ---
        twitter_count = self._store[self._store["platform"] == "TWITTER"].shape[0]
        print(f"\n✅ Twitter Data Loaded. Total Twitter Rows: {twitter_count}")
        print(f"\n📊 Total Combined Rows: {self._store.shape[0]}")

    def process_sentiment_scores(self):
        """Calculate sentiment scores for the stock data."""
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")

        print("--- Calculating Sentiment Scores (VADER) ---")
        analyzer = SentimentIntensityAnalyzer()

        # Using tqdm with pandas apply is possible via `tqdm.pandas()`
        tqdm.pandas(desc="Analyzing Sentiment")

        self._store["sentiment_score"] = self._store["text"].progress_apply(
            lambda x: analyzer.polarity_scores(str(x))["compound"]
        )

        avg_score = self._store["sentiment_score"].mean()
        print(
            f"\n✅ Sentiment Analysis Complete. Average Compound Score: {avg_score:.4f}"
        )

    def detect_and_handle_nulls(self):
        """
        Handle null values in the stock data.
        """
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")

        initial_count = len(self._store)
        self._store.dropna(subset=["text", "date"], inplace=True)
        self._store.reset_index(drop=True, inplace=True)

        dropped_count = initial_count - len(self._store)
        if dropped_count > 0:
            print(f"\n⚠️ Dropped {dropped_count} rows due to null Text or Date.")

    def detect_and_handle_outliers(self):
        """
        Daily z-score filter on Sentiment_Score.
        """
        if self._store is None:
            raise ValueError("Dataframe is empty. Load data first.")
        if "sentiment_score" not in self._store.columns:
            raise ValueError("Sentiment_Score not found.")

        print("--- Detecting Sentiment Outliers ---")

        def _zscore(s: pd.Series) -> pd.Series:
            mean = s.mean()
            std_deviation = s.std(ddof=0)
            if std_deviation == 0:
                return pd.Series(0.0, index=s.index)
            return (s - mean) / std_deviation

        self._store["sentiment_z"] = self._store.groupby("date")[
            "sentiment_score"
        ].transform(_zscore)

        mask = self._store["sentiment_z"].abs() > 2.5
        outlier_count = mask.sum()

        if mask.any():
            removed = self._store.loc[mask].copy()
            removed["outlier_desc"] = "daily_sentiment_z>2.5"

            if self._outliers is None:
                self._outliers = removed
            else:
                self._outliers = pd.concat([self._outliers, removed], ignore_index=True)

            self._store = self._store.loc[~mask].reset_index(drop=True)

        print(
            f"\n🧹 Outlier Detection Complete. Removed {outlier_count} outliers (Z-Score > 2.5)."
        )


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
        Aggregate sentiment daily by platform.
        """
        if self._sentiment_df is None:
            raise ValueError("No sentiment data to aggregate.")

        print("--- Aggregating Daily Sentiment ---")

        df = self._sentiment_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].copy()

        # Daily, per-platform aggregates
        grouped = (
            df.groupby([df["date"].dt.date, "platform"])["sentiment_score"]
            .agg(avg_sentiment="mean", volume="count", variance=lambda s: s.var(ddof=0))
            .reset_index()
            .rename(columns={"date": "aggregate_date"})
        )

        grouped["extreme_sentiment_flag"] = grouped["avg_sentiment"].abs() > 0.8

        merged = grouped.pivot(index="aggregate_date", columns="platform")

        merged.columns = [
            f"{metric}_{platform}".lower().replace(" ", "_")
            for metric, platform in merged.columns
        ]

        merged = merged.reset_index()
        self._sentiment_aggregate_df = merged

        print(
            f"\n✅ Aggregation Complete. Generated {len(self._sentiment_aggregate_df)} daily records."
        )
        print(
            f"\n📅 Date Range: {self._sentiment_aggregate_df['aggregate_date'].min()} to {self._sentiment_aggregate_df['aggregate_date'].max()}"
        )
