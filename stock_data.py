from typing import Optional, cast

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

    def add_simple_features(self):
        """Add very basic helper columns."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        # 1) Log return of Close (safer than raw %)
        ret = np.log(self._raw["Close"] / self._raw["Close"].shift(1)).fillna(0.0)
        self._raw["return"] = ret

        # 2) Intraday range as a simple quality signal
        hl_range = (
            (self._raw["High"] - self._raw["Low"]) / self._raw["Close"].shift(1)
        ).fillna(0.0)
        self._raw["hl_range"] = hl_range

    def detect_and_handle_outliers(self):
        """Detect and handle outliers in the stock data."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._raw["is_outlier"] = False
        self._raw["outlier_desc"] = ""

        # 1) Make sure data is consistent: Open|High|Close|Low -> Making sure no impossible values exist
        self.detect_inconsistencies()

        # 2) Univariate IQR for all numerical columns -> flag as outlier for comparison with large sentiment moves
        self.detect_univariate_outliers()

        # 3) Simple multivariate outlier detection: big move + low volume -> quarantine
        self.detect_and_quarantine_multivariate_outliers()

    def detect_inconsistencies(self):
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

    def detect_univariate_outliers(self):
        """IQR method for univariate outlier detection."""
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
                prefix = np.where(self._raw.loc[idx, col] > upper, "high", "low")
                self._raw.loc[idx, "outlier_desc"] += prefix + f"_{col.lower()};"

    def detect_and_quarantine_multivariate_outliers(self):
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

    def detect_and_handle_nulls(self):
        """
        Handle null values in the stock data.
        Handled by another teammate.
        """
        pass

    def finalize(self):
        """Finalize the data store by copying raw data to store."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")
        self._store = self._raw.copy()

    def _convert_currency_to_float(self, column: str):
        """Convert currency formatted strings to float in the specified column."""
        if self._raw is None:
            raise ValueError("Dataframe is empty. Load data first.")

        self._raw[column] = (
            self._raw[column].replace("[\\$,]", "", regex=True).astype(float)
        )


class DailyIndicatorStore:
    def __init__(self, stock_data: StockDataStore):
        self._stock_store = stock_data.data
        self._raw: Optional[pd.DataFrame] = None
        self._store: Optional[pd.DataFrame] = None

    @property
    def normalized(self) -> pd.DataFrame:
        if self._store is None:
            raise ValueError("Normalized indicators not calculated yet.")
        return self._store

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            raise ValueError("Raw indicators not calculated yet.")
        return self._raw

    def calculate_indicators(self):
        """Calculate technical indicators from stock data."""
        if self._stock_store is None or len(self._stock_store) == 0:
            raise ValueError("Stock data is empty. Load data first.")

        df_src = self._stock_store

        # Create the indicator frame using the same index so we have one row per day
        ind = pd.DataFrame(index=df_src.index)
        ind["calculation_date"] = df_src["Date"]
        ind["ticker"] = df_src["Ticker"]

        close = cast(pd.Series, df_src["Close"])

        # Simple moving averages (allow values from the start)
        ind["sma_20"] = close.rolling(window=20, min_periods=1).mean()
        ind["sma_50"] = close.rolling(window=50, min_periods=1).mean()
        ind["sma_200"] = close.rolling(window=200, min_periods=1).mean()

        # RSI (use EMA-style smoothing so it fills from the start nicely)
        ind["rsi_14"] = self._calculate_rsi(close, window=14)

        # MACD (classic: macd line only; add signal if you want)
        macd = self._calculate_macd(close)
        ind["macd"] = macd

        # Bollinger Bands (20-day, allow early values)
        sma20 = close.rolling(window=20, min_periods=1).mean()
        std20 = close.rolling(window=20, min_periods=1).std(ddof=0)
        ind["bollinger_upper"] = sma20 + 2 * std20
        ind["bollinger_lower"] = sma20 - 2 * std20

        # Volatility 20d: rolling std of returns, annualized
        ind["volatility_20d"] = df_src["return"].rolling(window=20, min_periods=1).std(
            ddof=0
        ) * np.sqrt(252)

        # Reset index so it's a clean daily table
        ind = ind.reset_index(drop=True)
        self._raw = ind

    def normalize_indicators(self):
        """
        Create normalized/relative versions of raw indicators.
        - Keep raw columns as-is.
        - Add easy-to-model relative features (ratios/positions).
        """
        if self._stock_store is None or len(self._stock_store) == 0:
            raise ValueError("Stock data is empty. Load data first.")

        if self._raw is None or len(self._raw) == 0:
            raise ValueError("Indicator data is empty. Calculate indicators first.")

        ind = self._raw.copy()

        # Convenience aliases
        close = self._stock_store["Close"]
        sma20 = ind["sma_20"]
        sma50 = ind["sma_50"]
        sma200 = ind["sma_200"]

        # RSI to [0,1]
        ind["rsi_14_norm"] = ind["rsi_14"] / 100.0

        # Relative to moving averages (dimensionless)
        ind["sma_20_ratio"] = (close / sma20.replace(0, np.nan)) - 1.0
        ind["sma_50_ratio"] = (close / sma50.replace(0, np.nan)) - 1.0

        # Golden/Death cross
        prev_above = sma50.shift(1) > sma200.shift(1)
        curr_above = sma50 > sma200

        ind["golden_cross"] = (~prev_above) & curr_above
        ind["death_cross"] = prev_above & (~curr_above)

        # MACD as a percent of price (dimensionless)
        ind["macd_pct"] = ind["macd"] / close.replace(0, np.nan)

        # Bollinger position in band: roughly in [-1, +1]
        std20 = (ind["bollinger_upper"] - ind["sma_20"]) / 2.0  # recover std20
        denom = (2.0 * std20).replace(0, np.nan)
        ind["bb_pos"] = (close - sma20) / denom

        # Volatility z-score across the single ticker (optional but useful)
        vol = ind["volatility_20d"]
        ind["volatility_20d_z"] = (vol - vol.mean()) / (
            vol.std(ddof=0) if vol.std(ddof=0) != 0 else np.nan
        )

        ind["rsi_14_norm"] = ind["rsi_14_norm"].clip(0, 1)
        ind["bb_pos"] = ind["bb_pos"].clip(-3, 3)

        for c in ["macd_pct", "sma_20_ratio", "sma_50_ratio"]:
            ind[c] = ind[c].clip(-0.5, 0.5)

        keep_cols = [
            "calculation_date",
            "ticker",
            "rsi_14_norm",
            "sma_20_ratio",
            "sma_50_ratio",
            "golden_cross",
            "death_cross",
            "macd_pct",
            "bb_pos",
            "volatility_20d_z",
        ]

        self._store = cast(pd.DataFrame, ind[keep_cols].copy())

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        # Use Wilder's RSI with EMA smoothing to avoid big NaN runs
        delta = series.diff().fillna(0.0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return cast(pd.Series, rsi).fillna(0.0)

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26):
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=1).mean()
        macd_line = ema_fast - ema_slow

        return macd_line
