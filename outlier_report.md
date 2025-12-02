### Outlier Detection

#### Objectives
- Identify and flag anomalous values that could distort model training.
- Handle both univariate outliers in market time series and a simple multivariate anomaly pattern.
- Catch raw data inconsistencies early (schema/price relationship issues) to prevent downstream contamination.

#### Scope
- Numerical stock data: `Open, High, Low, Close, Volume`
- Derived features: daily `return`, `hl_range` are used to detect multivariate anomalies.  (e.g., big price moves on low volume)
- Sentiment data: VADER `Sentiment_Score` at the post level (tweets/redit at the moment).

#### Data Quality and Consistency Checks
We first remove raw inconsistencies in OHLC relationships to ensure only feasible price bars remain:
- Enforce: `Low <= min(Open, Close)`, `High >= max(Open, Close)`, and `Low <= High`.
- Rows violating these constraints are flagged and quarantined into `_outliers` with the descriptor `inconsistent_data`.

Concrete implementation:
```python
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
```

Example output:
```
          Date   Close  Volume    Open    High     Low Ticker    return  \
1240 2025-10-10  400.49       0  436.54  443.13  411.45   TSLA -0.083898

      hl_range  is_outlier        outlier_desc
1240  0.072737        True  inconsistent_data;
```

#### Univariate Outlier Detection (IQR Method)
We apply IQR-based detection on key numeric columns. This is intentionally simple and robust for heavy-tailed distributions. We flag and do not correct the data in order to allow model visibility into unusual days, potentially allowing for corollation between sentiment data and market anomalies.:
- For each of `Open, High, Low, Close, Volume`, compute Q1, Q3, IQR and flag values outside `[Q1 − 1.5*IQR, Q3 + 1.5*IQR]`.
- Flags are appended to `outlier_desc` with the column name for traceability. These outliers remain in the dataset for modeling, but are easily identifiable for either further analysis, exclusion or specific training strategies downstream.

Concrete implementation:
```python
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
                prefix = np.where(self._raw.loc[idx, col] > upper, "high", "low")
                self._raw.loc[idx, "outlier_desc"] += prefix + f"_{col.lower()};"
```

Example output:
```
           Date   Close     Volume    Open    High     Low Ticker    return  \
0    2020-11-02  133.50   87063369  131.33  135.66  130.77   TSLA  0.000000
1    2020-11-03  141.30  103055170  136.58  142.59  135.56   TSLA  0.056784
2    2020-11-04  140.33   96429190  143.54  145.13  139.03   TSLA -0.006888
3    2020-11-05  146.03   85243569  142.77  146.67  141.33   TSLA  0.039815
...         ...     ...        ...     ...     ...     ...    ...       ...
1250 2025-10-27  452.42  105867500  439.98  460.16  438.69   TSLA  0.042212
1251 2025-10-28  460.55   80185670  454.78  467.00  451.60   TSLA  0.017810
1252 2025-10-29  461.51   67983540  462.50  465.70  452.65   TSLA  0.002082
1253 2025-10-30  440.10   72447940  451.05  455.06  439.61   TSLA -0.047502

      hl_range  is_outlier                              outlier_desc
0     0.000000       False
1     0.052659       False
2     0.043171       False
3     0.038053       False
...        ...         ...                                       ...
1250  0.049502        True  high_open;high_high;high_low;high_close;
1251  0.034039        True  high_open;high_high;high_low;high_close;
1252  0.028336        True  high_open;high_high;high_low;high_close;
1253  0.033477        True  high_open;high_high;high_low;high_close;
```

#### Multivariate Outlier Detection (Sanity Check)
To catch unusual joint configurations, we add a conservative rule-based screen that quarantines “big move + low volume” events:
- Compute absolute log return, this was recommended after discussion with llm's as a more stable measure than raw % change.
- Flag days above a high-move percentile AND below a low-volume percentile.
- These rows are quarantined to `_outliers` with the descriptor `big_move_low_volume`.
- These outliers should be rare, as our dataset comes directly from the Nasdaq historical data. However, this serves as a way to make sure that potential glitches or misreported data points get caught before technical indicator computation.

Concrete implementation:
```python
def _detect_and_quarantine_multivariate_outliers(self):
    """Quarantine 'big move + low volume' using simple percentiles, but on returns."""
    if self._raw is None:
        raise ValueError("Dataframe is empty. Load data first.")

    abs_ret = self._raw["return"].abs()
    high_move_thr = abs_ret.quantile(1 - self.outlier_threshold)
    low_vol_thr = self._raw["Volume"].quantile(self.outlier_threshold)

    high_move = abs_ret > high_move_thr
    low_volume = self._raw["Volume"] < low_vol_thr
    unusual_pattern = high_move & low_volume

    if unusual_pattern.any():
        removed = self._raw.loc[unusual_pattern].copy()
        removed["is_outlier"] = True
        removed["outlier_desc"] += "big_move_low_volume;"

        self._outliers = (
            removed
            if self._outliers is None
            else pd.concat([self._outliers, removed], ignore_index=True)
        )
        self._raw = self._raw.loc[~unusual_pattern].reset_index(drop=True)
```

Example output:
```
          Date   Close  Volume    Open    High     Low Ticker    return  \
1240 2025-10-10  400.49       0  436.54  443.13  411.45   TSLA -0.083898

      hl_range  is_outlier        outlier_desc
1240  0.072737        True  big_move_low_volume;
```

#### Sentiment Outlier Filtering (Pre-Aggregation)
For sentiment, we first need to compute a VADER `Sentiment_Score` per article/tweet/redit posts, then apply z-score–based filtering to drop extreme items before daily aggregation. The z-score filtering is applied to daily data and not the entire dataset, this aims to keep overall data accuracy while making sure that extreme sentiment scores are excluded from daily aggregation. This keeps the daily aggregate from being dominated by a single viral or noisy post.
- Score per record using VADER compound, this produce a value between -1 and 1.
- Z-score thresholding (|z| > 2.5) on `Sentiment_Score` within the day prior to aggregation.
- The actual threshold probably needs to be tuned based on the extra data that will get added later. Right now having the threshold at 3 resulted in no outliers being flagged. Lowering to 2.5 produced a small number of outliers which seems reasonable but would need to be validated.
- Outliers are moved to `_outliers` with descriptor `daily_sentiment_z>2.5`. Following the outlier handling pattern that we have used in other places.

Concrete implementation (scoring and method contract):
```python
def _process_sentiment_scores(self):
    """Calculate sentiment scores for the stock data."""
    if self._store is None:
        raise ValueError("Dataframe is empty. Load data first.")

    analyzer = SentimentIntensityAnalyzer()
    self._store["Sentiment_Score"] = self._store["Text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

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

    # Identify outliers |z| > 2.5
    zscore_mask = self._store["sentiment_z"].abs() > 2.5

    if zscore_mask.any():
        removed = self._store.loc[zscore_mask].copy()
        removed["outlier_desc"] = "daily_sentiment_z>2.5"

        # Append to outliers store for audit
        if self._outliers is None:
            self._outliers = removed
        else:
            self._outliers = pd.concat([self._outliers, removed], ignore_index=True)

        # Keep non-outliers in main store
        self._store = self._store.loc[~mask].reset_index(drop=True)
```

Example output:
```
                       ID       Date Ticker         Username  \
2119  1546469276428734466 2022-07-11   TSLA  Zachasbeenthere
3063  1546363531695738881 2022-07-11   TSLA   grtissotauthor
4719  1546381782366683137 2022-07-11   TSLA    Astra55152472

                                                   Text  Retweets  Likes  \
2119  @JoeMagnus74 @Forbes He was for sure wrong on ...         0      0
3063  May 2022: my Instagram has been hacked July 20...         1      8
4719  Elon musk can taste my fat toes, he is a bitch...         0      1

      Sentiment_Score  sentiment_z         outlier_desc
2119          -0.9626    -2.502992  daily_sentiment_z>2.5
3063          -0.9766    -2.536197  daily_sentiment_z>2.5
4719          -0.9778    -2.539043  daily_sentiment_z>2.5
```

---

### Data Transformation

#### Technical Indicator Engineering
We compute standard daily technical indicators that are robust from the series start by using `min_periods=1` and EMA-style smoothing where appropriate:
- SMA(20/50/200)
- RSI(14) with Wilder-style EMA smoothing
- MACD(12,26) line
- Bollinger Bands(20, 2σ)
- Annualized rolling volatility over 20 days (std of returns × sqrt(252))

Concrete implementation:
```python
def _calculate_indicators(self) -> pd.DataFrame:
    if self._stock_store is None or len(self._stock_store) == 0:
        raise ValueError("Stock data is empty. Load data first.")

    df_src = self._stock_store
    ind = pd.DataFrame(index=df_src.index)
    ind["calculation_date"] = df_src["Date"]
    ind["ticker"] = df_src["Ticker"]

    close = cast(pd.Series, df_src["Close"])

    ind["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    ind["sma_50"] = close.rolling(window=50, min_periods=1).mean()
    ind["sma_200"] = close.rolling(window=200, min_periods=1).mean()

    ind["rsi_14"] = self._calculate_rsi(close, window=14)

    macd = self._calculate_macd(close)
    ind["macd"] = macd

    sma20 = close.rolling(window=20, min_periods=1).mean()
    std20 = close.rolling(window=20, min_periods=1).std(ddof=0)
    ind["bollinger_upper"] = sma20 + 2 * std20
    ind["bollinger_lower"] = sma20 - 2 * std20

    ind["volatility_20d"] = df_src["return"].rolling(window=20, min_periods=1).std(
        ddof=0
    ) * np.sqrt(252)

    ind = ind.reset_index(drop=True)
    return ind

def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
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
```

Example output:
```
     calculation_date ticker      sma_20      sma_50     sma_200     rsi_14  \
0          2020-11-02   TSLA  133.500000  133.500000  133.500000   0.000000
1          2020-11-03   TSLA  137.400000  137.400000  137.400000   0.000000
2          2020-11-04   TSLA  138.376667  138.376667  138.376667  88.189250
3          2020-11-05   TSLA  140.290000  140.290000  140.290000  93.241039
...               ...    ...         ...         ...         ...        ...
1250       2025-10-27   TSLA  440.313000  398.776800  335.978150  59.326749
1251       2025-10-28   TSLA  441.180000  401.376600  336.306200  61.748208
1252       2025-10-29   TSLA  442.019500  403.903600  336.640050  62.035623
1253       2025-10-30   TSLA  441.051500  406.119400  336.824000  52.551943

           macd  bollinger_upper  bollinger_lower  volatility_20d
0      0.000000       133.500000       133.500000        0.000000
1      0.622222       145.200000       129.600000        0.450708
2      1.025248       145.318642       131.434691        0.452911
3      1.784027       149.238374       131.341626        0.423370
...         ...              ...              ...             ...
1250  11.109596       457.011333       423.614667        0.514069
1251  11.881122       460.049454       422.310546        0.515823
1252  12.426778       462.837586       421.201414        0.515930
1253  11.004750       460.275113       421.827887        0.537945
```

#### Normalization of technical Indicators
We produce normalized/relative features that are dimensionless and bounded where possible for model stability:
- `rsi_14_norm` in [0,1].
- Distance to moving averages: `sma_20_ratio`, `sma_50_ratio`.
- Crossover flags: `golden_cross`, `death_cross`.
- Price-relative MACD: `macd_pct`.
- Relative Bollinger position: `bb_pos` (clipped).
- Volatility z-score across the series: `volatility_20d_z`.

Concrete implementation:
```python
def _normalize_indicators(self) -> pd.DataFrame:
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

    close = self._stock_store["Close"]
    sma20 = ind["sma_20"]
    sma50 = ind["sma_50"]
    sma200 = ind["sma_200"]

    ind["rsi_14_norm"] = ind["rsi_14"] / 100.0
    ind["sma_20_ratio"] = (close / sma20.replace(0, np.nan)) - 1.0
    ind["sma_50_ratio"] = (close / sma50.replace(0, np.nan)) - 1.0

    prev_above = sma50.shift(1) > sma200.shift(1)
    curr_above = sma50 > sma200
    ind["golden_cross"] = (~prev_above) & curr_above
    ind["death_cross"] = prev_above & (~curr_above)

    ind["macd_pct"] = ind["macd"] / close.replace(0, np.nan)

    std20 = (ind["bollinger_upper"] - ind["sma_20"]) / 2.0
    denom = (2.0 * std20).replace(0, np.nan)
    ind["bb_pos"] = (close - sma20) / denom

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
    return cast(pd.DataFrame, ind[keep_cols].copy())
```

Example output:
```
     calculation_date ticker  rsi_14_norm  sma_20_ratio  sma_50_ratio  \
0          2020-11-02   TSLA     0.000000      0.000000      0.000000
1          2020-11-03   TSLA     0.000000      0.028384      0.028384
2          2020-11-04   TSLA     0.881893      0.014116      0.014116
3          2020-11-05   TSLA     0.932410      0.040915      0.040915
...               ...    ...          ...           ...           ...
1250       2025-10-27   TSLA     0.593267      0.027496      0.134519
1251       2025-10-28   TSLA     0.617482      0.043905      0.147426
1252       2025-10-29   TSLA     0.620356      0.044094      0.142624
1253       2025-10-30   TSLA     0.525519     -0.002157      0.083671

      golden_cross  death_cross  macd_pct    bb_pos  volatility_20d_z
0            False        False  0.000000       NaN         -3.251769
1            False        False  0.004404  0.500000         -0.660703
2            False        False  0.007306  0.281380         -0.648035
3            False        False  0.012217  0.641457         -0.817864
...            ...          ...       ...       ...               ...
1250         False        False  0.024556  0.725042         -0.296442
1251         False        False  0.025798  1.026527         -0.286363
1252         False        False  0.026926  0.936229         -0.285744
1253         False        False  0.025005 -0.049496         -0.159187
```

#### Sentiment Score and Daily Aggregation
- A dedicated aggregation class provides the daily grouping interface.
- We ensure that only non-outlier sentiment scores (per the z-score filter) are included in the daily aggregation.


Concrete implementation:
```python
class SentimentAggregateCalculator:
    def __init__(self, stock_data: SentimentDataStore):
        self._stock_df = stock_data.data
        self._sentiment_aggregate_df = self._aggregate_sentiment_data()

    def _aggregate_sentiment_data(self) -> pd.DataFrame:
        """Aggregate sentiment data on a daily basis."""
        ...
```

Example output:
```
    aggregate_date  avg_news_sentiment  avg_social_sentiment  news_volume  \
0       2020-01-23            0.340000                   NaN          1.0
1       2020-01-29            0.358700                   NaN          6.0
2       2020-01-30            0.311400                   NaN          2.0
3       2020-01-31            0.079300                   NaN          1.0
..             ...                 ...                   ...          ...
469     2024-12-28            0.188120               -0.6820          5.0
470     2024-12-29            0.617850                   NaN          4.0
471     2024-12-30            0.040743                   NaN          7.0
472     2024-12-31            0.372956                   NaN          9.0

     social_volume  sentiment_variance_news  sentiment_variance_social  \
0              NaN                 0.000000                        NaN
1              NaN                 0.284231                        NaN
2              NaN                 0.113704                        NaN
3              NaN                 0.000000                        NaN
..             ...                      ...                        ...
469            1.0                 0.025593                        0.0
470            NaN                 0.008852                        NaN
471            NaN                 0.175953                        NaN
472            NaN                 0.104848                        NaN

    extreme_sentiment_flag_news extreme_sentiment_flag_social
0                         False                           NaN
1                         False                           NaN
2                         False                           NaN
3                         False                           NaN
..                          ...                           ...
469                       False                         False
470                       False                           NaN
471                       False                           NaN
472                       False                           NaN
```
