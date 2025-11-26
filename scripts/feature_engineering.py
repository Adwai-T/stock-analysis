import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame):
    """
    Add engineered stock price prediction features.
    df must contain: date, open, high, low, close, volume
    """

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ---- Returns ----
    df["pct_change"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # ---- Moving Averages ----
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    # ---- EMA ----
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # ---- RSI ----
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ---- ATR ----
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - prev_close).abs()
    low_close = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()

    # ---- Volatility ----
    df["volatility_20"] = df["log_return"].rolling(20).std()

    # ---- Lag/Shifts ----
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_5"] = df["close"].shift(5)

    # Clean NaN rows
    df = df.dropna().reset_index(drop=True)

    return df
