import os
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout,
    Flatten, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------- PATHS & CONFIG ----------

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLDATA_DIR = ROOT_DIR / "data" / "allData"
META_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# NIFTY 200 constituents CSV
NIFTY200_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
NIFTY200_LOCAL_PATH = META_DIR / "nifty200_list.csv"

SEQ_LEN = 30          # days in input sequence
YEARS = 2             # last 2 years per stock
BATCH_SIZE = 64
EPOCHS = 25

MODEL_PATH = MODEL_DIR / "parallel_lstm_cnn_nifty200.h5"
SCALER_PATH = MODEL_DIR / "parallel_lstm_cnn_nifty200_scaler.npz"


# ---------- FEATURE ENGINEERING ----------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["pct_change"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(),
                               (df["low"] - prev_close).abs()))
    df["tr"] = tr
    df["atr_14"] = df["tr"].rolling(14).mean()

    df["volatility_20"] = df["log_return"].rolling(20).std()

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_5"] = df["close"].shift(5)

    df = df.dropna().reset_index(drop=True)
    return df


# ---------- HELPERS ----------

def get_nifty200_symbols():
    if NIFTY200_LOCAL_PATH.exists():
        df = pd.read_csv(NIFTY200_LOCAL_PATH)
    else:
        print("Downloading NIFTY 200 list...")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.niftyindices.com"
            }
            resp = requests.get(NIFTY200_URL, headers=headers, timeout=20)
            resp.raise_for_status()
            csv_text = resp.content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
            META_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(NIFTY200_LOCAL_PATH, index=False)
            print(f"Saved NIFTY 200 list to {NIFTY200_LOCAL_PATH}")
        except Exception as e:
            print(f"Could not download NIFTY 200 list: {e}")
            print("Falling back to: all symbols in data/allData")
            return []

    df = df[df["Series"] == "EQ"]
    return [s.strip().upper() for s in df["Symbol"].tolist()]


def load_stock_df(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=365 * YEARS)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = add_features(df)
    return df


def build_dataset():
    print(f"Scanning {ALLDATA_DIR} for CSV files...")
    all_files = sorted(p for p in ALLDATA_DIR.glob("*.csv"))
    if not all_files:
        raise RuntimeError(f"No CSV files found in {ALLDATA_DIR}")

    nifty200_syms = set(get_nifty200_symbols())

    selected_files = []
    for f in all_files:
        symbol = f.stem.upper()
        if not nifty200_syms:
            use_it = True
        else:
            use_it = symbol in nifty200_syms
        if use_it:
            selected_files.append((symbol, f))

    if not selected_files:
        raise RuntimeError("No overlapping symbols between NIFTY200 and data/allData/*.csv")

    print(f"Using {len(selected_files)} symbols for training")

    all_features_list = []
    stock_data = {}

    for symbol, f in selected_files:
        df = load_stock_df(f)
        if len(df) <= SEQ_LEN + 1:
            print(f"  {symbol}: too few rows after 2y + features, skipping")
            continue

        feature_cols = [c for c in df.columns if c != "date"]
        feats = df[feature_cols].values.astype("float32")

        stock_data[symbol] = (df, feature_cols)
        all_features_list.append(feats)

    if not all_features_list:
        raise RuntimeError("No stock had enough data to build sequences with features.")

    all_features = np.vstack(all_features_list)
    feat_min = all_features.min(axis=0)
    feat_max = all_features.max(axis=0)
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0

    print("Global feature mins:", feat_min)
    print("Global feature maxs:", feat_max)

    X_list = []
    y_list = []

    for symbol, (df, feature_cols) in stock_data.items():
        feats = df[feature_cols].values.astype("float32")
        feats_norm = (feats - feat_min) / denom

        try:
            close_idx = feature_cols.index("close")
        except ValueError:
            raise RuntimeError("'close' not found in feature columns")

        n = len(feats_norm)
        for i in range(n - SEQ_LEN - 1):
            window = feats_norm[i:i + SEQ_LEN]
            target = feats_norm[i + SEQ_LEN, close_idx]
            X_list.append(window)
            y_list.append(target)

    X = np.stack(X_list)
    y = np.array(y_list, dtype="float32")

    print(f"Built dataset with features: X shape = {X.shape}, y shape = {y.shape}")
    return X, y, feat_min, feat_max


def build_parallel_lstm_cnn(seq_len: int, num_features: int):
    """
    Parallel LSTM + CNN branches merged:
      - Branch 1: LSTM(64)
      - Branch 2: Conv1D -> Conv1D -> Flatten
      - Concatenate -> Dense -> Output
    """
    inp = Input(shape=(seq_len, num_features))

    # LSTM branch
    x1 = LSTM(64)(inp)

    # CNN branch
    x2 = Conv1D(32, 3, activation="relu", padding="causal")(inp)
    x2 = Conv1D(32, 3, activation="relu", padding="causal")(x2)
    x2 = Flatten()(x2)

    # Merge
    x = Concatenate()([x1, x2])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def train_model():
    X, y, feat_min, feat_max = build_dataset()
    num_features = X.shape[-1]

    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    model = build_parallel_lstm_cnn(SEQ_LEN, num_features)

    checkpoint_cb = ModelCheckpoint(
        MODEL_PATH.as_posix(),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("Starting training (Parallel LSTM+CNN NIFTY200)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    model.save(MODEL_PATH.as_posix())
    print(f"Model saved to {MODEL_PATH}")

    np.savez(SCALER_PATH, feat_min=feat_min, feat_max=feat_max)
    print(f"Scaler saved to {SCALER_PATH}")

    return history


if __name__ == "__main__":
    train_model()
