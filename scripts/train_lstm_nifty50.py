import os
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------- PATHS & CONFIG ----------

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLDATA_DIR = ROOT_DIR / "data" / "allData"
META_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# NIFTY 50 constituents CSV
NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
NIFTY50_LOCAL_PATH = META_DIR / "nifty50_list.csv"

SEQ_LEN = 30          # days in input sequence
YEARS = 2             # last 2 years per stock
BATCH_SIZE = 64
EPOCHS = 20

MODEL_PATH = MODEL_DIR / "lstm_nifty50.h5"
SCALER_PATH = MODEL_DIR / "lstm_nifty50_scaler.npz"


# ---------- HELPERS ----------

def get_nifty50_symbols():
    """
    Return list of NIFTY 50 symbols (EQ series).
    - Use local cached CSV if present.
    - Else download from NSE Indices and cache.
    - If download fails, return [] to fall back to "all files in allData".
    """
    if NIFTY50_LOCAL_PATH.exists():
        df = pd.read_csv(NIFTY50_LOCAL_PATH)
    else:
        print("Downloading NIFTY 50 list...")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.niftyindices.com"
            }
            resp = requests.get(NIFTY50_URL, headers=headers, timeout=20)
            resp.raise_for_status()
            csv_text = resp.content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
            META_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(NIFTY50_LOCAL_PATH, index=False)
            print(f"Saved NIFTY 50 list to {NIFTY50_LOCAL_PATH}")
        except Exception as e:
            print(f"Could not download NIFTY 50 list: {e}")
            print("Falling back to: all symbols in data/allData")
            return []

    # CSV columns: Company Name, Industry, Symbol, Series, ISIN Code
    df = df[df["Series"] == "EQ"]
    return [s.strip().upper() for s in df["Symbol"].tolist()]


def load_stock_df(path: Path):
    """
    Load one stock CSV from data/allData.
    Expected columns: date,open,high,low,close,volume,empty
    Returns last `YEARS` years, sorted by date.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=365 * YEARS)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df


def build_dataset():
    """
    Build X, y from all NIFTY 50 stocks present in data/allData.
    X shape: (num_samples, SEQ_LEN, 5)
    y shape: (num_samples,)  -> next-day normalized close
    """
    print(f"Scanning {ALLDATA_DIR} for CSV files...")
    all_files = sorted(p for p in ALLDATA_DIR.glob("*.csv"))
    if not all_files:
        raise RuntimeError(f"No CSV files found in {ALLDATA_DIR}")

    nifty50_syms = set(get_nifty50_symbols())

    selected_files = []
    for f in all_files:
        symbol = f.stem.upper()
        if not nifty50_syms:
            use_it = True   # fallback: use everything
        else:
            use_it = symbol in nifty50_syms

        if use_it:
            selected_files.append((symbol, f))

    if not selected_files:
        raise RuntimeError("No overlapping symbols between NIFTY50 and data/allData/*.csv")

    print(f"Using {len(selected_files)} symbols for training")

    # First pass: collect all features to compute global min/max
    all_features_list = []
    stock_data = {}

    for symbol, f in selected_files:
        df = load_stock_df(f)
        if len(df) <= SEQ_LEN + 1:
            print(f"  {symbol}: too few rows after 2y filter, skipping")
            continue
        feats = df[["open", "high", "low", "close", "volume"]].values.astype("float32")
        stock_data[symbol] = df
        all_features_list.append(feats)

    if not all_features_list:
        raise RuntimeError("No stock had enough data to build sequences.")

    all_features = np.vstack(all_features_list)
    feat_min = all_features.min(axis=0)
    feat_max = all_features.max(axis=0)
    denom = (feat_max - feat_min)
    denom[denom == 0] = 1.0

    print("Global feature mins:", feat_min)
    print("Global feature maxs:", feat_max)

    # Second pass: build sequences
    X_list = []
    y_list = []
    close_idx = 3  # open, high, low, close, volume

    for symbol, df in stock_data.items():
        feats = df[["open", "high", "low", "close", "volume"]].values.astype("float32")
        feats_norm = (feats - feat_min) / denom

        n = len(feats_norm)
        for i in range(n - SEQ_LEN - 1):
            window = feats_norm[i:i + SEQ_LEN]
            target = feats_norm[i + SEQ_LEN, close_idx]
            X_list.append(window)
            y_list.append(target)

    X = np.stack(X_list)
    y = np.array(y_list, dtype="float32")

    print(f"Built dataset: X shape = {X.shape}, y shape = {y.shape}")
    return X, y, feat_min, feat_max


def train_model():
    X, y, feat_min, feat_max = build_dataset()
    num_features = X.shape[-1]

    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, num_features)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

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

    print("Starting training (NIFTY50)...")
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
