import os
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
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
EPOCHS = 20

MODEL_PATH = MODEL_DIR / "cnn_lstm_nifty200_feature.h5"          # <-- renamed
SCALER_PATH = MODEL_DIR / "cnn_lstm_nifty200_scaler_feature.npz" # <-- renamed


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
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com"}
            resp = requests.get(NIFTY200_URL, headers=headers, timeout=20)
            resp.raise_for_status()
            csv_text = resp.content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
            META_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(NIFTY200_LOCAL_PATH, index=False)
        except Exception:
            print("Could not download — using fallback")
            return []
    df = df[df["Series"] == "EQ"]
    return [s.strip().upper() for s in df["Symbol"].tolist()]


def load_stock_df(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = df["date"].max() - pd.Timedelta(days=365 * YEARS)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return add_features(df)


def build_dataset():
    print(f"Scanning {ALLDATA_DIR} for CSV files...")
    all_files = sorted(p for p in ALLDATA_DIR.glob("*.csv"))

    nifty200_syms = set(get_nifty200_symbols())
    selected = [(f.stem.upper(), f) for f in all_files if (not nifty200_syms or f.stem.upper() in nifty200_syms)]

    stock_data = {}
    all_feats = []

    for sym, f in selected:
        df = load_stock_df(f)
        if len(df) <= SEQ_LEN + 1:
            continue
        feat_cols = [c for c in df.columns if c != "date"]
        arr = df[feat_cols].values.astype("float32")
        stock_data[sym] = (df, feat_cols)
        all_feats.append(arr)

    all_arr = np.vstack(all_feats)
    feat_min = all_arr.min(axis=0)
    feat_max = all_arr.max(axis=0)
    scale = np.where((feat_max - feat_min) == 0, 1, feat_max - feat_min)

    X_list, y_list = [], []

    for sym, (df, feat_cols) in stock_data.items():
        arr = df[feat_cols].values.astype("float32")
        arr_norm = (arr - feat_min) / scale
        close_idx = feat_cols.index("close")

        for i in range(len(arr_norm) - SEQ_LEN - 1):
            X_list.append(arr_norm[i:i+SEQ_LEN])
            y_list.append(arr_norm[i+SEQ_LEN, close_idx])

    return np.stack(X_list), np.array(y_list), feat_min, feat_max


def train_model():
    X, y, feat_min, feat_max = build_dataset()
    num_features = X.shape[-1]

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # ---------- CNN + LSTM Hybrid ----------
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal", input_shape=(SEQ_LEN, num_features)),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    checkpoint_cb = ModelCheckpoint(MODEL_PATH.as_posix(), save_best_only=True, monitor="val_loss", verbose=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS,
                        batch_size=BATCH_SIZE, callbacks=[checkpoint_cb, earlystop_cb], verbose=1)

    model.save(MODEL_PATH.as_posix())
    np.savez(SCALER_PATH, feat_min=feat_min, feat_max=feat_max)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")

    return history


if __name__ == "__main__":
    train_model()
