import os
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense       # <--- Changed import from LSTM to GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------- PATHS & CONFIG ----------

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLDATA_DIR = ROOT_DIR / "data" / "allData"
META_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
NIFTY50_LOCAL_PATH = META_DIR / "nifty50_list.csv"

SEQ_LEN = 30
YEARS = 2
BATCH_SIZE = 64
EPOCHS = 20

MODEL_PATH = MODEL_DIR / "gru_nifty50.h5"      # <--- Changed filename to avoid overwrite
SCALER_PATH = MODEL_DIR / "gru_nifty50_scaler.npz"


# ---------- HELPERS ---------- (unchanged) ----------

def get_nifty50_symbols():
    if NIFTY50_LOCAL_PATH.exists():
        df = pd.read_csv(NIFTY50_LOCAL_PATH)
    else:
        print("Downloading NIFTY 50 list...")
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com"}
            resp = requests.get(NIFTY50_URL, headers=headers, timeout=20)
            resp.raise_for_status()
            csv_text = resp.content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
            META_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(NIFTY50_LOCAL_PATH, index=False)
        except Exception:
            print("Download failed - using all files found.")
            return []
    df = df[df["Series"] == "EQ"]
    return [s.strip().upper() for s in df["Symbol"].tolist()]


def load_stock_df(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    cutoff = df["date"].max() - pd.Timedelta(days=365 * YEARS)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def build_dataset():
    print(f"Scanning {ALLDATA_DIR} for CSV files...")
    all_files = sorted(p for p in ALLDATA_DIR.glob("*.csv"))
    if not all_files:
        raise RuntimeError("No CSV files found in data directory.")

    nifty50_syms = set(get_nifty50_symbols())
    selected_files = [(f.stem.upper(), f) for f in all_files if not nifty50_syms or f.stem.upper() in nifty50_syms]

    all_features_list = []
    stock_data = {}
    for symbol, f in selected_files:
        df = load_stock_df(f)
        if len(df) <= SEQ_LEN + 1:
            continue
        feats = df[["open", "high", "low", "close", "volume"]].values.astype("float32")
        stock_data[symbol] = df
        all_features_list.append(feats)

    if not all_features_list:
        raise RuntimeError("No usable stock data found.")

    all_features = np.vstack(all_features_list)
    feat_min = all_features.min(axis=0)
    feat_max = all_features.max(axis=0)
    denom = np.where((feat_max - feat_min) == 0, 1.0, feat_max - feat_min)

    X_list, y_list = [], []
    close_idx = 3

    for symbol, df in stock_data.items():
        feats = df[["open", "high", "low", "close", "volume"]].values.astype("float32")
        feats_norm = (feats - feat_min) / denom

        for i in range(len(feats_norm) - SEQ_LEN - 1):
            X_list.append(feats_norm[i:i + SEQ_LEN])
            y_list.append(feats_norm[i + SEQ_LEN, close_idx])

    X = np.stack(X_list)
    y = np.array(y_list, dtype="float32")

    return X, y, feat_min, feat_max


# ---------- TRAINING (GRU MODEL) ----------

def train_model():
    X, y, feat_min, feat_max = build_dataset()
    num_features = X.shape[-1]

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        GRU(64, input_shape=(SEQ_LEN, num_features)),   # <--- GRU layer
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    checkpoint_cb = ModelCheckpoint(MODEL_PATH.as_posix(), monitor="val_loss", save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )

    model.save(MODEL_PATH.as_posix())
    np.savez(SCALER_PATH, feat_min=feat_min, feat_max=feat_max)

    return history


if __name__ == "__main__":
    train_model()
