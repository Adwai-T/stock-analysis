import os
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, GlobalAveragePooling1D, Embedding
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
EPOCHS = 20

MODEL_PATH = MODEL_DIR / "transformer_nifty200_feature.h5"
SCALER_PATH = MODEL_DIR / "transformer_nifty200_scaler_feature.npz"


# ---------- FEATURE ENGINEERING ----------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

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
            print("Download failed — using fallback")
            return []

    return [s.strip().upper() for s in df[df["Series"] == "EQ"]["Symbol"]]


def load_stock_df(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=365 * YEARS)].reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return add_features(df)


def build_dataset():
    print(f"Scanning {ALLDATA_DIR}...")
    all_files = sorted(p for p in ALLDATA_DIR.glob("*.csv"))
    symbols = set(get_nifty200_symbols())

    stock_data, all_feats = {}, []

    for sym_file in all_files:
        sym = sym_file.stem.upper()
        if symbols and sym not in symbols:
            continue
        df = load_stock_df(sym_file)
        if len(df) <= SEQ_LEN + 1:
            continue
        feature_cols = [c for c in df.columns if c != "date"]
        arr = df[feature_cols].values.astype("float32")
        stock_data[sym] = (df, feature_cols)
        all_feats.append(arr)

    all_arr = np.vstack(all_feats)
    feat_min, feat_max = all_arr.min(axis=0), all_arr.max(axis=0)
    denom = np.where((feat_max - feat_min) == 0, 1, feat_max - feat_min)

    X_list, y_list = [], []

    for sym, (df, feature_cols) in stock_data.items():
        arr = df[feature_cols].values.astype("float32")
        arr_norm = (arr - feat_min) / denom
        close_idx = feature_cols.index("close")

        for i in range(len(arr_norm) - SEQ_LEN - 1):
            X_list.append(arr_norm[i:i + SEQ_LEN])
            y_list.append(arr_norm[i + SEQ_LEN, close_idx])

    return np.stack(X_list), np.array(y_list), feat_min, feat_max


# ---------- TRANSFORMER MODEL ----------

def build_transformer(seq_len, num_features):
    d_model = 64
    num_heads = 4
    ff_dim = 128
    dropout = 0.1

    inp = Input(shape=(seq_len, num_features))

    # 1) Linear projection
    x = Dense(d_model)(inp)

    # 2) Positional embedding
    positions = tf.range(seq_len)
    pos_emb = Embedding(input_dim=seq_len, output_dim=d_model)(positions)
    x = x + pos_emb

    # ---- Transformer Encoder Block ----
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)

    # Pool & output
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)

    return Model(inputs=inp, outputs=out)


def train_model():
    X, y, feat_min, feat_max = build_dataset()
    num_features = X.shape[-1]

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_transformer(SEQ_LEN, num_features)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_PATH.as_posix(), save_best_only=True, monitor="val_loss", verbose=1)
    earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    print("\n🚀 Training Transformer Model...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    model.save(MODEL_PATH.as_posix())
    np.savez(SCALER_PATH, feat_min=feat_min, feat_max=feat_max)

    print("\nModel saved:", MODEL_PATH)
    print("Scaler saved:", SCALER_PATH)

    return history


if __name__ == "__main__":
    train_model()
