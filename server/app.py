from pathlib import Path
from datetime import timedelta

import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, request
from kiteconnect import KiteConnect

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense

# ---------- PATHS / CONFIG ----------

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLDATA_DIR = ROOT_DIR / "data" / "allData"
MODEL_DIR = ROOT_DIR / "model"

# Load Kite credentials
CREDS_PATH = ROOT_DIR / "credentials.json"
with open(CREDS_PATH, "r") as f:
    c = json.load(f)

kite = KiteConnect(api_key=c["API_KEY"])
kite.set_access_token(c["ACCESS_TOKEN"])

SEQ_LEN = 30
FEAT_COLS = ["open", "high", "low", "close", "volume"]

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

MODEL_DIR = ROOT_DIR / "model"

SEQ_LEN = 30
FEAT_COLS = ["open", "high", "low", "close", "volume"]

# Cache: model_name -> (model, feat_min, feat_max, denom)
_model_cache = {}

# ---------- HELPERS ----------

def load_symbol_df(symbol: str) -> pd.DataFrame:
    """
    Load CSV like data/allData/CIPLA.csv
    Columns: date,open,high,low,close,volume,empty
    Return last 2 years of data sorted by date.
    """
    path = ALLDATA_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # last 2 years per symbol
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=365 * 2)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    # keep only needed columns
    df = df[["date", "open", "high", "low", "close", "volume"]]

    if len(df) < SEQ_LEN + 1:
        raise ValueError(f"Not enough data for {symbol} after filtering")

    return df


def predict_next_close(df: pd.DataFrame, model_name: str) -> float | None:
    """
    Predict next close using global model.
    - If model uses 5 features -> use raw OHLCV (FEAT_COLS)
    - If model uses more (e.g. 19) -> recompute engineered features and
      use all numeric columns except date.
    """
    model, feat_min, feat_max, denom = get_model_bundle(model_name)

    # Decide if this is a "simple" or "feature" model based on feature count
    num_features = len(feat_min)

    if num_features == len(FEAT_COLS):
        # Old/simple model: just OHLCV
        working_df = df.copy()
        feature_cols = FEAT_COLS
    else:
        # Feature model: recompute engineered features
        working_df = add_features(df)
        feature_cols = [c for c in working_df.columns if c != "date"]

        if len(feature_cols) != num_features:
            raise ValueError(
                f"Model expects {num_features} features, "
                f"but engineered df has {len(feature_cols)}"
            )

    feats = working_df[feature_cols].values.astype("float32")
    feats_norm = (feats - feat_min) / denom

    if len(feats_norm) < SEQ_LEN:
        return None

    window = feats_norm[-SEQ_LEN:]
    x = np.expand_dims(window, axis=0)

    # index of 'close' among feature columns
    try:
        close_idx = feature_cols.index("close")
    except ValueError:
        raise RuntimeError("'close' not found in feature columns")

    pred_norm = model.predict(x, verbose=0)[0, 0]

    close_min = feat_min[close_idx]
    close_max = feat_max[close_idx]
    close_denom = close_max - close_min if close_max > close_min else 1.0
    pred_close = pred_norm * close_denom + close_min

    return float(pred_close)

def df_to_records(df: pd.DataFrame):
    """Convert df to list of dicts with date as YYYY-MM-DD strings."""
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


# ---------- ROUTES ----------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

def get_model_bundle(model_name: str):
    """
    Load model + scaler for the given model_name (with caching).
    Expects files:
      ./model/<model_name>.h5
      ./model/<model_name>_scaler.npz
    """
    model_name = model_name.strip()
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODEL_DIR / f"{model_name}.h5"
    scaler_path = MODEL_DIR / f"{model_name}_scaler.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    print(f"Loading model: {model_path}")
    model = load_model(model_path.as_posix(), compile=False)

    print(f"Loading scaler: {scaler_path}")
    scaler = np.load(scaler_path)
    feat_min = scaler["feat_min"]
    feat_max = scaler["feat_max"]
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0

    bundle = (model, feat_min, feat_max, denom)
    _model_cache[model_name] = bundle
    return bundle


@app.route("/api/symbol/<symbol>")
def symbol_data(symbol):
    symbol = symbol.upper().strip()
    model_name = request.args.get("model", "lstm_nifty200")  # default model
    
    print("DEBUG_REQUEST_ARGS:", dict(request.args))
    print("DEBUG_MODEL:", model_name)
    
    # 1) Update local CSV using kite
    try:
        update_symbol_from_kite(symbol)
    except Exception as e:
        print("Kite update failed:", e)
        # allow fallback to local file even if update fails
    
    try:
        df = load_symbol_df(symbol)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to load data for {symbol}: {e}"}), 500

    # Full history
    full_records = df_to_records(df)

    # Last 20 days
    last20_df = df.tail(20)
    last20_records = df_to_records(last20_df)

    # "Today" = last available row in file
    today_row = df.iloc[-1:].copy()
    today_rec = df_to_records(today_row)[0]

    # Global model prediction
    try:
        next_close = predict_next_close(df, model_name)
        global_error = None
    except Exception as e:
        next_close = None
        global_error = str(e)
        print(f"Prediction error for {symbol} with model {model_name}:", e)
        
    # Per-symbol real-time LSTM + GRU training
    per_lstm = None
    per_gru = None
    per_error = None
    
    # Test-set predictions for last ~3 months using selected model
    test_points = []
    try:
        test_points = build_test_predictions(df, model_name)
    except Exception as e:
        print(f"Test-set prediction error for {symbol} with model {model_name}:", e)
    
    try:
        per_lstm, per_gru = train_per_symbol_models(df)
    except Exception as e:
        per_error = str(e)
        print(f"Per-symbol training error for {symbol}:", e)
        
    # Next date (just next calendar day after last date; not handling weekends specially)
    last_date = df["date"].max()
    next_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # Next date ...
    prediction = {
        "next_close": next_close,
        "next_date": next_date,
        "model": model_name,
        "per_symbol_lstm": per_lstm,
        "per_symbol_gru": per_gru,
        "global_error": global_error,
        "per_symbol_error": per_error,
    }
    
    return jsonify({
        "symbol": symbol,
        "data": full_records,
        "last20": last20_records,
        "today": today_rec,
        "prediction": prediction,
        "test_set": {
            "points": test_points
        }
    })
    
def update_symbol_from_kite(symbol: str):
    """
    Fetch last 2 years daily data from Kite for the symbol
    and overwrite data/allData/<symbol>.csv
    """
    symbol = symbol.upper()
    out_path = ALLDATA_DIR / f"{symbol}.csv"

    # Find token in NSE instruments
    instruments = kite.instruments("NSE")
    token = None
    for inst in instruments:
        if (
            inst["tradingsymbol"].upper() == symbol
            and inst["segment"] == "NSE"
            and inst["instrument_type"] == "EQ"
        ):
            token = inst["instrument_token"]
            break

    if token is None:
        raise ValueError(f"Symbol {symbol} not found in kite instruments")

    # Fetch 2-year history
    end_d = pd.Timestamp.now().date()
    start_d = end_d - pd.Timedelta(days=365 * 2)

    candles = kite.historical_data(
        instrument_token=token,
        from_date=start_d,
        to_date=end_d,
        interval="day",
        oi=True,
    )

    if not candles:
        raise RuntimeError(f"No data returned for {symbol}")

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Updated file: {out_path}")

def train_per_symbol_models(df: pd.DataFrame, epochs: int = 5, batch_size: int = 32):
    """
    Train a small LSTM and GRU on this symbol's last 2 years of data only,
    then return their next-day closing price predictions (denormalized).
    """
    feats = df[FEAT_COLS].values.astype("float32")

    # Need enough rows to build sequences
    n = len(feats)
    if n < SEQ_LEN + 5:  # a bit more forgiving
        raise ValueError(f"Not enough data for per-symbol training (n={n}, need>{SEQ_LEN+5})")

    # Per-symbol min/max scaling
    feat_min = feats.min(axis=0)
    feat_max = feats.max(axis=0)
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0
    feats_norm = (feats - feat_min) / denom

    X_list, y_list = [], []
    close_idx = 3  # open, high, low, close, volume

    for i in range(n - SEQ_LEN - 1):
        window = feats_norm[i:i + SEQ_LEN]
        target = feats_norm[i + SEQ_LEN, close_idx]
        X_list.append(window)
        y_list.append(target)

    X = np.stack(X_list)
    y = np.array(y_list, dtype="float32")

    # Simple time-based split
    split = max(int(len(X) * 0.8), 1)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    num_features = X.shape[-1]

    def make_lstm():
        m = Sequential([
            LSTM(32, input_shape=(SEQ_LEN, num_features)),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def make_gru():
        m = Sequential([
            GRU(32, input_shape=(SEQ_LEN, num_features)),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    # Train LSTM
    lstm_model = make_lstm()
    lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # change to 1 if you want console output
    )

    # Train GRU
    gru_model = make_gru()
    gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Prepare last window for prediction
    last_window = feats_norm[-SEQ_LEN:]
    x_pred = np.expand_dims(last_window, axis=0)

    lstm_pred_norm = lstm_model.predict(x_pred, verbose=0)[0, 0]
    gru_pred_norm = gru_model.predict(x_pred, verbose=0)[0, 0]

    # Denormalize using this symbol's close min/max
    close_min = feat_min[close_idx]
    close_max = feat_max[close_idx]
    close_denom = close_max - close_min if close_max > close_min else 1.0

    lstm_pred = lstm_pred_norm * close_denom + close_min
    gru_pred = gru_pred_norm * close_denom + close_min

    return float(lstm_pred), float(gru_pred)

def build_test_predictions(df: pd.DataFrame, model_name: str):
    """
    Use the selected model to predict close for the last ~3 months.
    Handles both simple (OHLCV) and feature-based models.
    """
    model, feat_min, feat_max, denom = get_model_bundle(model_name)
    num_features = len(feat_min)

    # Decide which features to use (simple vs feature model)
    if num_features == len(FEAT_COLS):
        # Simple model: OHLCV only
        working_df = df.copy()
        feature_cols = FEAT_COLS
    else:
        # Feature model: recompute engineered features
        working_df = add_features(df)
        feature_cols = [c for c in working_df.columns if c != "date"]
        if len(feature_cols) != num_features:
            raise ValueError(
                f"Model expects {num_features} features, "
                f"but engineered df has {len(feature_cols)}"
            )

    feats = working_df[feature_cols].values.astype("float32")
    # ❗ THIS WAS MISSING BEFORE — scale features using training scaler
    feats_norm = (feats - feat_min) / denom

    dates = working_df["date"].reset_index(drop=True)
    closes = working_df["close"].reset_index(drop=True)

    if len(feats_norm) < SEQ_LEN + 5:
        return []

    max_date = dates.max()
    cutoff = max_date - pd.Timedelta(days=90)  # ~3 months

    try:
        close_idx = feature_cols.index("close")
    except ValueError:
        raise RuntimeError("'close' not found in feature columns")

    records = []
    n = len(feats_norm)

    for i in range(n - SEQ_LEN - 1):
        target_idx = i + SEQ_LEN
        if dates.iloc[target_idx] < cutoff:
            continue

        window = feats_norm[i:i + SEQ_LEN]
        x = np.expand_dims(window, axis=0)

        pred_norm = model.predict(x, verbose=0)[0, 0]

        # Denormalize using global close min/max from training
        close_min = feat_min[close_idx]
        close_max = feat_max[close_idx]
        close_denom = close_max - close_min if close_max > close_min else 1.0
        pred_close = pred_norm * close_denom + close_min

        records.append({
            "date": dates.iloc[target_idx].strftime("%Y-%m-%d"),
            "actual": float(closes.iloc[target_idx]),
            "predicted": float(pred_close),
        })

    return records

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same feature engineering used in training.
    Input df must have: date, open, high, low, close, volume.
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

    # ---- RSI (14) ----
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ---- ATR (14) ----
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(),
                               (df["low"] - prev_close).abs()))
    df["tr"] = tr
    df["atr_14"] = df["tr"].rolling(14).mean()

    # ---- Volatility (20) ----
    df["volatility_20"] = df["log_return"].rolling(20).std()

    # ---- Lag features ----
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_5"] = df["close"].shift(5)

    # Drop rows with NaNs introduced by rolling/shift
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    app.run(debug=True)
