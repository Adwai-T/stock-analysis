from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory

from tensorflow.keras.models import load_model

# ---------- PATHS / CONFIG ----------

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLDATA_DIR = ROOT_DIR / "data" / "allData"
MODEL_DIR = ROOT_DIR / "model"

MODEL_PATH = MODEL_DIR / "lstm_nifty200.h5"
SCALER_PATH = MODEL_DIR / "lstm_nifty200_scaler.npz"

SEQ_LEN = 30
FEAT_COLS = ["open", "high", "low", "close", "volume"]

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

# ---------- LOAD MODEL & SCALER ONCE ----------

print("Loading LSTM model...")
model = load_model(MODEL_PATH.as_posix(), compile=False)
print("Model loaded from", MODEL_PATH)

print("Loading scaler...")
scaler = np.load(SCALER_PATH)
feat_min = scaler["feat_min"]
feat_max = scaler["feat_max"]
denom = feat_max - feat_min
denom[denom == 0] = 1.0
print("Scaler loaded from", SCALER_PATH)


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


def predict_next_close(df: pd.DataFrame) -> float | None:
    """
    Given a df of last 2 years (with FEAT_COLS), predict next day's closing price.
    """
    feats = df[FEAT_COLS].values.astype("float32")
    feats_norm = (feats - feat_min) / denom

    if len(feats_norm) < SEQ_LEN:
        return None

    window = feats_norm[-SEQ_LEN:]
    x = np.expand_dims(window, axis=0)  # (1, SEQ_LEN, num_features)

    pred_norm = model.predict(x, verbose=0)[0, 0]

    # denormalize using close's min/max (index 3)
    close_min = feat_min[3]
    close_max = feat_max[3]
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


@app.route("/api/symbol/<symbol>")
def symbol_data(symbol):
    symbol = symbol.upper().strip()
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

    # Prediction
    try:
        next_close = predict_next_close(df)
    except Exception as e:
        next_close = None
        print("Prediction error for", symbol, ":", e)

    # Next date (just next calendar day after last date; not handling weekends specially)
    last_date = df["date"].max()
    next_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    prediction = {
        "next_close": next_close,
        "next_date": next_date,
    }

    return jsonify({
        "symbol": symbol,
        "data": full_records,
        "last20": last20_records,
        "today": today_rec,
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)
