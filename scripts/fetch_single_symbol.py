import sys
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect

ROOT_DIR = Path(__file__).resolve().parent.parent
CREDS_PATH = ROOT_DIR / "credentials.json"
DATA_DIR = ROOT_DIR / "data"

INTERVAL = "day"
YEARS = 2


def load_kite():
    with open(CREDS_PATH, "r") as f:
        c = json.load(f)

    kite = KiteConnect(api_key=c["API_KEY"])
    kite.set_access_token(c["ACCESS_TOKEN"])
    return kite


def get_token_for_symbol(kite, symbol: str) -> int:
    instruments = kite.instruments("NSE")
    for inst in instruments:
        if (
            inst["tradingsymbol"] == symbol
            and inst["segment"] == "NSE"
            and inst["instrument_type"] == "EQ"
        ):
            return inst["instrument_token"]
    raise ValueError(f"Symbol {symbol} not found in NSE EQ instruments")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_single_symbol.py RELIANCE")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{symbol}.csv"

    kite = load_kite()
    token = get_token_for_symbol(kite, symbol)

    end_d = date.today()
    start_d = end_d - timedelta(days=365 * YEARS)
    print(f"Fetching {symbol} from {start_d} to {end_d}")

    candles = kite.historical_data(
        instrument_token=token,
        from_date=start_d,
        to_date=end_d,
        interval=INTERVAL,
        oi=True,
    )

    if not candles:
        print("No data returned.")
        return

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
