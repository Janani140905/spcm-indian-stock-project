import yfinance as yf
import pandas as pd
import argparse
from pathlib import Path
from typing import Sequence

from src.utils.indian_market_data import (
    DEFAULT_INDIAN_SYMBOLS,
    normalize_symbol,
    extract_stock_code,
    normalize_and_validate_ohlcv,
    DataValidationError,
)

# Map internal symbols to default Indian stock universe
STOCK_UNIVERSE = [
    extract_stock_code(s) for s in DEFAULT_INDIAN_SYMBOLS
] + [
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "KOTAKBANK"
]
# Remove duplicates while preserving order
STOCK_UNIVERSE = list(dict.fromkeys(STOCK_UNIVERSE))


def fetch_prices(
    start_date: str,
    end_date: str,
    output_path: str,
    symbols: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Fetch and validate historical Indian stock prices via yfinance for SPCM pipeline.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        output_path: Path to save the resulting CSV.
        symbols: Optional list of symbols or stock tickers to fetch.

    Returns:
        pd.DataFrame containing normalized price data.
    """
    stocks_to_fetch = list(symbols) if symbols else STOCK_UNIVERSE
    print(f"Fetching data for {len(stocks_to_fetch)} stocks from {start_date} to {end_date}...")

    all_data = []
    for s in stocks_to_fetch:
        yf_sym = normalize_symbol(s)
        stock_code = extract_stock_code(s)
        print(f"Fetching {yf_sym}...")
        try:
            df = yf.download(
                yf_sym,
                start=start_date,
                end=end_date,
                progress=False,
                multi_level_index=False
            )
            if df.empty:
                print(f"[!] Warning: No data found for {yf_sym}")
                continue

            normalized = normalize_and_validate_ohlcv(df, yf_sym)
            # Retain standard schema required by SPCM data_loader.py
            all_data.append(normalized[["stock", "date", "open", "high", "low", "close", "volume"]])
        except DataValidationError as ve:
            print(f"[!] Data validation warning for {yf_sym}: {ve}")
        except Exception as e:
            print(f"[!] Failed to fetch {yf_sym}: {e}")

    if not all_data:
        print("No data fetched.")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(["stock", "date"]).reset_index(drop=True)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_file, index=False)
    print(f"Successfully wrote {len(final_df)} rows to {out_file}")
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Indian Stock Prices via yfinance")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2020-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw/india/prices.csv", help="Output CSV path")
    parser.add_argument("--symbols", nargs="+", default=None, help="Optional list of symbols (e.g. RELIANCE TCS INFY)")

    args = parser.parse_args()
    fetch_prices(args.start, args.end, args.out, symbols=args.symbols)

