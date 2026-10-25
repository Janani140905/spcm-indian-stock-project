import yfinance as yf
import pandas as pd
import argparse
from pathlib import Path

from src.utils.data_loader import TARGET_STOCKS, normalize_indian_symbol

STOCK_UNIVERSE = TARGET_STOCKS

def fetch_prices(start_date: str, end_date: str, output_path: str):
    print(f"Fetching data for {len(STOCK_UNIVERSE)} stocks from {start_date} to {end_date}...")
    
    all_data = []
    for internal_sym in STOCK_UNIVERSE:
        yf_sym = f"{internal_sym}.NS"
        print(f"Fetching {yf_sym}...")
        try:
            # yfinance returns adjusted prices in 'Adj Close'
            df = yf.download(yf_sym, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"[!] Warning: No data found for {yf_sym}")
                continue
            
            df = df.reset_index()
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            # yfinance uses 'Date' instead of 'date'
            if 'date' not in df.columns and 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'date'})
                
            df["stock"] = internal_sym
            
            # Use unadjusted OHLC for base features, but typically Adj Close is preferred for ML returns.
            # We map standard columns required by data_loader.py
            required_cols = {"date", "open", "high", "low", "close", "volume"}
            if not required_cols.issubset(set(df.columns)):
                print(f"[!] Missing required columns for {yf_sym}. Found: {df.columns.tolist()}")
                continue
                
            all_data.append(df[["stock", "date", "open", "high", "low", "close", "volume"]])
        except Exception as e:
            print(f"[!] Failed to fetch {yf_sym}: {e}")
            
    if not all_data:
        print("No data fetched.")
        return
        
    final_df = pd.concat(all_data, ignore_index=True)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_file, index=False)
    print(f"Successfully wrote {len(final_df)} rows to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Indian Stock Prices via yfinance")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2020-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw/india/prices.csv", help="Output CSV path")
    
    args = parser.parse_args()
    fetch_prices(args.start, args.end, args.out)
