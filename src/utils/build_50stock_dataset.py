"""
Build 50-Stock Historical Price Dataset for Expanded SPCM Model Training.

Fetches genuine historical OHLCV data for all 50 NSE stocks across 12 sectors
from Yahoo Finance, validates data integrity, and produces a reproducible,
date-aligned dataset without synthetic data or duplicate observations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import yfinance as yf

from src.utils.stock_universe import FULL_STOCK_CATALOG, SECTOR_CATALOG

logger = logging.getLogger("build_50stock_dataset")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

OUTPUT_CSV = "data/processed/india/prices_50stocks.csv"
DATA_REPORT = "reports/50stocks_data_coverage_report.txt"

# Special ticker mappings where NSE listings updated
TICKER_OVERRIDES = {
    "TATAMOTORS": "TMPV.NS",  # Tata Motors Passenger Vehicles carries the continuous historical trading series
}


def fetch_and_clean_50_stocks(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    output_path: str = OUTPUT_CSV,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Fetch and validate historical price series for all 50 stocks.
    """
    logger.info(f"Starting historical data fetch for {len(FULL_STOCK_CATALOG)} stocks ({start_date} to {end_date})...")
    
    all_rows: List[pd.DataFrame] = []
    coverage_stats: Dict[str, dict] = {}
    
    for item in FULL_STOCK_CATALOG:
        symbol = item["symbol"]
        sector = item["sector"]
        ticker = TICKER_OVERRIDES.get(symbol, item["ticker"])
        
        logger.info(f"Downloading {symbol} via {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                multi_level_index=False,
            )
            
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient history for {symbol} ({len(df)} rows).")
                coverage_stats[symbol] = {
                    "sector": sector,
                    "ticker": ticker,
                    "rows": len(df),
                    "status": "EXCLUDED_INSUFFICIENT_DATA",
                    "start": None,
                    "end": None,
                }
                continue
            
            # Standardize columns
            df = df.reset_index()
            # Standardize column names lowercase
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            
            # Ensure required columns exist
            rename_map = {"datetime": "date"}
            df = df.rename(columns=rename_map)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            
            req_cols = ["open", "high", "low", "close", "volume"]
            missing = [c for c in req_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns {missing} in download for {symbol}")
            
            # Validate non-negative prices and positive volume
            valid_mask = (
                (df["open"] > 0) &
                (df["high"] > 0) &
                (df["low"] > 0) &
                (df["close"] > 0) &
                (df["volume"] >= 0) &
                (df["high"] >= df["low"])
            )
            df = df[valid_mask].copy()
            
            # Remove any duplicate dates
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            # Add metadata
            df["stock"] = symbol
            df["sector"] = sector
            
            clean_df = df[["stock", "sector", "date", "open", "high", "low", "close", "volume"]]
            all_rows.append(clean_df)
            
            coverage_stats[symbol] = {
                "sector": sector,
                "ticker": ticker,
                "rows": len(clean_df),
                "status": "INCLUDED",
                "start": str(clean_df["date"].min().date()),
                "end": str(clean_df["date"].max().date()),
            }
            logger.info(f"  ✓ {symbol} ({sector}): {len(clean_df)} bars ({coverage_stats[symbol]['start']} to {coverage_stats[symbol]['end']})")
            
        except Exception as exc:
            logger.error(f"Failed to retrieve {symbol} ({ticker}): {exc}")
            coverage_stats[symbol] = {
                "sector": sector,
                "ticker": ticker,
                "rows": 0,
                "status": f"FAILED: {exc}",
                "start": None,
                "end": None,
            }

    if not all_rows:
        raise RuntimeError("No stock data could be fetched.")

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_df = combined_df.sort_values(["stock", "date"]).reset_index(drop=True)
    
    # Save combined prices
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_p, index=False)
    logger.info(f"Successfully saved {len(combined_df)} rows for {combined_df['stock'].nunique()} stocks to '{out_p}'.")

    # Generate Coverage Report
    rep_p = Path(DATA_REPORT)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    
    included_count = sum(1 for v in coverage_stats.values() if v["status"] == "INCLUDED")
    total_bars = sum(v["rows"] for v in coverage_stats.values() if v["status"] == "INCLUDED")
    
    report_lines = [
        "=" * 80,
        "SPCM 50-STOCK EXPANDED TRAINING DATASET COVERAGE REPORT",
        "=" * 80,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Historical Target Window: {start_date} to {end_date}",
        f"Stocks Successfully Included: {included_count} / {len(FULL_STOCK_CATALOG)} (100% Target Met: {included_count == len(FULL_STOCK_CATALOG)})",
        f"Total Daily Observations: {total_bars:,} bars",
        "",
        "PER-SECTOR BREAKDOWN:",
        "-" * 80,
    ]
    
    for sec in SECTOR_CATALOG:
        sec_name = sec["name"]
        sec_stocks = [s for s, v in coverage_stats.items() if v["sector"] == sec_name]
        sec_bars = sum(coverage_stats[s]["rows"] for s in sec_stocks)
        report_lines.append(f"Sector: {sec_name} ({len(sec_stocks)} stocks, {sec_bars:,} bars)")
        for s in sec_stocks:
            info = coverage_stats[s]
            report_lines.append(f"  - {s:12s} [{info['ticker']:14s}]: {info['rows']} bars ({info['start']} to {info['end']}) - {info['status']}")
        report_lines.append("")

    report_lines.extend([
        "=" * 80,
        "DATA INTEGRITY CHECKS:",
        f"  1. Duplicate stock-date rows: {combined_df.duplicated(['stock', 'date']).sum()}",
        f"  2. Missing/NaN values in OHLCV: {combined_df[['open', 'high', 'low', 'close', 'volume']].isna().sum().to_dict()}",
        f"  3. Inverted price bars (high < low): {(combined_df['high'] < combined_df['low']).sum()}",
        f"  4. Negative volume rows: {(combined_df['volume'] < 0).sum()}",
        "=" * 80,
    ])

    with open(rep_p, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Coverage report written to '{rep_p}'.")
    return combined_df, coverage_stats


if __name__ == "__main__":
    fetch_and_clean_50_stocks()

