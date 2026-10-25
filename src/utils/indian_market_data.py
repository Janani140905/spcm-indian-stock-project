"""
Indian Market Data Fetcher & Normalizer for SPCM Pipeline (Phase 1).

Retrieves actual, recent Indian stock market data (NSE/BSE) via Yahoo Finance (yfinance),
normalizes the records to consistent schema and timezone, strictly validates against
missing/invalid prices and duplicate timestamps, and clearly labels the data feed classification
(delayed / historical).

NOTE ON FEED STATUS:
Yahoo Finance provides DELAYED quotes for Indian exchanges (NSE & BSE), typically with a
15-minute delay during market hours. Daily intervals represent end-of-day HISTORICAL data.
Periodic polling of this API does NOT constitute a real-time live feed.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

# Configure logger
logger = logging.getLogger("indian_market_data")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default Indian Stock Universe (Nifty 50 bellwethers)
DEFAULT_INDIAN_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

# Standard normalized columns expected by SPCM pipeline
NORMALIZED_COLUMNS = [
    "date",
    "timestamp",
    "symbol",
    "stock",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class FeedType(str, Enum):
    LIVE = "live"
    DELAYED = "delayed"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class MarketDataError(Exception):
    """Base exception for market data operations."""
    pass


class DataValidationError(MarketDataError):
    """Raised when market data fails validation checks."""
    pass


class ProviderConnectionError(MarketDataError):
    """Raised when the data provider cannot be reached."""
    pass


@dataclass
class MarketDataMetadata:
    """Metadata detailing provider source, feed type, and delivery characteristics."""
    provider: str = "yfinance (Yahoo Finance)"
    feed_type: FeedType = FeedType.DELAYED
    delay_minutes: int = 15
    is_live: bool = False
    market_session: str = "NSE/BSE (India)"
    latest_timestamp: pd.Timestamp | None = None
    symbols_requested: list[str] = field(default_factory=list)
    symbols_successful: list[str] = field(default_factory=list)
    symbols_failed: dict[str, str] = field(default_factory=dict)
    disclaimer: str = (
        "Yahoo Finance provides delayed market data (typically ~15 min delay for NSE/BSE) "
        "during trading hours, and end-of-day historical prices outside market hours. "
        "Periodic polling of this feed does NOT make it a real-time feed."
    )


@dataclass
class MarketDataResult:
    """Encapsulates the normalized DataFrame and associated metadata."""
    data: pd.DataFrame
    metadata: MarketDataMetadata


def normalize_symbol(symbol: str, default_exchange: str = "NS") -> str:
    """
    Ensure the symbol includes a recognized exchange suffix.
    e.g., 'RELIANCE' -> 'RELIANCE.NS', 'TCS.NS' -> 'TCS.NS', 'INFY.BO' -> 'INFY.BO'
    """
    sym = symbol.strip().upper()
    if "." in sym:
        return sym
    return f"{sym}.{default_exchange.upper()}"


def extract_stock_code(symbol: str) -> str:
    """
    Extract the base ticker symbol without exchange suffix.
    e.g., 'RELIANCE.NS' -> 'RELIANCE', 'TCS.BO' -> 'TCS'
    """
    return symbol.split(".")[0].upper()


def normalize_and_validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    strict_price_bounds: bool = True
) -> pd.DataFrame:
    """
    Normalizes OHLCV DataFrame and performs rigorous data validation:
      1. Validates non-empty response.
      2. Flattens MultiIndex columns if present.
      3. Normalizes timestamp to Indian Standard Time (IST) naive Timestamp.
      4. Standardizes column names and adds 'symbol' and 'stock'.
      5. Validates and drops missing values (NaNs in OHLCV).
      6. Validates price sanity (positive prices, high >= low, high >= open/close, low <= open/close, volume >= 0).
      7. Validates and eliminates duplicate timestamps.

    Returns:
        pd.DataFrame with columns: date, timestamp, symbol, stock, open, high, low, close, volume
    """
    if df is None or df.empty:
        raise DataValidationError(f"Empty data received for symbol '{symbol}'.")

    clean_df = df.copy()

    # Handle MultiIndex columns (from yfinance download)
    if isinstance(clean_df.columns, pd.MultiIndex):
        # Extract first level which represents OHLCV names
        clean_df.columns = [c[0] if isinstance(c, tuple) else c for c in clean_df.columns]

    # If datetime is in index, reset it
    if clean_df.index.name in ["Date", "Datetime", "date", "datetime"] or isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df = clean_df.reset_index()

    # Lowercase column names
    clean_df.columns = [str(c).strip().lower() for c in clean_df.columns]

    # Map datetime column to 'timestamp'
    time_col = None
    for candidate in ["datetime", "date", "timestamp"]:
        if candidate in clean_df.columns:
            time_col = candidate
            break

    if time_col is None:
        # Check if 'index' column exists and is datetime
        if "index" in clean_df.columns and pd.api.types.is_datetime64_any_dtype(clean_df["index"]):
            time_col = "index"

    if time_col is None:
        raise DataValidationError(f"Missing date/timestamp column in data for '{symbol}'. Columns found: {clean_df.columns.tolist()}")

    # Normalize timestamp/date
    ts_series = pd.to_datetime(clean_df[time_col], errors="coerce")
    if ts_series.isna().all():
        raise DataValidationError(f"Unable to parse dates in column '{time_col}' for symbol '{symbol}'.")

    # Timezone conversion to IST (Asia/Kolkata) then tz-naive for consistent pandas operations
    if ts_series.dt.tz is not None:
        ts_series = ts_series.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    clean_df["timestamp"] = ts_series
    clean_df["date"] = ts_series.dt.normalize() if (ts_series.dt.hour == 0).all() else ts_series

    # Check for required OHLCV columns
    required_ohlcv = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_ohlcv if c not in clean_df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required OHLCV columns {missing_cols} for symbol '{symbol}'.")

    # Add symbol tracking
    clean_df["symbol"] = normalize_symbol(symbol)
    clean_df["stock"] = extract_stock_code(symbol)

    # Coerce OHLCV to float
    for col in required_ohlcv:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Check for missing values (NaNs) in OHLCV
    nan_mask = clean_df[required_ohlcv].isna().any(axis=1)
    if nan_mask.any():
        dropped_count = nan_mask.sum()
        logger.warning(f"[{symbol}] Dropping {dropped_count} rows with NaN values in OHLCV.")
        clean_df = clean_df[~nan_mask].copy()

    if clean_df.empty:
        raise DataValidationError(f"All records for '{symbol}' contained NaNs or missing values.")

    # Validate price sanity
    invalid_mask = (
        (clean_df["open"] <= 0) |
        (clean_df["high"] <= 0) |
        (clean_df["low"] <= 0) |
        (clean_df["close"] <= 0) |
        (clean_df["volume"] < 0) |
        (clean_df["high"] < clean_df["low"])
    )

    if strict_price_bounds:
        # High must be >= open and close (within 0.01% floating tolerance)
        # Low must be <= open and close (within 0.01% floating tolerance)
        tol = 1e-4
        invalid_bounds = (
            (clean_df["high"] < clean_df["open"] * (1 - tol)) |
            (clean_df["high"] < clean_df["close"] * (1 - tol)) |
            (clean_df["low"] > clean_df["open"] * (1 + tol)) |
            (clean_df["low"] > clean_df["close"] * (1 + tol))
        )
        invalid_mask = invalid_mask | invalid_bounds

    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        logger.warning(f"[{symbol}] Dropping {invalid_count} rows with invalid price bounds (e.g. non-positive, high < low).")
        clean_df = clean_df[~invalid_mask].copy()

    if clean_df.empty:
        raise DataValidationError(f"All records for '{symbol}' had invalid price values.")

    # Deduplicate timestamps keeping latest
    initial_len = len(clean_df)
    clean_df = clean_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    dup_count = initial_len - len(clean_df)
    if dup_count > 0:
        logger.info(f"[{symbol}] Removed {dup_count} duplicate timestamp records.")

    clean_df = clean_df.reset_index(drop=True)
    return clean_df[NORMALIZED_COLUMNS]


class IndianMarketDataFetcher:
    """
    Reusable market data fetcher for Indian equities.
    Integrates with Yahoo Finance via yfinance, providing normalized OHLCV data
    with schema validation, exchange-aware symbol handling, and feed metadata.
    """

    def __init__(self, default_exchange: str = "NS", timeout: float = 15.0):
        self.default_exchange = default_exchange.upper()
        self.timeout = timeout

    def fetch_symbol(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch and normalize OHLCV data for a single Indian stock symbol.

        Args:
            symbol: Ticker symbol (e.g., 'RELIANCE.NS' or 'RELIANCE').
            period: Data period to download (e.g. '5d', '1mo', '3mo', '1y').
            interval: Bar interval (e.g. '1d', '5m', '15m', '60m').

        Returns:
            Normalized pd.DataFrame with columns: date, timestamp, symbol, stock, open, high, low, close, volume.

        Raises:
            DataValidationError: If data is empty or fails validation.
            ProviderConnectionError: If network or provider fails.
        """
        formatted_sym = normalize_symbol(symbol, self.default_exchange)
        try:
            ticker = yf.Ticker(formatted_sym)
            raw_df = ticker.history(period=period, interval=interval, timeout=self.timeout)
        except Exception as exc:
            msg = f"Network or provider error fetching '{formatted_sym}': {exc}"
            logger.error(msg)
            raise ProviderConnectionError(msg) from exc

        if raw_df is None or raw_df.empty:
            # Fallback to yf.download in case ticker.history hit a transient endpoint issue
            try:
                raw_df = yf.download(
                    formatted_sym,
                    period=period,
                    interval=interval,
                    progress=False,
                    multi_level_index=False,
                    timeout=self.timeout
                )
            except Exception as exc:
                msg = f"Fallback download failed for '{formatted_sym}': {exc}"
                logger.error(msg)
                raise ProviderConnectionError(msg) from exc

        return normalize_and_validate_ohlcv(raw_df, formatted_sym)

    def fetch_symbols(
        self,
        symbols: Sequence[str] | str,
        period: str = "5d",
        interval: str = "1d"
    ) -> MarketDataResult:
        """
        Fetch and normalize OHLCV data for multiple configurable Indian stock symbols.

        Args:
            symbols: Sequence of ticker symbols or comma/space-delimited string.
            period: Data period to download (default '5d').
            interval: Bar interval (default '1d').

        Returns:
            MarketDataResult containing combined normalized DataFrame and delivery metadata.
        """
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.replace(",", " ").split() if s.strip()]

        symbols_list = list(symbols) if symbols else list(DEFAULT_INDIAN_SYMBOLS)
        requested_formatted = [normalize_symbol(s, self.default_exchange) for s in symbols_list]

        # Determine feed classification
        if interval.lower() in ["1d", "1wk", "1mo"]:
            feed_type = FeedType.HISTORICAL
            delay_minutes = 0
        else:
            feed_type = FeedType.DELAYED
            delay_minutes = 15

        metadata = MarketDataMetadata(
            provider="yfinance (Yahoo Finance)",
            feed_type=feed_type,
            delay_minutes=delay_minutes,
            is_live=False,  # Yahoo Finance does NOT provide real-time tick/socket streaming for NSE/BSE
            symbols_requested=requested_formatted,
            symbols_successful=[],
            symbols_failed={},
        )

        frames: list[pd.DataFrame] = []

        for sym in requested_formatted:
            try:
                df_sym = self.fetch_symbol(sym, period=period, interval=interval)
                frames.append(df_sym)
                metadata.symbols_successful.append(sym)
                logger.info(f"Successfully fetched {len(df_sym)} records for {sym}")
            except Exception as e:
                err_msg = str(e)
                metadata.symbols_failed[sym] = err_msg
                logger.warning(f"Failed to fetch data for {sym}: {err_msg}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values(["stock", "timestamp"]).reset_index(drop=True)
            metadata.latest_timestamp = combined["timestamp"].max()
        else:
            combined = pd.DataFrame(columns=NORMALIZED_COLUMNS)

        return MarketDataResult(data=combined, metadata=metadata)


def cli_test_command():
    """Command-line interface to fetch and inspect recent Indian stock market data."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Fetch and normalize recent Indian stock market data (NSE/BSE) for SPCM pipeline."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        help="List of Indian stock symbols to fetch (e.g. RELIANCE.NS TCS.NS INFY.NS)"
    )
    parser.add_argument(
        "--period",
        default="5d",
        help="Data period to fetch (e.g. 5d, 1mo, 3mo)"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Bar interval (e.g. 1d, 5m, 15m)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save normalized records as CSV"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" INDIAN STOCK MARKET DATA INTEGRATION - PHASE 1")
    print("=" * 70)

    fetcher = IndianMarketDataFetcher()
    result = fetcher.fetch_symbols(args.symbols, period=args.period, interval=args.interval)
    meta = result.metadata

    print("\n[DATA SOURCE & FEED INFORMATION]")
    print(f"  Provider:           {meta.provider}")
    print(f"  Market:             {meta.market_session}")
    print(f"  Feed Classification: {meta.feed_type.value.upper()}")
    print(f"  Delay:              {meta.delay_minutes} minutes" if meta.delay_minutes > 0 else "  Delay:              End-of-day / Historical")
    print(f"  Is Real-Time Live:  {meta.is_live} (Periodic polling does NOT make this a real-time feed)")
    print(f"  Disclaimer:         {meta.disclaimer}")
    print(f"  Latest Timestamp:   {meta.latest_timestamp}")
    print(f"  Requested Symbols:  {len(meta.symbols_requested)} {meta.symbols_requested}")
    print(f"  Successful:         {len(meta.symbols_successful)} {meta.symbols_successful}")

    if meta.symbols_failed:
        print(f"  Failed Symbols:     {meta.symbols_failed}")

    df = result.data
    print("\n[FETCHED RECORDS SUMMARY]")
    if df.empty:
        print("  [!] Warning: No records retrieved or all symbols failed.")
    else:
        print(f"  Total Rows:         {len(df)}")
        print("\n  Per-Symbol Breakdown:")
        for sym, grp in df.groupby("symbol"):
            latest_row = grp.iloc[-1]
            print(f"    - {sym:<14}: {len(grp)} bars | Latest: {latest_row['timestamp']} | Close: {latest_row['close']:.2f} | Volume: {int(latest_row['volume']):,}")

        print("\n  Sample Recent Records (Tail):")
        cols_to_print = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        print(df[cols_to_print].tail(6).to_string(index=False))

    if args.out and not df.empty:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n[OUTPUT] Successfully exported {len(df)} normalized records to {out_path}")

    print("=" * 70)


if __name__ == "__main__":
    cli_test_command()
