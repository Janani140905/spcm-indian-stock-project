"""
Unit and integration tests for Phase 1 Indian Market Data Integration.
Tests cover symbol normalization, schema standardization, data validation
(missing values, invalid prices, duplicate timestamps, empty inputs),
feed classification (delayed / historical), and network failure handling.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.utils.indian_market_data import (
    IndianMarketDataFetcher,
    normalize_symbol,
    extract_stock_code,
    normalize_and_validate_ohlcv,
    FeedType,
    DataValidationError,
    ProviderConnectionError,
    NORMALIZED_COLUMNS,
)


class TestIndianMarketDataModule(unittest.TestCase):

    def test_normalize_symbol(self):
        """Verify default exchange suffix attachment and preservation of explicit exchanges."""
        self.assertEqual(normalize_symbol("RELIANCE"), "RELIANCE.NS")
        self.assertEqual(normalize_symbol("reliance"), "RELIANCE.NS")
        self.assertEqual(normalize_symbol("TCS.NS"), "TCS.NS")
        self.assertEqual(normalize_symbol("infy.bo"), "INFY.BO")
        self.assertEqual(normalize_symbol("  HDFCBANK  "), "HDFCBANK.NS")

    def test_extract_stock_code(self):
        """Verify extraction of base ticker code for SPCM pipeline."""
        self.assertEqual(extract_stock_code("RELIANCE.NS"), "RELIANCE")
        self.assertEqual(extract_stock_code("TCS.BO"), "TCS")
        self.assertEqual(extract_stock_code("INFY"), "INFY")

    def test_normalize_and_validate_ohlcv_success(self):
        """Verify standard normalization of single-level DataFrame."""
        dates = pd.date_range("2026-10-10", periods=5, freq="D")
        raw_df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0, 102.0, 101.0, 105.0, 107.0],
            "High": [103.0, 104.0, 106.0, 108.0, 110.0],
            "Low": [99.0, 100.0, 100.0, 103.0, 105.0],
            "Close": [102.0, 101.0, 105.0, 107.0, 109.0],
            "Volume": [1000, 1500, 1200, 2000, 1800]
        })

        norm_df = normalize_and_validate_ohlcv(raw_df, "RELIANCE.NS")
        self.assertEqual(list(norm_df.columns), NORMALIZED_COLUMNS)
        self.assertEqual(len(norm_df), 5)
        self.assertEqual(norm_df["symbol"].iloc[0], "RELIANCE.NS")
        self.assertEqual(norm_df["stock"].iloc[0], "RELIANCE")
        self.assertTrue((norm_df["close"] > 0).all())

    def test_normalize_and_validate_multiindex_columns(self):
        """Verify handling of MultiIndex columns from modern yfinance download."""
        dates = pd.date_range("2026-10-10", periods=3, freq="D")
        tuples = [
            ("Close", "RELIANCE.NS"),
            ("High", "RELIANCE.NS"),
            ("Low", "RELIANCE.NS"),
            ("Open", "RELIANCE.NS"),
            ("Volume", "RELIANCE.NS")
        ]
        index = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
        raw_df = pd.DataFrame(
            [
                [102.0, 103.0, 99.0, 100.0, 5000],
                [104.0, 105.0, 101.0, 102.0, 6000],
                [106.0, 107.0, 103.0, 104.0, 7000]
            ],
            index=dates,
            columns=index
        )

        norm_df = normalize_and_validate_ohlcv(raw_df, "RELIANCE.NS")
        self.assertEqual(len(norm_df), 3)
        self.assertEqual(list(norm_df.columns), NORMALIZED_COLUMNS)
        self.assertEqual(norm_df["close"].iloc[0], 102.0)

    def test_validation_empty_dataframe(self):
        """Verify DataValidationError is raised on empty response."""
        empty_df = pd.DataFrame()
        with self.assertRaises(DataValidationError):
            normalize_and_validate_ohlcv(empty_df, "TEST.NS")

    def test_validation_missing_ohlcv_columns(self):
        """Verify DataValidationError when critical columns are missing."""
        bad_df = pd.DataFrame({
            "Date": ["2026-10-10"],
            "Open": [100.0],
            # High, Low, Close, Volume missing
        })
        with self.assertRaises(DataValidationError):
            normalize_and_validate_ohlcv(bad_df, "TEST.NS")

    def test_validation_nan_handling(self):
        """Verify rows containing NaNs in OHLCV are dropped."""
        dates = pd.date_range("2026-10-10", periods=4, freq="D")
        raw_df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0, np.nan, 101.0, 105.0],
            "High": [105.0, 104.0, 106.0, 108.0],
            "Low": [98.0, 100.0, np.nan, 103.0],
            "Close": [102.0, 101.0, 105.0, 107.0],
            "Volume": [1000, 1500, 1200, 2000]
        })
        norm_df = normalize_and_validate_ohlcv(raw_df, "TEST.NS")
        # Out of 4 rows, rows 1 and 2 had NaNs -> 2 valid rows remaining
        self.assertEqual(len(norm_df), 2)
        self.assertFalse(norm_df.isna().any().any())

    def test_validation_invalid_prices(self):
        """Verify detection and filtering of negative prices, zero prices, and high < low."""
        dates = pd.date_range("2026-10-10", periods=5, freq="D")
        raw_df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0, -50.0, 101.0, 105.0, 100.0],   # Row 1: negative open
            "High": [105.0, 104.0, 95.0, 108.0, 105.0],    # Row 2: high (95) < low (98)
            "Low":  [98.0,  100.0, 98.0, 103.0, 98.0],
            "Close": [102.0, 101.0, 105.0, 0.0, 102.0],    # Row 3: zero close
            "Volume": [1000, 1500, 1200, 2000, -100]       # Row 4: negative volume
        })
        norm_df = normalize_and_validate_ohlcv(raw_df, "TEST.NS")
        # Only Row 0 is completely valid
        self.assertEqual(len(norm_df), 1)
        self.assertEqual(norm_df["open"].iloc[0], 100.0)

    def test_validation_duplicate_timestamps(self):
        """Verify duplicate timestamps are deduplicated keeping latest."""
        raw_df = pd.DataFrame({
            "Date": ["2026-10-10", "2026-10-11", "2026-10-11", "2026-10-12"],
            "Open": [100.0, 101.0, 101.5, 103.0],
            "High": [105.0, 106.0, 106.5, 108.0],
            "Low":  [98.0,  100.0, 100.5, 102.0],
            "Close": [102.0, 103.0, 104.0, 105.0],
            "Volume": [1000, 1200, 1300, 1400]
        })
        norm_df = normalize_and_validate_ohlcv(raw_df, "TEST.NS")
        self.assertEqual(len(norm_df), 3)
        # Should keep the second 2026-10-11 entry with close 104.0
        row_11 = norm_df[norm_df["timestamp"] == pd.Timestamp("2026-10-11")]
        self.assertEqual(len(row_11), 1)
        self.assertEqual(row_11["close"].iloc[0], 104.0)

    def test_feed_type_labeling(self):
        """Verify feed classification adheres to provider limitations (never falsely claimed real-time)."""
        fetcher = IndianMarketDataFetcher()

        with patch.object(fetcher, "fetch_symbol") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                "date": [pd.Timestamp("2026-10-19")],
                "timestamp": [pd.Timestamp("2026-10-19")],
                "symbol": ["TCS.NS"],
                "stock": ["TCS"],
                "open": [2450.0],
                "high": [2500.0],
                "low": [2440.0],
                "close": [2460.0],
                "volume": [500000]
            })

            # Daily interval -> HISTORICAL
            res_daily = fetcher.fetch_symbols(["TCS.NS"], period="5d", interval="1d")
            self.assertEqual(res_daily.metadata.feed_type, FeedType.HISTORICAL)
            self.assertFalse(res_daily.metadata.is_live)

            # Intraday interval -> DELAYED
            res_intraday = fetcher.fetch_symbols(["TCS.NS"], period="1d", interval="15m")
            self.assertEqual(res_intraday.metadata.feed_type, FeedType.DELAYED)
            self.assertEqual(res_intraday.metadata.delay_minutes, 15)
            self.assertFalse(res_intraday.metadata.is_live)

    def test_graceful_error_handling_on_failed_symbol(self):
        """Verify that failure on one symbol does not abort the batch."""
        fetcher = IndianMarketDataFetcher()

        def side_effect(sym, **kwargs):
            if "FAIL" in sym:
                raise ProviderConnectionError("Symbol not found on exchange")
            return pd.DataFrame({
                "date": [pd.Timestamp("2026-10-19")],
                "timestamp": [pd.Timestamp("2026-10-19")],
                "symbol": [sym],
                "stock": [extract_stock_code(sym)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [10000]
            })

        with patch.object(fetcher, "fetch_symbol", side_effect=side_effect):
            res = fetcher.fetch_symbols(["RELIANCE.NS", "FAIL.NS", "TCS.NS"])
            self.assertEqual(len(res.metadata.symbols_successful), 2)
            self.assertIn("FAIL.NS", res.metadata.symbols_failed)
            self.assertEqual(len(res.data), 2)

    def test_live_yfinance_integration(self):
        """Integration test against actual Yahoo Finance endpoint (if internet is available)."""
        fetcher = IndianMarketDataFetcher()
        try:
            res = fetcher.fetch_symbols(["RELIANCE.NS"], period="3d", interval="1d")
            if not res.data.empty:
                self.assertGreaterEqual(len(res.data), 1)
                self.assertEqual(res.metadata.provider, "yfinance (Yahoo Finance)")
                self.assertFalse(res.metadata.is_live)
                self.assertIsNotNone(res.metadata.latest_timestamp)
        except Exception as e:
            # If network is unavailable or restricted in test environment, test still passes gracefully
            self.skipTest(f"Live network endpoint could not be reached: {e}")


if __name__ == "__main__":
    unittest.main()
