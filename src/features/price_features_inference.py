"""
Inference-Time Price Feature Generation for Indian SPCM Integration (Phase 2).

Provides point-in-time, leak-free generation of price features for live/delayed
Indian stock market data (NSE/BSE) using existing training definitions and
transformation functions from `src/models/spcm.py`.

Key Safeguards:
1. Reuses exact training functions (`build_price_features`, `PRICE_LAGS`).
2. Requires sufficient historical warm-up data (>= 32 trading days for 30-day lags).
3. Extracts the latest valid feature row per symbol at its observation timestamp.
4. Strictly NEVER calculates forward-return labels (`fwd_ret` or `label`).
5. Enforces point-in-time correctness and chronological ordering.
6. Employs frozen training-fitted preprocessing (imputer, scaler, model) only;
   never refits on incoming market data.
7. Explicitly reports unavailable model artifacts, schema mismatches, or missing
   data rather than silently retraining or zero-filling.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from src.models.spcm import PRICE_LAGS, build_price_features
from src.utils.indian_market_data import (
    DEFAULT_INDIAN_SYMBOLS,
    DataValidationError,
    FeedType,
    IndianMarketDataFetcher,
    MarketDataResult,
    extract_stock_code,
    normalize_symbol,
)

logger = logging.getLogger("price_features_inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Minimum required trading records to compute 30-day lag on pct_change (ret_1d)
# Row 0: ret_1d is NaN
# Row 1: ret_1d is valid
# Row 31: shift(30) maps to Row 1 (valid). Total records = 32 (indices 0..31).
MIN_REQUIRED_WARMUP_BARS = max(PRICE_LAGS) + 2  # 30 + 2 = 32 bars

# Default warm-up period requested from Yahoo Finance to ensure >= 32 trading days
# 60 calendar days typically provides ~40-42 trading days in Indian market.
DEFAULT_WARMUP_PERIOD = "60d"

# Standard price feature column schema for raw OHLCV inputs (40 features)
STANDARD_OHLCV_PRICE_FEATURES = [
    "open", "high", "low", "close", "volume", "range_co", "range_hl", "ret_1d",
    "open_lag1", "high_lag1", "low_lag1", "close_lag1", "volume_lag1", "range_co_lag1", "range_hl_lag1", "ret_1d_lag1",
    "open_lag5", "high_lag5", "low_lag5", "close_lag5", "volume_lag5", "range_co_lag5", "range_hl_lag5", "ret_1d_lag5",
    "open_lag10", "high_lag10", "low_lag10", "close_lag10", "volume_lag10", "range_co_lag10", "range_hl_lag10", "ret_1d_lag10",
    "open_lag30", "high_lag30", "low_lag30", "close_lag30", "volume_lag30", "range_co_lag30", "range_hl_lag30", "ret_1d_lag30",
]

# Historical Price-Only schema from Phase 3 experiments (46 features, including turnover & previous_close)
HISTORICAL_PRICE_ONLY_FEATURES = [
    "open", "high", "low", "close", "previous_close", "volume", "turnover", "range_co", "range_hl", "ret_1d",
    "open_lag1", "high_lag1", "low_lag1", "close_lag1", "volume_lag1", "turnover_lag1", "range_co_lag1", "range_hl_lag1", "ret_1d_lag1",
    "open_lag5", "high_lag5", "low_lag5", "close_lag5", "volume_lag5", "turnover_lag5", "range_co_lag5", "range_hl_lag5", "ret_1d_lag5",
    "open_lag10", "high_lag10", "low_lag10", "close_lag10", "volume_lag10", "turnover_lag10", "range_co_lag10", "range_hl_lag10", "ret_1d_lag10",
    "open_lag30", "high_lag30", "low_lag30", "close_lag30", "volume_lag30", "turnover_lag30", "range_co_lag30", "range_hl_lag30", "ret_1d_lag30",
]


class InferenceFeatureError(Exception):
    """Base exception for inference feature operations."""
    pass


class InsufficientHistoryError(InferenceFeatureError):
    """Raised when available price history is too short to compute required lags."""
    pass


class MissingFeatureDataError(InferenceFeatureError):
    """Raised when required feature inputs are NaN or missing in the inference row."""
    pass


class SchemaMismatchError(InferenceFeatureError):
    """Raised when generated features do not match expected model schema."""
    pass


class ModelArtifactError(InferenceFeatureError):
    """Raised when trained model artifacts (model, scaler, imputer) are missing or invalid."""
    pass


@dataclass
class ModelArtifactBundle:
    """
    Encapsulates frozen, training-fitted preprocessing and model artifacts.
    Prevents data leakage by ensuring no imputer, scaler, or model is ever refitted
    on incoming inference data.
    """
    model: Any
    scaler: MinMaxScaler
    imputer: SimpleImputer
    feature_cols: list[str]
    model_name: str = "spcm_classifier"
    trained_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: str | Path) -> None:
        """Persist the artifact bundle to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model artifact bundle saved to {path}")

    @classmethod
    def load(cls, filepath: str | Path) -> ModelArtifactBundle:
        """
        Load artifact bundle from disk. Raises ModelArtifactError if missing or invalid.
        """
        path = Path(filepath)
        if not path.exists():
            raise ModelArtifactError(
                f"Trained model artifact not found at '{path}'. "
                "Inference cannot proceed without frozen training artifacts. "
                "Silent retraining on live data or filling with arbitrary zeros is prohibited."
            )
        try:
            bundle = joblib.load(path)
        except Exception as e:
            raise ModelArtifactError(f"Failed to load model artifact bundle from '{path}': {e}") from e

        if not isinstance(bundle, ModelArtifactBundle):
            raise ModelArtifactError(
                f"Loaded object from '{path}' is {type(bundle)}, expected ModelArtifactBundle."
            )

        # Validate internal components
        for attr in ["model", "scaler", "imputer", "feature_cols"]:
            if getattr(bundle, attr, None) is None:
                raise ModelArtifactError(f"Model artifact bundle at '{path}' is missing '{attr}'.")

        return bundle

    def verify_features(self, feature_df: pd.DataFrame, allow_sentiment_nans: bool = True) -> pd.DataFrame:
        """
        Verify that feature_df matches the expected schema and ordering.
        Checks for NaNs in strictly required price columns (raising MissingFeatureDataError),
        while allowing NaNs in sentiment columns ('Sent', 'PV', 'AI') when allow_sentiment_nans is True
        to be handled by self.imputer.transform().
        """
        verify_feature_schema(feature_df, self.feature_cols)

        # Check for NaNs
        ordered_df = feature_df[self.feature_cols].copy()
        sentiment_imputable_cols = {"Sent", "PV", "AI"} if allow_sentiment_nans else set()
        strictly_required_cols = [c for c in self.feature_cols if c not in sentiment_imputable_cols]
        nan_cols = [c for c in strictly_required_cols if ordered_df[c].isna().any()]
        if nan_cols:
            nan_details = {col: int(ordered_df[col].isna().sum()) for col in nan_cols}
            raise MissingFeatureDataError(
                f"Feature row contains missing values (NaNs) in required columns: {nan_details}. "
                "Silently filling required features with arbitrary zeros is prohibited."
            )
        return ordered_df

    def predict_proba(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute inference probabilities using training-fitted preprocessing only.
        Strictly calls transform(); never calls fit() or fit_transform().
        """
        ordered_df = self.verify_features(feature_df)
        X = ordered_df.values.astype(float)

        # Apply frozen preprocessing
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        raw_probs = self.model.predict_proba(X_scaled)
        if raw_probs.shape[1] > 1:
            probs = raw_probs[:, 1]
        else:
            cls = getattr(self.model, "classes_", [1])[0]
            probs = raw_probs[:, 0] if cls == 1 else 1.0 - raw_probs[:, 0]

        result_df = pd.DataFrame({"proba": probs}, index=feature_df.index)
        for col in ["stock", "symbol", "timestamp", "date"]:
            if col in feature_df.columns:
                result_df[col] = feature_df[col]

        return result_df


@dataclass
class InferenceResult:
    """Encapsulates the generated latest feature rows, observation metadata, and history."""
    features_df: pd.DataFrame
    observation_timestamps: dict[str, pd.Timestamp]
    feature_names: list[str]
    history_df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def verify_feature_schema(
    feature_df: pd.DataFrame,
    expected_schema: Sequence[str],
    strict_ordering: bool = True
) -> bool:
    """
    Verifies that feature_df contains all expected feature columns.

    Args:
        feature_df: DataFrame containing candidate feature columns.
        expected_schema: Exact list of feature names expected.
        strict_ordering: If True, checks that column order matches expected_schema.

    Returns:
        True if compatible.

    Raises:
        SchemaMismatchError: If any columns are missing, unexpected, or out of order.
    """
    actual_cols = [c for c in feature_df.columns if c not in ["stock", "symbol", "date", "timestamp"]]
    expected_set = set(expected_schema)
    actual_set = set(actual_cols)

    missing = [c for c in expected_schema if c not in actual_set]
    unexpected = [c for c in actual_cols if c not in expected_set]

    if missing or unexpected:
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing required features ({len(missing)}): {missing}")
        if unexpected:
            msg_parts.append(f"Unexpected features ({len(unexpected)}): {unexpected}")
        raise SchemaMismatchError(f"Feature schema mismatch! {'; '.join(msg_parts)}")

    if strict_ordering and list(actual_cols) != list(expected_schema):
        raise SchemaMismatchError(
            f"Feature ordering mismatch! Actual columns order does not match expected schema order."
        )

    return True


def get_price_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Extract pure price feature column names from a DataFrame, excluding
    identifiers, timestamps, forward labels, and sentiment columns.
    """
    exclude = {
        "stock", "symbol", "date", "timestamp",
        "fwd_ret", "label",
        "news_available", "headline_count", "bullish_count", "bearish_count", "neutral_count",
        "Sent", "PV", "AI", "Homo"
    }
    return [c for c in df.columns if c not in exclude]


class PriceFeatureInferenceEngine:
    """
    Generates inference-time price features for Indian equities.
    Consumes sufficient historical warm-up data, generates lagged price features
    via existing project code, verifies point-in-time correctness, and extracts
    the latest valid feature row per symbol.
    """

    def __init__(
        self,
        fetcher: IndianMarketDataFetcher | None = None,
        min_history_bars: int = MIN_REQUIRED_WARMUP_BARS,
        lags: tuple[int, ...] = PRICE_LAGS,
    ):
        self.fetcher = fetcher or IndianMarketDataFetcher()
        self.min_history_bars = min_history_bars
        self.lags = lags

    def generate_features_from_prices(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price features from an existing normalized OHLCV DataFrame.
        Reuses `build_price_features` from `src/models/spcm.py` and strictly
        omits label generation.

        Args:
            prices_df: DataFrame with columns: date/timestamp, stock/symbol, open, high, low, close, volume.

        Returns:
            DataFrame containing prices and computed price features.

        Raises:
            DataValidationError: If input is empty or missing required OHLCV columns.
            InsufficientHistoryError: If history per stock is less than min_history_bars.
        """
        if prices_df is None or prices_df.empty:
            raise DataValidationError("Empty price DataFrame provided for feature generation.")

        df = prices_df.copy()

        # Standardize stock identifier
        if "stock" not in df.columns:
            if "symbol" in df.columns:
                df["stock"] = df["symbol"].apply(extract_stock_code)
            else:
                raise DataValidationError("DataFrame missing required 'stock' or 'symbol' column.")

        # Standardize date / timestamp
        if "date" not in df.columns:
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
            else:
                raise DataValidationError("DataFrame missing required 'date' or 'timestamp' column.")
        else:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Verify required OHLCV columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required OHLCV columns for feature building: {missing_cols}")

        # Ensure sorted chronologically by stock and date
        sort_col = "timestamp" if "timestamp" in df.columns else "date"
        df = df.sort_values(["stock", sort_col]).reset_index(drop=True)

        # Validate sufficient history per stock
        for stock, grp in df.groupby("stock"):
            if len(grp) < self.min_history_bars:
                raise InsufficientHistoryError(
                    f"Stock '{stock}' has only {len(grp)} bars of history, but at least "
                    f"{self.min_history_bars} bars are required to compute {max(self.lags)}-day lags "
                    "without producing NaN values. Please fetch sufficient historical warm-up data."
                )

        # REUSE EXISTING PROJECT FUNCTION
        # build_price_features computes range_co, range_hl, ret_1d, and lags (1, 5, 10, 30)
        feat_df = build_price_features(df, lags=self.lags)

        # LEAKAGE SAFEGUARD: Assert that no forward-looking labels exist
        if "fwd_ret" in feat_df.columns or "label" in feat_df.columns:
            feat_df = feat_df.drop(columns=[c for c in ["fwd_ret", "label"] if c in feat_df.columns])

        return feat_df

    def extract_latest_feature_rows(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the latest valid inference feature row for each stock.
        Validates that all required price features are populated (non-NaN).

        Args:
            feature_df: DataFrame produced by generate_features_from_prices.

        Returns:
            DataFrame containing 1 row per stock representing the latest observation.

        Raises:
            MissingFeatureDataError: If any feature in the latest row is NaN.
        """
        if feature_df.empty:
            raise DataValidationError("Cannot extract latest features from an empty DataFrame.")

        sort_col = "timestamp" if "timestamp" in feature_df.columns else "date"
        sorted_df = feature_df.sort_values(["stock", sort_col])

        # Extract last chronological row per stock (tail(1) preserves actual row including NaNs)
        latest_rows = sorted_df.groupby("stock", as_index=False).tail(1).reset_index(drop=True)

        # Identify price feature columns
        feature_cols = get_price_feature_columns(latest_rows)

        # Validate no NaNs in the latest feature row
        nan_summary: dict[str, list[str]] = {}
        for _, row in latest_rows.iterrows():
            stock = row["stock"]
            stock_nans = [col for col in feature_cols if pd.isna(row[col])]
            if stock_nans:
                nan_summary[stock] = stock_nans

        if nan_summary:
            raise MissingFeatureDataError(
                f"Latest feature row contains NaN values for stocks: {nan_summary}. "
                "Ensure sufficient warm-up history is provided. Silent zero-filling is prohibited."
            )

        return latest_rows

    def fetch_and_generate_features(
        self,
        symbols: Sequence[str] | str,
        period: str = DEFAULT_WARMUP_PERIOD,
        interval: str = "1d"
    ) -> InferenceResult:
        """
        Fetch sufficient historical warm-up data via IndianMarketDataFetcher and
        generate the latest valid inference feature row for each symbol.

        Args:
            symbols: List of symbols or comma-separated string (e.g. ['RELIANCE.NS', 'TCS.NS']).
            period: Historical warm-up period (default '60d' to ensure >= 32 trading days).
            interval: Bar interval (default '1d').

        Returns:
            InferenceResult containing the latest feature rows and metadata.

        Raises:
            ProviderConnectionError: If data provider cannot be reached.
            DataValidationError: If fetched data fails validation.
            InsufficientHistoryError: If fetched data has fewer than min_history_bars.
        """
        logger.info(f"Fetching {period} warm-up data for symbols: {symbols}")
        market_result: MarketDataResult = self.fetcher.fetch_symbols(symbols, period=period, interval=interval)
        df = market_result.data

        if df.empty:
            failed = market_result.metadata.symbols_failed
            raise DataValidationError(f"No market data could be fetched for requested symbols. Failures: {failed}")

        # Generate price features
        features_full = self.generate_features_from_prices(df)

        # Extract latest row per stock
        latest_rows = self.extract_latest_feature_rows(features_full)

        # Extract observation timestamps
        ts_col = "timestamp" if "timestamp" in latest_rows.columns else "date"
        timestamps = {row["stock"]: pd.to_datetime(row[ts_col]) for _, row in latest_rows.iterrows()}

        feature_cols = get_price_feature_columns(latest_rows)

        return InferenceResult(
            features_df=latest_rows,
            observation_timestamps=timestamps,
            feature_names=feature_cols,
            history_df=features_full,
            metadata={
                "feed_type": market_result.metadata.feed_type.value,
                "provider": market_result.metadata.provider,
                "warmup_period": period,
                "interval": interval,
                "symbols_requested": market_result.metadata.symbols_requested,
                "symbols_successful": market_result.metadata.symbols_successful,
                "symbols_failed": market_result.metadata.symbols_failed,
            }
        )


def generate_inference_price_features(
    symbols: Sequence[str] | str = DEFAULT_INDIAN_SYMBOLS,
    period: str = DEFAULT_WARMUP_PERIOD,
    expected_schema: Sequence[str] | None = None,
) -> InferenceResult:
    """
    Convenience functional API to fetch historical warm-up data and generate
    the latest valid inference feature row for Indian symbols.
    """
    engine = PriceFeatureInferenceEngine()
    result = engine.fetch_and_generate_features(symbols=symbols, period=period)

    if expected_schema is not None:
        verify_feature_schema(result.features_df, expected_schema)

    return result


def cli():
    """CLI to fetch warm-up history and print latest feature rows and observation timestamps."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate inference-time price features for Indian equities."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        help="List of Indian stock symbols (default: RELIANCE.NS TCS.NS INFY.NS)"
    )
    parser.add_argument(
        "--period",
        default="60d",
        help="Historical warm-up period to fetch (default: 60d to ensure >= 32 trading days)"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to a saved ModelArtifactBundle for schema verification & prediction"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to export latest feature rows as CSV"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" INDIAN SPCM INFERENCE-TIME PRICE FEATURE GENERATION (PHASE 2)")
    print("=" * 80)

    engine = PriceFeatureInferenceEngine()
    try:
        result = engine.fetch_and_generate_features(args.symbols, period=args.period)
    except InsufficientHistoryError as ihe:
        print(f"\n[!] INSUFFICIENT HISTORY ERROR: {ihe}")
        return
    except Exception as exc:
        print(f"\n[!] ERROR: {exc}")
        return

    meta = result.metadata
    print(f"\n[DATA FEED STATUS]")
    print(f"  Provider:         {meta.get('provider')}")
    print(f"  Feed Type:        {str(meta.get('feed_type', '')).upper()}")
    print(f"  Warm-up Period:   {meta.get('warmup_period')}")
    successful = meta.get('symbols_successful', [])
    print(f"  Successful:       {len(successful)} {successful}")
    failed = meta.get('symbols_failed', {})
    if failed:
        print(f"  Failed:           {failed}")

    latest_df = result.features_df
    feature_cols = result.feature_names

    print(f"\n[LATEST INFERENCE FEATURE ROWS]")
    print(f"  Total Symbols:    {len(latest_df)}")
    print(f"  Feature Count:    {len(feature_cols)} features per symbol")
    print(f"  Feature Columns:  {feature_cols[:6]} ... {feature_cols[-6:]}")

    # Display per-symbol latest features and timestamps
    for _, row in latest_df.iterrows():
        stock = row["stock"]
        sym = row.get("symbol", f"{stock}.NS")
        ts = result.observation_timestamps[stock]
        close = row["close"]
        ret_1d = row["ret_1d"]
        ret_30 = row["ret_1d_lag30"]
        print(f"\n  Symbol: {sym:<12} (Stock: {stock})")
        print(f"    Observation Timestamp: {ts}")
        print(f"    Latest Close:          {close:.2f}")
        print(f"    1-Day Return:          {ret_1d:+.4f} ({ret_1d * 100:+.2f}%)")
        print(f"    30-Day Lag Return:     {ret_30:+.4f} ({ret_30 * 100:+.2f}%)")
        print(f"    Range (Close-Open):    {row['range_co']:+.2f}")
        print(f"    Range (High-Low):      {row['range_hl']:+.2f}")
        print(f"    Volume:                {int(row['volume']):,}")

    # Verify against standard schema
    try:
        verify_feature_schema(latest_df, STANDARD_OHLCV_PRICE_FEATURES)
        print(f"\n[SCHEMA VERIFICATION]")
        print(f"  Standard OHLCV Schema: PASS (All {len(STANDARD_OHLCV_PRICE_FEATURES)} features verified)")
    except SchemaMismatchError as sme:
        print(f"\n[SCHEMA VERIFICATION NOTE] {sme}")

    # Check Model Artifact Status
    print(f"\n[MODEL ARTIFACT STATUS]")
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"  [!] Model artifact not found at '{model_path}'.")
            print("  Precise Issue: Model artifact is unavailable. Silent retraining is prohibited.")
        else:
            try:
                bundle = ModelArtifactBundle.load(model_path)
                print(f"  Loaded Artifact:   {bundle.model_name}")
                print(f"  Expected Features: {len(bundle.feature_cols)}")
                # Run inference prediction using frozen artifacts
                pred_df = bundle.predict_proba(latest_df)
                print("\n  Predicted Bullish Probabilities:")
                for _, p_row in pred_df.iterrows():
                    print(f"    - {p_row['stock']}: P(label=1) = {p_row['proba']:.4f}")
            except Exception as e:
                print(f"  [!] Model Artifact Error: {e}")
    else:
        print("  No model artifact path specified.")
        print("  Note: Model inference requires frozen training artifacts. Preprocessing/model refitting is prohibited.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols_to_save = ["stock", "symbol", "date", "timestamp"] + feature_cols
        cols_to_save = [c for c in cols_to_save if c in latest_df.columns]
        latest_df[cols_to_save].to_csv(out_path, index=False)
        print(f"\n[OUTPUT] Exported latest feature rows to {out_path}")

    print("=" * 80)


if __name__ == "__main__":
    cli()
