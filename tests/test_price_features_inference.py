"""
Unit tests for Phase 2: Inference-Time Price Feature Generation.

Tests verify:
1. Exact mathematical correctness of lag calculations (lags 1, 5, 10, 30).
2. Detection and rejection of insufficient history (e.g. 5 or 31 bars).
3. Missing data handling and strict rejection of silent zero-filling.
4. Feature-schema compatibility verification (match, missing, extra, reordered).
5. Strict absence of future-return labels (fwd_ret, label) in inference outputs.
6. Point-in-time correctness and observation timestamp alignment.
7. Explicit reporting when trained model artifacts are unavailable.
8. Isolation of training-fitted preprocessing (no fit/refit during inference).
9. End-to-end integration and verification against actual trained model input schema.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from src.features.price_features_inference import (
    PriceFeatureInferenceEngine,
    ModelArtifactBundle,
    verify_feature_schema,
    get_price_feature_columns,
    STANDARD_OHLCV_PRICE_FEATURES,
    HISTORICAL_PRICE_ONLY_FEATURES,
    InsufficientHistoryError,
    MissingFeatureDataError,
    SchemaMismatchError,
    ModelArtifactError,
    MIN_REQUIRED_WARMUP_BARS,
)
from src.utils.indian_market_data import DataValidationError, FeedType, MarketDataMetadata, MarketDataResult
from src.models.spcm import train_spcm, build_fusion_matrix, build_labels, build_price_features


class TestPriceFeaturesInference(unittest.TestCase):

    def _create_synthetic_prices(self, n_bars: int = 40, stock: str = "RELIANCE", step: float = 10.0) -> pd.DataFrame:
        """Helper to create deterministic price history with known geometric progression."""
        dates = pd.date_range("2026-08-01", periods=n_bars, freq="B")
        base_price = 2000.0
        # Price increases by step rupees per day
        closes = [base_price + i * step for i in range(n_bars)]
        opens = [c - 2.0 for c in closes]
        highs = [c + 5.0 for c in closes]
        lows = [c - 5.0 for c in closes]
        volumes = [100000.0 + i * 1000.0 for i in range(n_bars)]

        return pd.DataFrame({
            "stock": [stock] * n_bars,
            "symbol": [f"{stock}.NS"] * n_bars,
            "date": dates,
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

    def test_lag_calculations_exact_values(self):
        """Verify exact mathematical correctness of base and lagged price features."""
        n = 45
        df = self._create_synthetic_prices(n_bars=n, stock="RELIANCE")
        engine = PriceFeatureInferenceEngine()
        feat_df = engine.generate_features_from_prices(df)

        # Check latest row index (t = n - 1)
        t = n - 1
        latest = feat_df.iloc[t]

        # 1. Base range features
        expected_range_co = df.loc[t, "close"] - df.loc[t, "open"]
        expected_range_hl = df.loc[t, "high"] - df.loc[t, "low"]
        self.assertAlmostEqual(latest["range_co"], expected_range_co)
        self.assertAlmostEqual(latest["range_hl"], expected_range_hl)

        # 2. Daily return
        expected_ret_1d = (df.loc[t, "close"] - df.loc[t - 1, "close"]) / df.loc[t - 1, "close"]
        self.assertAlmostEqual(latest["ret_1d"], expected_ret_1d, places=7)

        # 3. Lagged features: lag 1, lag 5, lag 10, lag 30
        for lag in [1, 5, 10, 30]:
            self.assertEqual(latest[f"close_lag{lag}"], df.loc[t - lag, "close"])
            self.assertEqual(latest[f"open_lag{lag}"], df.loc[t - lag, "open"])
            self.assertEqual(latest[f"volume_lag{lag}"], df.loc[t - lag, "volume"])
            self.assertEqual(latest[f"range_co_lag{lag}"], feat_df.loc[t - lag, "range_co"])

            # Verify lagged returns
            expected_lag_ret = (df.loc[t - lag, "close"] - df.loc[t - lag - 1, "close"]) / df.loc[t - lag - 1, "close"]
            self.assertAlmostEqual(latest[f"ret_1d_lag{lag}"], expected_lag_ret, places=7)

    def test_insufficient_history_rejection(self):
        """Verify that providing fewer than 32 daily records raises InsufficientHistoryError."""
        engine = PriceFeatureInferenceEngine()

        # Case A: 5 daily records (typical Phase 1 default period="5d")
        df_5 = self._create_synthetic_prices(n_bars=5)
        with self.assertRaises(InsufficientHistoryError) as ctx:
            engine.generate_features_from_prices(df_5)
        self.assertIn("at least 32 bars are required", str(ctx.exception))
        self.assertIn("5 bars", str(ctx.exception))

        # Case B: 31 daily records (30-day lag on ret_1d would produce NaN on row 30)
        df_31 = self._create_synthetic_prices(n_bars=31)
        with self.assertRaises(InsufficientHistoryError) as ctx:
            engine.generate_features_from_prices(df_31)
        self.assertIn("at least 32 bars are required", str(ctx.exception))

        # Case C: 32 daily records (exact minimum required history -> succeeds!)
        df_32 = self._create_synthetic_prices(n_bars=32)
        feat_32 = engine.generate_features_from_prices(df_32)
        latest_32 = engine.extract_latest_feature_rows(feat_32)
        self.assertEqual(len(latest_32), 1)
        # Ensure zero NaNs on the latest row
        feature_cols = get_price_feature_columns(latest_32)
        self.assertFalse(latest_32[feature_cols].isna().any().any())

    def test_missing_data_and_nan_detection(self):
        """Verify detection of missing values and strict prevention of silent zero-filling."""
        engine = PriceFeatureInferenceEngine()

        # Case A: Missing required OHLCV column
        bad_cols_df = pd.DataFrame({"stock": ["RELIANCE"] * 40, "close": [100.0] * 40})
        with self.assertRaises(DataValidationError):
            engine.generate_features_from_prices(bad_cols_df)

        # Case B: Unhandled NaNs in the latest row trigger MissingFeatureDataError
        df_with_nan = self._create_synthetic_prices(n_bars=40)
        # Artificially inject NaN on the latest bar's close price
        df_with_nan.loc[39, "close"] = np.nan
        # When building features, range_co and ret_1d on row 39 will become NaN
        feat_df = build_price_features(df_with_nan)
        with self.assertRaises(MissingFeatureDataError) as ctx:
            engine.extract_latest_feature_rows(feat_df)
        self.assertIn("contains NaN values", str(ctx.exception))
        self.assertIn("RELIANCE", str(ctx.exception))

    def test_feature_schema_compatibility(self):
        """Verify schema checking against standard OHLCV and historical feature schemas."""
        df = self._create_synthetic_prices(n_bars=35)
        engine = PriceFeatureInferenceEngine()
        feat_df = engine.generate_features_from_prices(df)
        latest_df = engine.extract_latest_feature_rows(feat_df)

        # 1. Standard OHLCV Schema: Exactly matches 40 features
        self.assertTrue(verify_feature_schema(latest_df, STANDARD_OHLCV_PRICE_FEATURES))

        # 2. Missing columns: Historical Phase 3 schema expects turnover and previous_close
        with self.assertRaises(SchemaMismatchError) as ctx:
            verify_feature_schema(latest_df, HISTORICAL_PRICE_ONLY_FEATURES)
        self.assertIn("Missing required features", str(ctx.exception))
        self.assertIn("turnover", str(ctx.exception))
        self.assertIn("previous_close", str(ctx.exception))

        # 3. Extra/Unexpected columns
        extra_df = latest_df.copy()
        extra_df["rogue_feature"] = 123.45
        with self.assertRaises(SchemaMismatchError) as ctx:
            verify_feature_schema(extra_df, STANDARD_OHLCV_PRICE_FEATURES)
        self.assertIn("Unexpected features", str(ctx.exception))
        self.assertIn("rogue_feature", str(ctx.exception))

        # 4. Out of order columns with strict_ordering=True
        reversed_schema = list(reversed(STANDARD_OHLCV_PRICE_FEATURES))
        with self.assertRaises(SchemaMismatchError) as ctx:
            verify_feature_schema(latest_df, reversed_schema, strict_ordering=True)
        self.assertIn("Feature ordering mismatch", str(ctx.exception))

    def test_absence_of_future_label_generation(self):
        """Verify that future returns (fwd_ret) and labels are strictly omitted during inference."""
        df = self._create_synthetic_prices(n_bars=40)
        engine = PriceFeatureInferenceEngine()

        feat_df = engine.generate_features_from_prices(df)
        latest_df = engine.extract_latest_feature_rows(feat_df)

        # Must NOT contain forward return or target labels
        self.assertNotIn("fwd_ret", feat_df.columns)
        self.assertNotIn("label", feat_df.columns)
        self.assertNotIn("fwd_ret", latest_df.columns)
        self.assertNotIn("label", latest_df.columns)

    def test_point_in_time_and_observation_timestamp(self):
        """Verify that latest feature rows strictly reflect the latest point-in-time timestamp."""
        dates = pd.date_range("2026-09-01", periods=35, freq="B")
        df_rel = self._create_synthetic_prices(n_bars=35, stock="RELIANCE")
        df_tcs = self._create_synthetic_prices(n_bars=35, stock="TCS")
        combined = pd.concat([df_rel, df_tcs], ignore_index=True)

        engine = PriceFeatureInferenceEngine()
        feat_df = engine.generate_features_from_prices(combined)
        latest_df = engine.extract_latest_feature_rows(feat_df)

        self.assertEqual(len(latest_df), 2)
        rel_row = latest_df[latest_df["stock"] == "RELIANCE"].iloc[0]
        tcs_row = latest_df[latest_df["stock"] == "TCS"].iloc[0]

        expected_last_timestamp = df_rel["timestamp"].iloc[-1]
        self.assertEqual(pd.to_datetime(rel_row["timestamp"]), expected_last_timestamp)
        self.assertEqual(pd.to_datetime(tcs_row["timestamp"]), expected_last_timestamp)

    def test_model_artifact_unavailable_reporting(self):
        """Verify that missing model artifacts raise ModelArtifactError rather than retraining."""
        non_existent_path = "non_existent_model_dir/spcm_model.joblib"
        with self.assertRaises(ModelArtifactError) as ctx:
            ModelArtifactBundle.load(non_existent_path)
        self.assertIn("Trained model artifact not found", str(ctx.exception))
        self.assertIn("Silent retraining on live data", str(ctx.exception))

    def test_model_artifact_frozen_preprocessing_isolation(self):
        """Verify that imputer and scaler are strictly evaluated via transform(), never fit()."""
        # Fit mock imputer and scaler on training distribution
        train_features = STANDARD_OHLCV_PRICE_FEATURES
        X_train = np.ones((5, len(train_features))) * 50.0

        imputer = SimpleImputer(strategy="median").fit(X_train)
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        bundle = ModelArtifactBundle(
            model=mock_model,
            scaler=scaler,
            imputer=imputer,
            feature_cols=train_features,
            model_name="test_nn_model"
        )

        # Create incoming test feature row with different distribution
        df = self._create_synthetic_prices(n_bars=35)
        engine = PriceFeatureInferenceEngine()
        feat_df = engine.generate_features_from_prices(df)
        latest_row = engine.extract_latest_feature_rows(feat_df)

        # Mock fit/fit_transform to assert they are NEVER called
        with patch.object(bundle.imputer, "fit", side_effect=AssertionError("imputer.fit was called during inference!")):
            with patch.object(bundle.scaler, "fit", side_effect=AssertionError("scaler.fit was called during inference!")):
                with patch.object(bundle.imputer, "fit_transform", side_effect=AssertionError("imputer.fit_transform called!")):
                    with patch.object(bundle.scaler, "fit_transform", side_effect=AssertionError("scaler.fit_transform called!")):
                        pred_df = bundle.predict_proba(latest_row)

        self.assertIn("proba", pred_df.columns)
        self.assertAlmostEqual(pred_df["proba"].iloc[0], 0.7)

    def test_warmup_data_fetching_integration(self):
        """Verify end-to-end fetching and inference feature generation via mock fetcher."""
        mock_fetcher = MagicMock()
        df_fetched = self._create_synthetic_prices(n_bars=40, stock="RELIANCE")

        mock_result = MarketDataResult(
            data=df_fetched,
            metadata=MarketDataMetadata(
                provider="yfinance (Yahoo Finance)",
                feed_type=FeedType.HISTORICAL,
                symbols_requested=["RELIANCE.NS"],
                symbols_successful=["RELIANCE.NS"],
            )
        )
        mock_fetcher.fetch_symbols.return_value = mock_result

        engine = PriceFeatureInferenceEngine(fetcher=mock_fetcher)
        result = engine.fetch_and_generate_features(["RELIANCE.NS"], period="60d")

        self.assertEqual(len(result.features_df), 1)
        self.assertIn("RELIANCE", result.observation_timestamps)
        self.assertEqual(len(result.feature_names), 40)
        self.assertEqual(result.metadata["feed_type"], "historical")

    def test_verification_against_actual_trained_model_expected_input_schema(self):
        """
        Train an SPCM model using the project's actual training code (train_spcm),
        package it into a ModelArtifactBundle, generate live inference features,
        and verify full schema compatibility and probability generation.
        """
        # 1. Create multi-stock synthetic history for training with diverse returns
        stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        train_dfs = [self._create_synthetic_prices(n_bars=60, stock=s, step=2.0 + i * 10.0) for i, s in enumerate(stocks)]
        train_prices = pd.concat(train_dfs, ignore_index=True)
        # Retain standard price columns expected by training pipeline (exclude non-price identifiers like symbol)
        train_prices = train_prices[["stock", "date", "open", "high", "low", "close", "volume"]]

        # 2. Build training features and labels using actual SPCM training functions
        price_feat = build_price_features(train_prices)
        labels = build_labels(train_prices, horizon=5, top_pct=0.4)
        fusion = build_fusion_matrix(price_feat, sentiment_ind=None, homophily_ind=None, labels=labels)

        # 3. Train actual project model (Random Forest baseline)
        model, scaler, imputer, feature_cols = train_spcm(
            fusion,
            classifier="rf",
            include_sentiment=False
        )

        # 4. Package into ModelArtifactBundle
        bundle = ModelArtifactBundle(
            model=model,
            scaler=scaler,
            imputer=imputer,
            feature_cols=feature_cols,
            model_name="rf_spcm_price_only"
        )

        # 5. Generate inference features on new/incoming warm-up data (40 bars)
        new_prices = pd.concat([self._create_synthetic_prices(n_bars=40, stock=s) for s in stocks], ignore_index=True)
        engine = PriceFeatureInferenceEngine()
        features_full = engine.generate_features_from_prices(new_prices)
        latest_rows = engine.extract_latest_feature_rows(features_full)

        # 6. Verify feature row matches the actual trained model's expected input schema
        self.assertTrue(verify_feature_schema(latest_rows, bundle.feature_cols))

        # 7. Execute inference using frozen bundle
        predictions = bundle.predict_proba(latest_rows)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(((predictions["proba"] >= 0.0) & (predictions["proba"] <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
