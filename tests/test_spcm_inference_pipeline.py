"""
Unit Tests for End-to-End Indian SPCM Inference Backend (Phase 4).

Validates:
1. Frozen ModelArtifactBundle serialization, loading, and component validation.
2. Zero-refit guarantee (asserts fit/fit_transform are never invoked during inference).
3. End-to-end inference execution with mocked market data and news sentiment.
4. Cross-sectional ranking consistency (rank monotonically corresponds to probability).
5. Top-K recommendation selection logic.
6. NaN imputation semantics for stocks without news (Sent/PV/AI NaNs handled via frozen imputer).
7. Strict omission of forward-looking labels (fwd_ret, label).
8. Error resilience on missing artifacts, invalid schemas, and short history.

Note: Runs completely offline using deterministic mocks without network access.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from src.features.price_features_inference import (
    ModelArtifactBundle,
    ModelArtifactError,
    MissingFeatureDataError,
    SchemaMismatchError,
    InsufficientHistoryError,
    STANDARD_OHLCV_PRICE_FEATURES,
)
from src.inference.spcm_inference_pipeline import (
    SPCMInferencePipeline,
    SPCMInferenceResult,
    StockPrediction,
)
from src.utils.indian_market_data import (
    IndianMarketDataFetcher,
    MarketDataMetadata,
    MarketDataResult,
    FeedType,
)
from src.utils.indian_news_fetcher import (
    IndianNewsFetcher,
    MockNewsProvider,
    NewsArticle,
    IST_TIMEZONE,
)


class TestSPCMArtifactBundle(unittest.TestCase):
    """Tests for artifact bundle serialization, loading, and frozen preprocessing."""

    def setUp(self):
        self.feature_cols = STANDARD_OHLCV_PRICE_FEATURES + ["Sent", "PV", "AI", "news_available", "headline_count"]
        X_dummy = np.ones((10, len(self.feature_cols))) * 100.0
        # Inject NaNs in sentiment columns to simulate training
        X_dummy[:5, -5:] = np.nan
        X_dummy[5:, -5:] = [0.0, np.log(2), 0.0, 1.0, 1.0]

        self.imputer = SimpleImputer(strategy="median").fit(X_dummy)
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.imputer.transform(X_dummy))

        y_dummy = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        self.model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
            self.scaler.transform(self.imputer.transform(X_dummy)), y_dummy
        )

        self.bundle = ModelArtifactBundle(
            model=self.model,
            scaler=self.scaler,
            imputer=self.imputer,
            feature_cols=self.feature_cols,
            model_name="test_spcm_rf",
            trained_at="2026-10-21T10:00:00",
            metadata={"test_key": "test_val", "feature_count": len(self.feature_cols)}
        )

    def test_save_and_load_roundtrip(self):
        """Bundle must save and reload identically from disk."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "test_bundle.joblib"
            self.bundle.save(bundle_path)

            reloaded = ModelArtifactBundle.load(bundle_path)
            self.assertEqual(reloaded.model_name, "test_spcm_rf")
            self.assertEqual(len(reloaded.feature_cols), len(self.feature_cols))
            self.assertEqual(reloaded.metadata.get("test_key"), "test_val")

    def test_missing_bundle_raises_error(self):
        """Loading a non-existent bundle file must raise ModelArtifactError."""
        with self.assertRaises(ModelArtifactError) as ctx:
            ModelArtifactBundle.load("non_existent_directory/bundle.joblib")
        self.assertIn("Trained model artifact not found", str(ctx.exception))

    def test_zero_refit_guarantee(self):
        """Inference must strictly call transform() and NEVER call fit() or fit_transform()."""
        # Create test feature row
        row_dict = {col: 105.0 for col in STANDARD_OHLCV_PRICE_FEATURES}
        # Add sentiment columns with NaNs (simulating no news)
        row_dict.update({"Sent": np.nan, "PV": np.nan, "AI": np.nan, "news_available": 0, "headline_count": 0})
        row_df = pd.DataFrame([row_dict])
        row_df["stock"] = "RELIANCE"

        with patch.object(self.bundle.imputer, "fit", side_effect=AssertionError("imputer.fit was called!")):
            with patch.object(self.bundle.scaler, "fit", side_effect=AssertionError("scaler.fit was called!")):
                with patch.object(self.bundle.imputer, "fit_transform", side_effect=AssertionError("imputer.fit_transform called!")):
                    with patch.object(self.bundle.scaler, "fit_transform", side_effect=AssertionError("scaler.fit_transform called!")):
                        with patch.object(self.bundle.model, "fit", side_effect=AssertionError("model.fit called!")):
                            preds = self.bundle.predict_proba(row_df)

        self.assertEqual(len(preds), 1)
        self.assertIn("proba", preds.columns)
        self.assertGreaterEqual(preds["proba"].iloc[0], 0.0)
        self.assertLessEqual(preds["proba"].iloc[0], 1.0)

    def test_missing_price_feature_raises_error(self):
        """Missing values in price columns (e.g. close_lag30) must raise MissingFeatureDataError."""
        row_dict = {col: 105.0 for col in STANDARD_OHLCV_PRICE_FEATURES}
        row_dict["close_lag30"] = np.nan  # Corrupt price lag
        row_dict.update({"Sent": np.nan, "PV": np.nan, "AI": np.nan, "news_available": 0, "headline_count": 0})
        row_df = pd.DataFrame([row_dict])

        with self.assertRaises(MissingFeatureDataError) as ctx:
            self.bundle.verify_features(row_df, allow_sentiment_nans=True)
        self.assertIn("close_lag30", str(ctx.exception))


class DeterministicDummyClassifier:
    """Picklable deterministic classifier for pipeline unit tests."""
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = []
        for i in range(len(X)):
            p = max(0.1, 0.8 - (i * 0.2))
            probs.append([1.0 - p, p])
        return np.array(probs)


class TestSPCMInferencePipeline(unittest.TestCase):
    """Tests for end-to-end SPCM inference, ranking, and recommendations."""

    def _create_mock_market_data(self, stocks: list[str], n_bars: int = 40) -> pd.DataFrame:
        dates = pd.date_range("2026-09-01", periods=n_bars, freq="B")
        rows = []
        for s in stocks:
            px = 1000.0
            for d in dates:
                px += np.random.normal(0.5, 5.0)
                rows.append({
                    "date": d.normalize(),
                    "timestamp": pd.Timestamp(d),
                    "symbol": f"{s}.NS",
                    "stock": s,
                    "open": px - 2.0,
                    "high": px + 5.0,
                    "low": px - 5.0,
                    "close": px,
                    "volume": 100000.0,
                })
        return pd.DataFrame(rows)

    def setUp(self):
        # Create temporary bundle for pipeline testing
        self.feature_cols = STANDARD_OHLCV_PRICE_FEATURES + ["Sent", "PV", "AI", "news_available", "headline_count"]
        X = np.ones((20, len(self.feature_cols))) * 50.0
        X[:10, -5:] = np.nan
        X[10:, -5:] = [0.1, np.log(2), 0.05, 1.0, 2.0]

        self.imputer = SimpleImputer(strategy="median").fit(X)
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.imputer.transform(X))

        # Use picklable deterministic classifier
        self.model = DeterministicDummyClassifier()

        self.bundle = ModelArtifactBundle(
            model=self.model,
            scaler=self.scaler,
            imputer=self.imputer,
            feature_cols=self.feature_cols,
            model_name="mock_spcm_pipeline_model",
            trained_at="2026-10-21T10:00:00",
            metadata={
                "classifier": "mock",
                "model_name": "mock_model",
                "feature_mode": "price_and_sentiment",
                "feature_count": len(self.feature_cols),
                "universe_stocks": ["RELIANCE", "TCS", "INFY"],
                "start_date": "2021-01-01",
                "end_date": "2024-12-31",
                "training_samples": 20,
                "in_sample_auc": 0.85,
            }
        )

        import tempfile
        from pathlib import Path
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.bundle_path = Path(self.tmp_dir.name) / "test_pipeline_bundle.joblib"
        self.bundle.save(self.bundle_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_end_to_end_prediction_with_mock_market_and_news(self):
        """Verify full end-to-end inference pipeline produces ranked recommendations."""
        stocks = ["RELIANCE", "TCS", "INFY"]
        market_df = self._create_mock_market_data(stocks, n_bars=40)

        # Mock MarketDataFetcher
        mock_market_fetcher = MagicMock(spec=IndianMarketDataFetcher)
        mock_market_fetcher.default_exchange = "NS"
        mock_market_fetcher.fetch_symbols.return_value = MarketDataResult(
            data=market_df,
            metadata=MarketDataMetadata(
                provider="MockMarket",
                feed_type=FeedType.HISTORICAL,
                delay_minutes=0,
                symbols_requested=[f"{s}.NS" for s in stocks],
                symbols_successful=[f"{s}.NS" for s in stocks]
            )
        )

        # Mock NewsFetcher with 1 positive headline for RELIANCE
        canned_news = [
            {
                "stock": "RELIANCE",
                "symbol": "RELIANCE.NS",
                "title": "Reliance profits surge with record quarterly beat",
                "published_at": "2026-10-21 10:00:00",
                "source": "MockNews",
                "url": "https://mock.com/rel",
            }
        ]
        mock_news_fetcher = IndianNewsFetcher(custom_provider=MockNewsProvider(canned_articles=canned_news))

        # Initialize SPCMInferencePipeline
        pipeline = SPCMInferencePipeline(
            model_path=self.bundle_path,
            market_fetcher=mock_market_fetcher,
            news_fetcher=mock_news_fetcher,
        )

        result: SPCMInferenceResult = pipeline.predict(symbols=stocks, top_k=2)

        self.assertEqual(len(result.predictions), 3)
        self.assertEqual(result.top_k, 2)

        # Ranking validation: rank 1 must have highest proba, rank 2 second, rank 3 third
        ranks = [p.rank for p in result.predictions]
        probs = [p.probability for p in result.predictions]
        self.assertEqual(ranks, [1, 2, 3])
        self.assertTrue(probs[0] >= probs[1] >= probs[2])

        # Top-K recommendation validation: exactly 2 recommended
        recs = result.get_recommended_stocks()
        self.assertEqual(len(recs), 2)
        self.assertTrue(result.predictions[0].is_recommended)
        self.assertTrue(result.predictions[1].is_recommended)
        self.assertFalse(result.predictions[2].is_recommended)

        # Check RELIANCE has sentiment indicators populated
        rel_pred = [p for p in result.predictions if p.stock == "RELIANCE"][0]
        self.assertEqual(rel_pred.news_count, 1)
        self.assertEqual(rel_pred.sentiment_label, "bullish")
        self.assertAlmostEqual(rel_pred.sent_score, 1.0)

        # Check TCS has no news but still receives valid prediction via imputer
        tcs_pred = [p for p in result.predictions if p.stock == "TCS"][0]
        self.assertEqual(tcs_pred.news_count, 0)
        self.assertEqual(tcs_pred.sentiment_label, "no_news")
        self.assertIsNone(tcs_pred.sent_score)
        self.assertGreaterEqual(tcs_pred.probability, 0.0)

    def test_absence_of_future_labels_in_inference_output(self):
        """Assert that forward-return labels (fwd_ret, label) are NEVER calculated or present."""
        stocks = ["RELIANCE", "TCS", "INFY"]
        market_df = self._create_mock_market_data(stocks, n_bars=40)

        mock_market_fetcher = MagicMock(spec=IndianMarketDataFetcher)
        mock_market_fetcher.default_exchange = "NS"
        mock_market_fetcher.fetch_symbols.return_value = MarketDataResult(
            data=market_df,
            metadata=MarketDataMetadata()
        )
        mock_news_fetcher = IndianNewsFetcher(custom_provider=MockNewsProvider([]))

        pipeline = SPCMInferencePipeline(
            model_path=self.bundle_path,
            market_fetcher=mock_market_fetcher,
            news_fetcher=mock_news_fetcher,
        )
        result = pipeline.predict(symbols=stocks)

        self.assertNotIn("fwd_ret", result.predictions_df.columns)
        self.assertNotIn("label", result.predictions_df.columns)
        self.assertNotIn("fwd_ret", result.feature_matrix_df.columns)
        self.assertNotIn("label", result.feature_matrix_df.columns)

    def test_insufficient_history_error_handling(self):
        """Providing fewer than 32 trading days of market data must raise InsufficientHistoryError."""
        stocks = ["RELIANCE"]
        # Only 10 bars (too short for 30-day lag)
        short_market_df = self._create_mock_market_data(stocks, n_bars=10)

        mock_market_fetcher = MagicMock(spec=IndianMarketDataFetcher)
        mock_market_fetcher.default_exchange = "NS"
        mock_market_fetcher.fetch_symbols.return_value = MarketDataResult(
            data=short_market_df,
            metadata=MarketDataMetadata()
        )
        mock_news_fetcher = IndianNewsFetcher(custom_provider=MockNewsProvider([]))

        pipeline = SPCMInferencePipeline(
            model_path=self.bundle_path,
            market_fetcher=mock_market_fetcher,
            news_fetcher=mock_news_fetcher,
        )

        with self.assertRaises(InsufficientHistoryError) as ctx:
            pipeline.predict(symbols=stocks)
        self.assertIn("at least 32 bars are required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
