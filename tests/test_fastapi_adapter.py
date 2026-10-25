"""
Unit and integration tests for FastAPI adapter (Phase 5).
"""
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.inference.fastapi_adapter import app, STOCK_CATALOG


class TestFastAPIAdapter(unittest.TestCase):
    """Test suite for SPCM FastAPI Adapter."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["service"], "spcm_python_ml_engine")
        self.assertEqual(data["supported_stocks_count"], 10)
        self.assertEqual(data["features_count"], 45)
        self.assertEqual(len(data["target_universe"]), 10)

    def test_stocks_endpoint(self):
        response = self.client.get("/api/stocks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 10)
        symbols = [s["symbol"] for s in data["stocks"]]
        self.assertIn("RELIANCE", symbols)
        self.assertIn("TCS", symbols)
        self.assertIn("HDFCBANK", symbols)
        self.assertIn("INFY", symbols)

    def test_stocks_history_invalid_symbol(self):
        response = self.client.get("/api/stocks/INVALID_SYM/history")
        self.assertEqual(response.status_code, 404)

    def test_predict_endpoint_validation(self):
        # Invalid top_k > 10
        response = self.client.post("/api/predict", json={"top_k": 20})
        self.assertEqual(response.status_code, 422)

    @patch("src.inference.fastapi_adapter.get_pipeline")
    def test_predict_endpoint_metadata_regression(self, mock_get_pipeline):
        """Regression test ensuring MarketDataMetadata and NewsFetchMetadata serialize without attribute errors."""
        import pandas as pd
        from src.utils.indian_market_data import MarketDataMetadata, FeedType
        from src.utils.indian_news_fetcher import NewsFetchMetadata
        from src.inference.spcm_inference_pipeline import StockPrediction, SPCMInferenceResult

        mock_pipeline = MagicMock()
        mock_pipeline.bundle.metadata = {"model_name": "test_bundle"}

        m_meta = MarketDataMetadata(
            provider="yfinance (Yahoo Finance)",
            feed_type=FeedType.HISTORICAL,
            delay_minutes=0,
            symbols_requested=["RELIANCE.NS"],
            symbols_successful=["RELIANCE.NS"],
            symbols_failed={},
        )
        n_meta = NewsFetchMetadata(
            provider="google_news",
            symbols_requested=["RELIANCE.NS"],
            symbols_successful=["RELIANCE.NS"],
            total_articles_fetched=2,
            total_articles_deduplicated=2,
        )
        pred = StockPrediction(
            stock="RELIANCE",
            symbol="RELIANCE.NS",
            close_price=2500.0,
            probability=0.35,
            rank=1,
            is_recommended=True,
            sentiment_label="neutral",
            news_count=2,
            sent_score=0.0,
            pv_volume=1.0,
            ai_index=0.5,
            observation_timestamp=pd.Timestamp.now(),
            trading_date=pd.Timestamp("2024-12-30"),
        )
        mock_res = SPCMInferenceResult(
            predictions=[pred],
            predictions_df=pd.DataFrame([{"stock": "RELIANCE"}]),
            feature_matrix_df=pd.DataFrame([{"stock": "RELIANCE"}]),
            market_metadata=m_meta,
            news_metadata=n_meta,
            model_metadata={"model_name": "test_model"},
            top_k=1,
        )
        mock_pipeline.predict.return_value = mock_res
        mock_get_pipeline.return_value = mock_pipeline

        response = self.client.post("/api/predict", json={"symbols": ["RELIANCE"], "top_k": 1})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("market_metadata", data)
        self.assertEqual(data["market_metadata"]["symbols_successful"], ["RELIANCE.NS"])
        self.assertEqual(data["market_metadata"]["provider"], "yfinance (Yahoo Finance)")
        self.assertIn("news_metadata", data)
        self.assertEqual(data["news_metadata"]["total_articles_fetched"], 2)

    @patch("src.inference.fastapi_adapter.get_pipeline")
    def test_predict_endpoint_market_data_failure_returns_502(self, mock_get_pipeline):
        """Test that market data retrieval failure returns HTTP 502 with informative details."""
        from src.features.price_features_inference import InferenceFeatureError

        mock_pipeline = MagicMock()
        mock_pipeline.predict.side_effect = InferenceFeatureError(
            "Market data fetch failed for all requested symbols. Failed details: {'RELIANCE.NS': 'Network timeout'}"
        )
        mock_get_pipeline.return_value = mock_pipeline

        response = self.client.post("/api/predict", json={"symbols": ["RELIANCE"]})
        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Network timeout", data["detail"])


if __name__ == "__main__":
    unittest.main()
