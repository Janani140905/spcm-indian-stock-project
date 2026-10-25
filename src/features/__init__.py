"""Features package for SPCM Indian Stock Project."""

__all__ = [
    "PriceFeatureInferenceEngine",
    "ModelArtifactBundle",
    "InferenceResult",
    "generate_inference_price_features",
    "verify_feature_schema",
    "get_price_feature_columns",
    "STANDARD_OHLCV_PRICE_FEATURES",
    "HISTORICAL_PRICE_ONLY_FEATURES",
    "InferenceFeatureError",
    "InsufficientHistoryError",
    "MissingFeatureDataError",
    "SchemaMismatchError",
    "ModelArtifactError",
    "MIN_REQUIRED_WARMUP_BARS",
    "DEFAULT_WARMUP_PERIOD",
    # Phase 3 News Sentiment Inference exports
    "IndianNewsSentimentPipeline",
    "BaseSentimentScorer",
    "FinBertSentimentScorer",
    "FinancialLexiconSentimentScorer",
    "MockSentimentScorer",
    "get_sentiment_scorer",
    "align_timestamp_to_trading_day",
    "calculate_spcm_formulas",
    "score_articles",
    "aggregate_point_in_time_sentiment",
    "EXPECTED_SENTIMENT_FUSION_COLUMNS",
    "FULL_DAILY_SENTIMENT_COLUMNS",
]

_PRICE_EXPORTS = {
    "PriceFeatureInferenceEngine", "ModelArtifactBundle", "InferenceResult",
    "generate_inference_price_features", "verify_feature_schema",
    "get_price_feature_columns", "STANDARD_OHLCV_PRICE_FEATURES",
    "HISTORICAL_PRICE_ONLY_FEATURES", "InferenceFeatureError",
    "InsufficientHistoryError", "MissingFeatureDataError", "SchemaMismatchError",
    "ModelArtifactError", "MIN_REQUIRED_WARMUP_BARS", "DEFAULT_WARMUP_PERIOD",
}

_SENTIMENT_EXPORTS = {
    "IndianNewsSentimentPipeline", "BaseSentimentScorer", "FinBertSentimentScorer",
    "FinancialLexiconSentimentScorer", "MockSentimentScorer", "get_sentiment_scorer",
    "align_timestamp_to_trading_day", "calculate_spcm_formulas", "score_articles",
    "aggregate_point_in_time_sentiment", "EXPECTED_SENTIMENT_FUSION_COLUMNS",
    "FULL_DAILY_SENTIMENT_COLUMNS",
}


def __getattr__(name: str):
    import importlib
    if name in _PRICE_EXPORTS:
        mod = importlib.import_module("src.features.price_features_inference")
        return getattr(mod, name)
    if name in _SENTIMENT_EXPORTS:
        mod = importlib.import_module("src.features.news_sentiment_inference")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
