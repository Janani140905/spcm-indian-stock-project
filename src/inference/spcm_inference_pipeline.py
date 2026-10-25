"""
End-to-End Indian SPCM Inference Backend (Phase 4).

Connects:
  1. Indian Market Data Fetcher & Normalizer (Phase 1)
  2. Point-in-Time Price Feature Inference Engine (Phase 2)
  3. Indian News Fetcher & Sentiment Pipeline (Phase 3)
  4. Frozen SPCM Model Artifact Bundle (Phase 4)

Provides leak-free, zero-refit stock ranking and recommendation for Indian equities
(NSE/BSE), rigorously enforcing Point-in-Time trading day alignment, NaN imputation
via frozen training medians, and cross-sectional Top-K recommendation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.features.news_sentiment_inference import (
    NSE_MARKET_CLOSE_TIME,
    FinancialLexiconSentimentScorer,
    IndianNewsSentimentPipeline,
    aggregate_point_in_time_sentiment,
    score_articles,
)
from src.features.price_features_inference import (
    DEFAULT_WARMUP_PERIOD,
    InferenceFeatureError,
    InsufficientHistoryError,
    MissingFeatureDataError,
    ModelArtifactBundle,
    ModelArtifactError,
    PriceFeatureInferenceEngine,
    SchemaMismatchError,
    verify_feature_schema,
)
from src.utils.indian_market_data import (
    DEFAULT_INDIAN_SYMBOLS,
    FeedType,
    IndianMarketDataFetcher,
    MarketDataMetadata,
    MarketDataResult,
    extract_stock_code,
    normalize_symbol,
)
from src.utils.indian_news_fetcher import (
    IST_TIMEZONE,
    IndianNewsFetcher,
    NewsFetchMetadata,
    NewsFetchResult,
)

logger = logging.getLogger("spcm_inference_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "models/spcm_model_bundle.joblib"


@dataclass
class StockPrediction:
    """Represents an inference prediction and recommendation for a single stock."""
    stock: str                         # Base ticker (e.g. 'RELIANCE')
    symbol: str                        # Exchange symbol (e.g. 'RELIANCE.NS')
    observation_timestamp: pd.Timestamp# Latest market data observation time (IST)
    trading_date: pd.Timestamp         # Effective trading date
    close_price: float                 # Latest closing price
    probability: float                 # Predicted P(label=1) in [0, 1]
    rank: int                          # 1-based cross-sectional rank (1 = highest P)
    is_recommended: bool               # True if stock in Top-K
    news_count: int = 0                # Number of aligned headlines
    sentiment_label: str = "no_news"   # 'positive', 'negative', 'neutral', 'no_news'
    sent_score: float | None = None    # Sent in [-1, 1] or None
    pv_volume: float | None = None     # PV or None
    ai_index: float | None = None      # AI in [0, 1] or None


@dataclass
class SPCMInferenceResult:
    """Encapsulates end-to-end inference predictions, ranking, and metadata."""
    predictions: list[StockPrediction]
    predictions_df: pd.DataFrame
    feature_matrix_df: pd.DataFrame
    market_metadata: MarketDataMetadata
    news_metadata: NewsFetchMetadata
    model_metadata: dict[str, Any]
    top_k: int
    inference_timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz=IST_TIMEZONE))

    def get_recommended_stocks(self) -> list[str]:
        """Return list of recommended stock tickers."""
        return [p.stock for p in self.predictions if p.is_recommended]


class SPCMInferencePipeline:
    """
    Production-grade end-to-end inference engine for Indian SPCM.
    Loads a frozen training artifact bundle, ingests real-time/delayed market data
    and news sentiment, fuses them point-in-time, and generates stock recommendations.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        market_fetcher: IndianMarketDataFetcher | None = None,
        news_fetcher: IndianNewsFetcher | None = None,
        default_exchange: str = "NS",
    ):
        self.model_path = Path(model_path)
        self.default_exchange = default_exchange.upper()

        # Load frozen ModelArtifactBundle
        logger.info(f"Loading frozen SPCM model artifact bundle from '{self.model_path}'...")
        self.bundle = ModelArtifactBundle.load(self.model_path)
        logger.info(
            f"Loaded model '{self.bundle.model_name}' "
            f"({type(self.bundle.model).__name__}, {len(self.bundle.feature_cols)} features)."
        )

        self.market_fetcher = market_fetcher or IndianMarketDataFetcher(default_exchange=self.default_exchange)
        self.price_engine = PriceFeatureInferenceEngine(fetcher=self.market_fetcher)
        self.news_fetcher = news_fetcher or IndianNewsFetcher()

    def predict(
        self,
        symbols: Sequence[str] | str | None = None,
        top_k: int = 3,
        warmup_period: str = DEFAULT_WARMUP_PERIOD,
        max_articles_per_symbol: int = 10,
        max_news_age_days: int = 7,
    ) -> SPCMInferenceResult:
        """
        Execute end-to-end prediction and ranking for the specified Indian stocks.

        Args:
            symbols: Ticker symbols (e.g. ['RELIANCE', 'TCS', 'INFY']).
                     Defaults to model's trained universe.
            top_k: Number of highest-ranked stocks to recommend (default: 3).
            warmup_period: Data period to fetch for lag computation (default: '60d').
            max_articles_per_symbol: Maximum news articles per stock.
            max_news_age_days: Filter out news older than this threshold.

        Returns:
            SPCMInferenceResult containing predictions, rankings, and audit metadata.
        """
        # Resolve target symbols
        if symbols is None:
            target_symbols = self.bundle.metadata.get("universe_stocks", DEFAULT_INDIAN_SYMBOLS)
        elif isinstance(symbols, str):
            target_symbols = [s.strip() for s in symbols.replace(",", " ").split() if s.strip()]
        else:
            target_symbols = list(symbols)

        norm_symbols = [normalize_symbol(s, self.default_exchange) for s in target_symbols]
        stock_codes = [extract_stock_code(s) for s in norm_symbols]
        logger.info(f"Running SPCM inference for {len(norm_symbols)} stocks: {stock_codes}")

        # ---------------------------------------------------------------------
        # 1. Fetch Market Data & Generate Point-in-Time Price Features
        # ---------------------------------------------------------------------
        logger.info(f"Fetching {warmup_period} market data warm-up for price lags...")
        market_res: MarketDataResult = self.market_fetcher.fetch_symbols(
            symbols=norm_symbols,
            period=warmup_period,
            interval="1d"
        )
        if market_res.data.empty:
            failed_info = f" Failed details: {market_res.metadata.symbols_failed}" if market_res.metadata.symbols_failed else ""
            raise InferenceFeatureError(f"Market data fetch failed for all requested symbols.{failed_info}")

        # Compute price features with lags (1, 5, 10, 30)
        price_feat_all = self.price_engine.generate_features_from_prices(market_res.data)
        latest_price_feat = self.price_engine.extract_latest_feature_rows(price_feat_all)
        logger.info(f"Generated latest price features for {len(latest_price_feat)} stocks.")

        # ---------------------------------------------------------------------
        # 2. Ingest News & Generate Point-in-Time Sentiment Features
        # ---------------------------------------------------------------------
        sentiment_included = any(c in self.bundle.feature_cols for c in ["Sent", "PV", "AI", "news_available"])
        scored_news_df = pd.DataFrame()
        daily_sentiment_df = pd.DataFrame()
        news_meta = NewsFetchMetadata()

        if sentiment_included:
            logger.info("Model requires sentiment features. Ingesting aligned news...")
            news_res: NewsFetchResult = self.news_fetcher.fetch_news_for_symbols(
                symbols=norm_symbols,
                max_articles_per_symbol=max_articles_per_symbol,
                max_age_days=max_news_age_days,
            )
            news_meta = news_res.metadata

            if news_res.articles:
                scorer = FinancialLexiconSentimentScorer()
                scored_news_df = score_articles(news_res.articles, scorer=scorer)
            logger.info(f"Scored {len(scored_news_df)} articles across requested symbols.")

        # ---------------------------------------------------------------------
        # 3. Fuse Price Features & Sentiment Features
        # ---------------------------------------------------------------------
        logger.info("Fusing price and sentiment features into model matrix...")
        fusion_df = latest_price_feat.copy()

        if sentiment_included:
            from src.features.news_sentiment_inference import calculate_spcm_formulas
            sent_records = []
            for stk in stock_codes:
                stk_news = scored_news_df[scored_news_df["stock"] == stk] if not scored_news_df.empty else pd.DataFrame()
                if not stk_news.empty:
                    n_tot = len(stk_news)
                    n_bull = int((stk_news["sentiment_label"] == "positive").sum())
                    n_bear = int((stk_news["sentiment_label"] == "negative").sum())
                    n_neu = int((stk_news["sentiment_label"] == "neutral").sum())
                    s, p, a = calculate_spcm_formulas(n_bull, n_bear, n_neu)
                    sent_records.append({
                        "stock": stk,
                        "headline_count": n_tot,
                        "bullish_count": n_bull,
                        "bearish_count": n_bear,
                        "neutral_count": n_neu,
                        "Sent": s,
                        "PV": p,
                        "AI": a,
                        "news_available": 1 if n_tot > 0 else 0,
                    })
                else:
                    sent_records.append({
                        "stock": stk,
                        "headline_count": 0,
                        "bullish_count": 0,
                        "bearish_count": 0,
                        "neutral_count": 0,
                        "Sent": np.nan,
                        "PV": np.nan,
                        "AI": np.nan,
                        "news_available": 0,
                    })
            sent_df = pd.DataFrame(sent_records)
            cols_to_merge = ["Sent", "PV", "AI", "news_available", "headline_count"]
            fusion_df = pd.merge(fusion_df, sent_df[["stock"] + cols_to_merge], on="stock", how="left")

        # ---------------------------------------------------------------------
        # 4. Verify Schema & Evaluate Model Predictions
        # ---------------------------------------------------------------------
        logger.info("Applying frozen preprocessing and predicting forward probabilities...")
        # Verify schema: ordered_df has exact columns matching bundle.feature_cols
        ordered_df = self.bundle.verify_features(fusion_df, allow_sentiment_nans=True)
        X = ordered_df.values.astype(float)

        # Apply frozen preprocessing (NEVER fit)
        X_imputed = self.bundle.imputer.transform(X)
        X_scaled = self.bundle.scaler.transform(X_imputed)

        raw_probs = self.bundle.model.predict_proba(X_scaled)
        if raw_probs.shape[1] > 1:
            probabilities = raw_probs[:, 1]
        else:
            cls = getattr(self.bundle.model, "classes_", [1])[0]
            probabilities = raw_probs[:, 0] if cls == 1 else 1.0 - raw_probs[:, 0]

        fusion_df["proba"] = probabilities

        # ---------------------------------------------------------------------
        # 5. Cross-Sectional Ranking & Top-K Recommendation
        # ---------------------------------------------------------------------
        # Rank by probability descending (1 = highest P(label=1))
        fusion_df["rank"] = fusion_df["proba"].rank(ascending=False, method="min").astype(int)
        effective_top_k = min(top_k, len(fusion_df))
        fusion_df["is_recommended"] = fusion_df["rank"] <= effective_top_k

        # ---------------------------------------------------------------------
        # 6. Assemble Structured Output
        # ---------------------------------------------------------------------
        predictions: list[StockPrediction] = []
        for _, row in fusion_df.sort_values("rank").iterrows():
            stk = row["stock"]
            sym = normalize_symbol(stk, self.default_exchange)

            # Sentiment indicators
            n_count = int(row.get("headline_count", 0))
            s_val = float(row["Sent"]) if "Sent" in row and pd.notna(row["Sent"]) else None
            p_val = float(row["PV"]) if "PV" in row and pd.notna(row["PV"]) else None
            a_val = float(row["AI"]) if "AI" in row and pd.notna(row["AI"]) else None

            if n_count == 0:
                s_label = "no_news"
            elif s_val is not None:
                if s_val > 0.05:
                    s_label = "bullish"
                elif s_val < -0.05:
                    s_label = "bearish"
                else:
                    s_label = "neutral"
            else:
                s_label = "neutral"

            pred = StockPrediction(
                stock=stk,
                symbol=sym,
                observation_timestamp=pd.Timestamp(row.get("timestamp", row["date"])),
                trading_date=pd.Timestamp(row["date"]),
                close_price=float(row["close"]),
                probability=float(row["proba"]),
                rank=int(row["rank"]),
                is_recommended=bool(row["is_recommended"]),
                news_count=n_count,
                sentiment_label=s_label,
                sent_score=s_val,
                pv_volume=p_val,
                ai_index=a_val,
            )
            predictions.append(pred)

        pred_dicts = [
            {
                "stock": p.stock,
                "symbol": p.symbol,
                "observation_timestamp": p.observation_timestamp,
                "close_price": p.close_price,
                "probability": p.probability,
                "rank": p.rank,
                "is_recommended": p.is_recommended,
                "news_count": p.news_count,
                "sentiment_label": p.sentiment_label,
                "Sent": p.sent_score if p.sent_score is not None else np.nan,
                "PV": p.pv_volume if p.pv_volume is not None else np.nan,
                "AI": p.ai_index if p.ai_index is not None else np.nan,
            }
            for p in predictions
        ]
        pred_df = pd.DataFrame(pred_dicts)

        return SPCMInferenceResult(
            predictions=predictions,
            predictions_df=pred_df,
            feature_matrix_df=ordered_df,
            market_metadata=market_res.metadata,
            news_metadata=news_meta,
            model_metadata=self.bundle.metadata,
            top_k=effective_top_k,
        )


def cli_inference():
    """Command-line interface to execute live end-to-end Indian SPCM inference."""
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Phase 4: Complete End-to-End Indian SPCM Inference Backend."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE", "TCS", "INFY"],
        help="Stock symbols to predict (e.g. RELIANCE TCS INFY or RELIANCE.NS TCS.NS INFY.NS)"
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to frozen model artifact bundle (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of Top-K stocks to recommend (default: 3)"
    )
    parser.add_argument(
        "--warmup",
        default="60d",
        help="Historical market data warm-up period (default: 60d)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to export predictions CSV"
    )
    parser.add_argument(
        "--track-outcomes",
        action="store_true",
        help="Record prediction outcomes for 10-day forward resolution"
    )

    args = parser.parse_args()

    print("=" * 85)
    print(" END-TO-END INDIAN SPCM INFERENCE BACKEND - PHASE 4")
    print("=" * 85)

    pipeline = SPCMInferencePipeline(model_path=args.model_path)
    symbols = []
    for s in args.symbols:
        symbols.extend([part.strip() for part in s.split(",") if part.strip()])

    result = pipeline.predict(
        symbols=symbols,
        top_k=args.top_k,
        warmup_period=args.warmup,
    )

    if args.track_outcomes:
        from src.evaluation.outcome_tracker import PredictionOutcomeTracker
        tracker = PredictionOutcomeTracker()
        recorded = tracker.record_inference_result(result)
        logger.info(f"Recorded {len(recorded)} predictions in PENDING status for 10-day forward tracking.")

    m_meta = result.market_metadata
    n_meta = result.news_metadata
    b_meta = result.model_metadata

    # 1. Pipeline & Provider Status
    print("\n[1. DATA SOURCE & FEED INFORMATION]")
    print(f"  Market Provider:          {m_meta.provider}")
    print(f"  Feed Classification:      {m_meta.feed_type.value.upper()} (Delay: {m_meta.delay_minutes} min)")
    print(f"  News Provider:            {n_meta.provider} (API Key: {n_meta.api_key_required})")
    print(f"  News Dedup Stats:         {n_meta.total_articles_fetched} fetched -> {n_meta.total_articles_deduplicated} deduped")
    print(f"  Inference Timestamp (IST):{result.inference_timestamp}")

    # 2. Frozen Model Artifact Details
    print("\n[2. FROZEN MODEL ARTIFACT METADATA]")
    print(f"  Model Artifact Path:      {Path(args.model_path).resolve()}")
    print(f"  Model Architecture:       {b_meta.get('classifier', 'nn').upper()} ({b_meta.get('model_name')})")
    print(f"  Feature Mode:             {b_meta.get('feature_mode')}")
    print(f"  Total Features Used:      {b_meta.get('feature_count')}")
    print(f"  Training Window:          {b_meta.get('start_date')} to {b_meta.get('end_date')}")
    print(f"  Training Sample Count:    {b_meta.get('training_samples')} observations")
    print(f"  In-Sample Training AUC:   {b_meta.get('in_sample_auc', float('nan')):.4f}")

    # 3. Predictions and Cross-Sectional Ranking
    print("\n[3. PREDICTIONS & CROSS-SECTIONAL RANKING]")
    display_cols = [
        "rank", "stock", "close_price", "probability", "is_recommended",
        "sentiment_label", "news_count", "Sent", "PV", "AI"
    ]
    df_disp = result.predictions_df[display_cols].copy()
    df_disp.columns = [
        "Rank", "Stock", "Close (₹)", "P(Bullish)", "Top-K",
        "Sentiment", "Headlines", "Sent", "PV", "AI"
    ]
    print(df_disp.to_string(index=False))

    # 4. Top-K Recommendations
    rec_stocks = result.get_recommended_stocks()
    print("\n[4. TOP-K PORTFOLIO RECOMMENDATION]")
    print(f"  Target Selection Count:   {result.top_k}")
    print(f"  Recommended Stocks:       {', '.join(rec_stocks)}")
    for p in result.predictions:
        if p.is_recommended:
            print(f"    - Rank {p.rank}: {p.stock:<10} | P(Bullish) = {p.probability:.4f} | Close = ₹{p.close_price:.2f} | Sentiment = {p.sentiment_label.upper()}")

    # Export if requested
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        result.predictions_df.to_csv(out_p, index=False)
        print(f"\n[OUTPUT] Predictions saved to '{out_p.resolve()}'.")

    print("=" * 85)


if __name__ == "__main__":
    cli_inference()
