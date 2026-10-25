"""
FastAPI HTTP Adapter for Indian SPCM Inference Pipeline (Phase 5).

Wraps the production SPCMInferencePipeline and Indian Market Data/News fetchers,
exposing a clean REST API consumed by the Express backend and React dashboard.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features.price_features_inference import (
    DEFAULT_WARMUP_PERIOD,
    InferenceFeatureError,
    InsufficientHistoryError,
    ModelArtifactBundle,
)
from src.inference.spcm_inference_pipeline import (
    DEFAULT_MODEL_PATH,
    SPCMInferencePipeline,
    SPCMInferenceResult,
    StockPrediction,
)
from src.utils.indian_market_data import (
    DEFAULT_INDIAN_SYMBOLS,
    IndianMarketDataFetcher,
    normalize_symbol,
)
from src.utils.indian_news_fetcher import IST_TIMEZONE, IndianNewsFetcher
from src.features.news_sentiment_inference import score_articles

logger = logging.getLogger("spcm_fastapi")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

from contextlib import asynccontextmanager

# Global singleton pipeline instance
pipeline_instance: Optional[SPCMInferencePipeline] = None


def get_pipeline() -> Optional[SPCMInferencePipeline]:
    global pipeline_instance
    if pipeline_instance is None:
        try:
            model_path = Path(DEFAULT_MODEL_PATH)
            if model_path.exists():
                logger.info(f"Initializing SPCMInferencePipeline from {model_path}...")
                pipeline_instance = SPCMInferencePipeline(model_path=model_path)
                logger.info("SPCMInferencePipeline initialized successfully.")
            else:
                logger.warning(f"Model bundle not found at {model_path}.")
        except Exception as exc:
            logger.error(f"Failed to initialize SPCMInferencePipeline: {exc}", exc_info=True)
    return pipeline_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload pipeline
    get_pipeline()
    yield
    # Shutdown logic if any
    pass


app = FastAPI(
    title="SPCM Intelligence - Indian Stock Prediction API",
    description="Inference backend for Multimodal Indian Stock Price Classification Model (SPCM)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow local Express backend and frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Canonical 10-stock universe metadata
STOCK_CATALOG = [
    {"symbol": "RELIANCE", "ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy & Conglomerate", "exchange": "NSE"},
    {"symbol": "TCS", "ticker": "TCS.NS", "name": "Tata Consultancy Services Ltd", "sector": "Information Technology", "exchange": "NSE"},
    {"symbol": "HDFCBANK", "ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking & Financial Services", "exchange": "NSE"},
    {"symbol": "INFY", "ticker": "INFY.NS", "name": "Infosys Ltd", "sector": "Information Technology", "exchange": "NSE"},
    {"symbol": "ICICIBANK", "ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking & Financial Services", "exchange": "NSE"},
    {"symbol": "HINDUNILVR", "ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd", "sector": "Consumer Goods (FMCG)", "exchange": "NSE"},
    {"symbol": "ITC", "ticker": "ITC.NS", "name": "ITC Ltd", "sector": "Consumer Goods & Conglomerate", "exchange": "NSE"},
    {"symbol": "SBIN", "ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Public Sector Banking", "exchange": "NSE"},
    {"symbol": "BHARTIARTL", "ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "sector": "Telecommunications", "exchange": "NSE"},
    {"symbol": "KOTAKBANK", "ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking & Financial Services", "exchange": "NSE"},
]


# --- Request & Response Models ---

class PredictRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="List of stock tickers (e.g. ['RELIANCE', 'TCS'])")
    top_k: int = Field(default=3, ge=1, le=10, description="Top K recommendations")
    warmup_period: str = Field(default="60d", description="Historical price warm-up window")


class PredictionItem(BaseModel):
    stock: str
    symbol: str
    close_price: float
    probability: float
    rank: int
    is_recommended: bool
    sentiment_label: str
    news_count: int
    sent_score: Optional[float] = None
    pv_volume: Optional[float] = None
    ai_index: Optional[float] = None
    observation_timestamp: str
    trading_date: str


class PredictResponse(BaseModel):
    status: str
    is_demo_data: bool = False
    inference_timestamp: str
    top_k: int
    recommended_stocks: List[str]
    predictions: List[PredictionItem]
    model_metadata: dict[str, Any]
    market_metadata: dict[str, Any]
    news_metadata: dict[str, Any]


# --- Endpoints ---

@app.get("/health")
def health():
    """Health check reporting Python ML service, model bundle, and metadata."""
    pipeline = get_pipeline()
    is_loaded = pipeline is not None
    meta = pipeline.bundle.metadata if is_loaded and pipeline else {}
    return {
        "status": "healthy" if is_loaded else "degraded",
        "service": "spcm_python_ml_engine",
        "model_loaded": is_loaded,
        "model_path": str(Path(DEFAULT_MODEL_PATH).resolve()),
        "model_name": meta.get("model_name", "spcm_nn_price_and_sentiment"),
        "classifier": meta.get("classifier", "nn"),
        "features_count": len(pipeline.bundle.feature_cols) if is_loaded and pipeline else 45,
        "training_window": f"{meta.get('start_date')} to {meta.get('end_date')}",
        "in_sample_auc": meta.get("in_sample_auc"),
        "supported_stocks_count": len(STOCK_CATALOG),
        "target_universe": [s["symbol"] for s in STOCK_CATALOG],
        "target_definition": "Top 30% of cross-sectional 10-day forward returns (binary classification)",
        "feed_classification": "HISTORICAL / DELAYED (Yahoo Finance NSE, ~15m delay during market hours)",
        "timestamp": datetime.now(tz=IST_TIMEZONE).isoformat(),
    }


@app.get("/api/stocks")
def get_stocks():
    """Returns canonical 10 Indian stock catalog with sector and exchange info."""
    return {
        "count": len(STOCK_CATALOG),
        "stocks": STOCK_CATALOG,
    }


@app.get("/api/stocks/{symbol}/history")
def get_stock_history(
    symbol: str,
    period: str = Query(default="6mo", pattern="^(1mo|3mo|6mo|1y|2y|5y)$")
):
    """
    Fetches genuine historical OHLCV chart series for a given Indian stock ticker.
    """
    clean_sym = symbol.upper().replace(".NS", "")
    matching = [s for s in STOCK_CATALOG if s["symbol"] == clean_sym]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not supported. Supported: {[s['symbol'] for s in STOCK_CATALOG]}")

    exchange_ticker = f"{clean_sym}.NS"
    fetcher = IndianMarketDataFetcher()
    try:
        # Fetch OHLCV history
        df = fetcher.fetch_symbol(exchange_ticker, period=period, interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No market data records returned for {exchange_ticker}")

        df = df.sort_values("date")
        records = []
        for _, row in df.iterrows():
            d_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            records.append({
                "date": d_str,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
            })

        latest_close = records[-1]["close"] if records else 0.0
        prev_close = records[-2]["close"] if len(records) >= 2 else latest_close
        change = round(latest_close - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close > 0 else 0.0

        return {
            "symbol": clean_sym,
            "ticker": exchange_ticker,
            "company_name": matching[0]["name"],
            "sector": matching[0]["sector"],
            "period": period,
            "record_count": len(records),
            "latest_price": latest_close,
            "change": change,
            "change_pct": change_pct,
            "feed_status": "HISTORICAL_EOD",
            "candles": records,
        }
    except Exception as exc:
        logger.warning(f"Live market fetch failed for {symbol} ({exc}); attempting local price dataset fallback...")
        prices_csv = Path("data/processed/india/prices.csv")
        if prices_csv.exists():
            try:
                local_df = pd.read_csv(prices_csv)
                local_df["date"] = pd.to_datetime(local_df["date"])
                stock_df = local_df[local_df["stock"] == clean_sym].sort_values("date")
                if not stock_df.empty:
                    period_days = {"1mo": 22, "3mo": 66, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}.get(period, 126)
                    stock_df = stock_df.tail(period_days)
                    records = []
                    for _, row in stock_df.iterrows():
                        d_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
                        records.append({
                            "date": d_str,
                            "open": round(float(row["open"]), 2),
                            "high": round(float(row["high"]), 2),
                            "low": round(float(row["low"]), 2),
                            "close": round(float(row["close"]), 2),
                            "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                        })
                    latest_close = records[-1]["close"] if records else 0.0
                    prev_close = records[-2]["close"] if len(records) >= 2 else latest_close
                    change = round(latest_close - prev_close, 2)
                    change_pct = round((change / prev_close) * 100, 2) if prev_close > 0 else 0.0
                    return {
                        "symbol": clean_sym,
                        "ticker": exchange_ticker,
                        "company_name": matching[0]["name"],
                        "sector": matching[0]["sector"],
                        "period": period,
                        "record_count": len(records),
                        "latest_price": latest_close,
                        "change": change,
                        "change_pct": change_pct,
                        "feed_status": "HISTORICAL_LOCAL_CACHE",
                        "candles": records,
                    }
            except Exception as cache_exc:
                logger.error(f"Fallback to local cache failed: {cache_exc}")
        logger.error(f"Error fetching history for {symbol}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data for {symbol}: {str(exc)}")


@app.post("/api/predict", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    """
    Executes end-to-end SPCM inference using the frozen MLP model bundle,
    real-time/delayed market price features, and aligned financial news sentiment.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="SPCM Model Artifact Bundle is not loaded or missing. Run 'py -3.11 -m src.models.train_and_export_spcm' first."
        )

    symbols = req.symbols
    if not symbols:
        symbols = [s["symbol"] for s in STOCK_CATALOG]
    else:
        symbols = [s.upper().replace(".NS", "").strip() for s in symbols]

    try:
        result: SPCMInferenceResult = pipeline.predict(
            symbols=symbols,
            top_k=req.top_k,
            warmup_period=req.warmup_period,
        )

        pred_items: List[PredictionItem] = []
        for p in result.predictions:
            pred_items.append(
                PredictionItem(
                    stock=p.stock,
                    symbol=p.symbol,
                    close_price=round(float(p.close_price), 2),
                    probability=round(float(p.probability), 4),
                    rank=int(p.rank),
                    is_recommended=bool(p.is_recommended),
                    sentiment_label=p.sentiment_label,
                    news_count=int(p.news_count),
                    sent_score=round(float(p.sent_score), 4) if p.sent_score is not None and not pd.isna(p.sent_score) else None,
                    pv_volume=round(float(p.pv_volume), 4) if p.pv_volume is not None and not pd.isna(p.pv_volume) else None,
                    ai_index=round(float(p.ai_index), 4) if p.ai_index is not None and not pd.isna(p.ai_index) else None,
                    observation_timestamp=p.observation_timestamp.isoformat() if hasattr(p.observation_timestamp, "isoformat") else str(p.observation_timestamp),
                    trading_date=p.trading_date.strftime("%Y-%m-%d") if hasattr(p.trading_date, "strftime") else str(p.trading_date)[:10],
                )
            )

        m_meta = result.market_metadata
        n_meta = result.news_metadata

        if not pred_items:
            failed_info = f" Failed details: {m_meta.symbols_failed}" if m_meta and m_meta.symbols_failed else ""
            raise HTTPException(
                status_code=502,
                detail=f"Unable to generate predictions because market data could not be retrieved for requested symbols.{failed_info}"
            )

        feed_type_str = m_meta.feed_type.value if hasattr(m_meta.feed_type, "value") else str(m_meta.feed_type)

        return PredictResponse(
            status="success",
            is_demo_data=False,
            inference_timestamp=result.inference_timestamp.isoformat(),
            top_k=result.top_k,
            recommended_stocks=result.get_recommended_stocks(),
            predictions=pred_items,
            model_metadata=result.model_metadata,
            market_metadata={
                "provider": m_meta.provider,
                "feed_type": feed_type_str,
                "delay_minutes": m_meta.delay_minutes,
                "is_live": m_meta.is_live,
                "market_session": m_meta.market_session,
                "symbols_requested": m_meta.symbols_requested,
                "symbols_successful": m_meta.symbols_successful,
                "symbols_failed": m_meta.symbols_failed,
                "latest_timestamp": m_meta.latest_timestamp.isoformat() if m_meta.latest_timestamp is not None else None,
            },
            news_metadata={
                "provider": n_meta.provider,
                "symbols_requested": n_meta.symbols_requested,
                "symbols_successful": n_meta.symbols_successful,
                "symbols_failed": n_meta.symbols_failed,
                "total_articles_fetched": n_meta.total_articles_fetched,
                "total_articles_deduplicated": n_meta.total_articles_deduplicated,
                "rate_limit_info": n_meta.rate_limit_info,
                "notes": n_meta.notes,
            },
        )
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=f"Insufficient history: {str(exc)}")
    except InferenceFeatureError as exc:
        raise HTTPException(status_code=502, detail=f"Inference feature generation failed: {str(exc)}")
    except Exception as exc:
        logger.error(f"Inference execution failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


@app.get("/api/news")
def get_news(
    symbols: Optional[str] = Query(default=None, description="Comma-separated stock symbols"),
    limit: int = Query(default=30, ge=1, le=100)
):
    """
    Fetches real Indian financial news articles for specified stocks,
    scores their sentiment with FinBERT / Lexicon fallback, and returns structured cards.
    """
    fetcher = IndianNewsFetcher()
    if symbols:
        target_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        target_symbols = [s["symbol"] for s in STOCK_CATALOG]

    try:
        fetch_res = fetcher.fetch_news_for_symbols(target_symbols, max_articles_per_symbol=5)
        articles = fetch_res.articles
        scored = score_articles(articles)

        formatted_articles = []
        for _, row in scored.head(limit).iterrows():
            pub_dt = row.get("published_at")
            pub_date_str = pub_dt.strftime("%Y-%m-%d %H:%M IST") if hasattr(pub_dt, "strftime") else str(pub_dt)
            pos_p = float(row.get("positive_prob", 0.0))
            neg_p = float(row.get("negative_prob", 0.0))
            sent_score = round(pos_p - neg_p, 4)
            formatted_articles.append({
                "symbol": str(row.get("symbol", "")).replace(".NS", ""),
                "title": str(row.get("title", "")),
                "source": str(row.get("source", "")),
                "url": str(row.get("url", "")),
                "published_at": pub_date_str,
                "sentiment_label": str(row.get("sentiment_label", "neutral")),
                "sentiment_score": sent_score,
                "is_relevant": True,
            })

        return {
            "total_articles": len(formatted_articles),
            "provider": fetch_res.metadata.provider,
            "deduplicated_count": getattr(fetch_res.metadata, "total_articles_deduplicated", len(formatted_articles)),
            "sentiment_method": getattr(fetch_res.metadata, "sentiment_method", "FinBERT / Lexicon Fallback"),
            "articles": formatted_articles,
        }
    except Exception as exc:
        logger.error(f"Error fetching news: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(exc)}")
