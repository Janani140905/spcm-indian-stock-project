"""
Inference-Time News Ingestion & Point-in-Time Sentiment Integration (Phase 3).

Provides point-in-time, leak-free sentiment processing for live and historical
Indian stock news, fully compatible with the existing SPCM training schema.

Key Principles:
1. Reuses exact training-time sentiment formulas (Sent, PV, AI) from `src/features/india_sentiment.py`.
2. Supports FinBERT (ProsusAI/finbert) as the primary financial sentiment model,
   with a high-quality deterministic financial lexicon fallback for environments without
   PyTorch / Hugging Face dependencies.
3. Enforces Point-in-Time correctness:
   - News published after market close (15:30 IST) or on weekends/holidays is strictly
     mapped to the next trading day.
   - Supports both intraday cutoff alignment and conservative training-strict daily lag.
   - Prevents future-information leakage into market predictions.
4. Preserves NaN semantics for no-news days and neutral-only days:
   - Never silently replaces missing sentiment with arbitrary zero values.
   - Sets news_available=0 on no-news days and preserves Sent=NaN, PV=NaN, AI=NaN,
     allowing downstream SimpleImputer in `src/models/spcm.py` to handle imputation.
5. Employs frozen model pipelines and provides comprehensive schema validation.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pytz

from src.utils.indian_market_data import (
    DEFAULT_INDIAN_SYMBOLS,
    extract_stock_code,
    normalize_symbol,
)
from src.utils.indian_news_fetcher import (
    IST_TIMEZONE,
    IndianNewsFetcher,
    NewsArticle,
    NewsFetchMetadata,
    NewsFetchResult,
    deduplicate_articles,
)

logger = logging.getLogger("news_sentiment_inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Standard NSE/BSE regular trading session close time in IST (15:30)
NSE_MARKET_CLOSE_TIME = dt_time(15, 30, 0)

# Expected SPCM sentiment columns in fusion matrix
EXPECTED_SENTIMENT_FUSION_COLUMNS = [
    "stock",
    "date",
    "Sent",
    "PV",
    "AI",
    "news_available",
    "headline_count",
]

# Full daily feature schema matching `src/features/india_sentiment.py`
FULL_DAILY_SENTIMENT_COLUMNS = [
    "stock",
    "date",
    "headline_count",
    "bullish_count",
    "bearish_count",
    "neutral_count",
    "positive_mean_prob",
    "negative_mean_prob",
    "neutral_mean_prob",
    "news_available",
    "Sent",
    "PV",
    "AI",
]


class SentimentPipelineError(Exception):
    """Base exception for sentiment pipeline errors."""
    pass


class ModelUnavailableError(SentimentPipelineError):
    """Raised when the requested sentiment model backend is unavailable."""
    pass


# ---------------------------------------------------------------------------
# Sentiment Scorer Interfaces & Implementations
# ---------------------------------------------------------------------------

class BaseSentimentScorer(ABC):
    """Abstract interface for financial sentiment scoring backends."""

    @abstractmethod
    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Score a list of news headlines or article texts.

        Returns:
            List of dicts with keys:
              - 'sentiment_label': 'positive', 'negative', or 'neutral'
              - 'positive_prob': float in [0, 1]
              - 'negative_prob': float in [0, 1]
              - 'neutral_prob': float in [0, 1]
        """
        raise NotImplementedError


class FinBertSentimentScorer(BaseSentimentScorer):
    """
    Primary financial sentiment scorer using ProsusAI/finbert via Hugging Face transformers.
    Matches the exact model and parameters used during training in `src/features/india_sentiment.py`.

    Requirements:
      - Packages: `torch`, `transformers`
      - Model weights: ~438MB downloaded from Hugging Face hub on initial run.
    """

    def __init__(self, device: str = "auto", batch_size: int = 16):
        self.device_str = device
        self.batch_size = batch_size
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ModelUnavailableError(
                "ProsusAI/finbert requires 'torch' and 'transformers'. "
                "Install them with 'pip install torch transformers' or use backend='lexicon'."
            ) from exc

        if self.device_str == "auto":
            device_id = 0 if torch.cuda.is_available() else -1
        elif self.device_str in ("cuda", "gpu", "0"):
            device_id = 0
        else:
            device_id = -1

        logger.info(f"Loading ProsusAI/finbert pipeline (device: {'CUDA' if device_id >= 0 else 'CPU'})...")
        try:
            self._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device_id,
                top_k=None,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load ProsusAI/finbert model weights: {exc}") from exc

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        import torch
        with torch.inference_mode():
            raw_results = self._pipeline(texts, batch_size=self.batch_size, truncation=True, max_length=512)

        scored_outputs = []
        for res in raw_results:
            if isinstance(res, dict):
                res = [res]
            scores = {d["label"].lower(): float(d["score"]) for d in res}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral", 0.0)

            # Normalize probabilities to ensure sum is exactly 1.0
            total = pos + neg + neu
            if total > 0:
                pos, neg, neu = pos / total, neg / total, neu / total
            else:
                pos, neg, neu = 0.0, 0.0, 1.0

            max_label = max({"positive": pos, "negative": neg, "neutral": neu}, key=lambda k: scores.get(k, 0.0))
            scored_outputs.append({
                "sentiment_label": max_label,
                "positive_prob": pos,
                "negative_prob": neg,
                "neutral_prob": neu,
            })

        return scored_outputs


class FinancialLexiconSentimentScorer(BaseSentimentScorer):
    """
    High-precision, deterministic financial sentiment scorer based on Loughran-McDonald
    and domain-specific Indian market financial terminology.

    Used as an immediate, zero-weight fallback when PyTorch/Transformers are not installed,
    or for deterministic offline testing. Produces calibrated probability vectors summing to 1.0.
    """

    BULLISH_KEYWORDS = {
        "profit", "profits", "profitable", "profitability", "growth", "grow", "grows", "growing",
        "beat", "beats", "beating", "surge", "surges", "surged", "surging", "rally", "rallies",
        "rallied", "rallying", "gain", "gains", "gained", "gaining", "high", "higher", "highest",
        "upgrade", "upgrades", "upgraded", "upgrading", "outperform", "outperformed", "dividend",
        "deal", "deals", "order", "orders", "expansion", "record", "bullish", "boom", "booming",
        "jump", "jumps", "jumped", "jumping", "soar", "soars", "soared", "soaring", "strong",
        "stronger", "positive", "win", "wins", "won", "winning", "acquisition", "buyback",
        "turnaround", "robust", "rise", "rises", "rising", "climbs", "climbed", "climbing",
        "advances", "advanced", "advancing", "revenue", "upbeat", "boost", "boosts", "boosted",
        "boosting", "milestone", "partnership", "invest", "investment", "inflow", "inflows",
        "rebound", "rebounds", "optimism", "optimistic", "stellar", "upside", "target", "soaring",
    }

    BEARISH_KEYWORDS = {
        "loss", "losses", "drop", "drops", "dropped", "dropping", "fall", "falls", "fell",
        "falling", "slump", "slumps", "slumped", "slumping", "decline", "declines", "declined",
        "declining", "miss", "misses", "missed", "missing", "fraud", "penalty", "probe",
        "probes", "probed", "downgrade", "downgrades", "downgraded", "downgrading", "underperform",
        "warning", "warnings", "debt", "debts", "default", "defaults", "defaulted", "weak",
        "weakness", "bearish", "plunge", "plunges", "plunged", "plunging", "tumble", "tumbles",
        "tumbled", "tumbling", "crash", "crashes", "crashed", "crashing", "slashed", "slashing",
        "down", "cut", "cuts", "cutting", "sink", "sinks", "sank", "sinking", "deficit",
        "risk", "risks", "scam", "investigation", "investigations", "layoff", "layoffs",
        "lower", "negative", "caution", "outflow", "outflows", "selloff", "selloffs",
        "fine", "fined", "fines", "headwinds", "headwind", "pessimism", "pessimistic",
    }

    NEGATORS = {"not", "no", "never", "failed", "fails", "failure", "neither", "barely", "hardly", "unlikely"}

    def __init__(self, neutral_prior: float = 0.5, temperature: float = 1.0):
        self.neutral_prior = neutral_prior
        self.temperature = temperature

    def score_text(self, text: str) -> dict[str, Any]:
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = clean.split()

        bull_score = 0.0
        bear_score = 0.0

        for i, word in enumerate(tokens):
            is_negated = False
            # Check window of 2 preceding tokens for negation
            for j in range(max(0, i - 2), i):
                if tokens[j] in self.NEGATORS:
                    is_negated = True
                    break

            is_bull = word in self.BULLISH_KEYWORDS
            is_bear = word in self.BEARISH_KEYWORDS

            # Check basic stemming if not directly matched
            if not is_bull and not is_bear:
                if word.endswith("ing") and len(word) > 5:
                    stem = word[:-3]
                    is_bull = stem in self.BULLISH_KEYWORDS or (stem + "e") in self.BULLISH_KEYWORDS
                    is_bear = stem in self.BEARISH_KEYWORDS or (stem + "e") in self.BEARISH_KEYWORDS
                elif word.endswith("ed") and len(word) > 4:
                    stem = word[:-2]
                    is_bull = stem in self.BULLISH_KEYWORDS or (stem + "e") in self.BULLISH_KEYWORDS
                    is_bear = stem in self.BEARISH_KEYWORDS or (stem + "e") in self.BEARISH_KEYWORDS
                elif word.endswith("s") and len(word) > 3:
                    stem = word[:-1]
                    is_bull = stem in self.BULLISH_KEYWORDS
                    is_bear = stem in self.BEARISH_KEYWORDS

            if is_bull:
                if is_negated:
                    bear_score += 1.0
                else:
                    bull_score += 1.0
            elif is_bear:
                if is_negated:
                    bull_score += 0.8
                else:
                    bear_score += 1.0

        # Calibrate raw logits into probabilities using softmax with neutral prior
        # Logits: [bull, bear, neutral_baseline]
        logit_pos = bull_score / self.temperature
        logit_neg = bear_score / self.temperature
        logit_neu = self.neutral_prior / self.temperature if (bull_score == 0 and bear_score == 0) else 0.2 / self.temperature

        max_logit = max(logit_pos, logit_neg, logit_neu)
        exp_pos = math.exp(logit_pos - max_logit)
        exp_neg = math.exp(logit_neg - max_logit)
        exp_neu = math.exp(logit_neu - max_logit)
        sum_exp = exp_pos + exp_neg + exp_neu

        p_pos = exp_pos / sum_exp
        p_neg = exp_neg / sum_exp
        p_neu = exp_neu / sum_exp

        # Determine discrete label
        if p_pos > p_neg and p_pos > p_neu:
            label = "positive"
        elif p_neg > p_pos and p_neg > p_neu:
            label = "negative"
        else:
            label = "neutral"

        return {
            "sentiment_label": label,
            "positive_prob": float(p_pos),
            "negative_prob": float(p_neg),
            "neutral_prob": float(p_neu),
        }

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.score_text(t) for t in texts]


class MockSentimentScorer(BaseSentimentScorer):
    """Deterministic mock scorer with user-injected outputs for testing."""

    def __init__(self, mapping: dict[str, dict[str, Any]] | None = None, default_label: str = "neutral"):
        self.mapping = mapping or {}
        self.default_label = default_label

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        results = []
        for text in texts:
            if text in self.mapping:
                results.append(self.mapping[text])
            else:
                # Substring matching fallback
                matched = False
                for k, v in self.mapping.items():
                    if k.lower() in text.lower():
                        results.append(v)
                        matched = True
                        break
                if not matched:
                    if self.default_label == "positive":
                        results.append({"sentiment_label": "positive", "positive_prob": 0.85, "negative_prob": 0.05, "neutral_prob": 0.10})
                    elif self.default_label == "negative":
                        results.append({"sentiment_label": "negative", "positive_prob": 0.05, "negative_prob": 0.85, "neutral_prob": 0.10})
                    else:
                        results.append({"sentiment_label": "neutral", "positive_prob": 0.10, "negative_prob": 0.10, "neutral_prob": 0.80})
        return results


def get_sentiment_scorer(backend: str = "auto", device: str = "auto") -> BaseSentimentScorer:
    """
    Factory function for obtaining a compatible financial sentiment scorer.

    Args:
        backend: 'auto', 'finbert', 'lexicon', or 'mock'.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        Instance of BaseSentimentScorer.
    """
    if backend == "mock":
        return MockSentimentScorer()

    if backend == "lexicon":
        return FinancialLexiconSentimentScorer()

    if backend == "finbert":
        return FinBertSentimentScorer(device=device)

    # backend == "auto": Try FinBERT; fall back to FinancialLexicon with informative message
    try:
        import torch
        import transformers
        return FinBertSentimentScorer(device=device)
    except Exception as exc:
        logger.info(
            f"FinBERT backend unavailable ({exc}). "
            "Using deterministic FinancialLexiconSentimentScorer. "
            "To enable FinBERT: pip install torch transformers"
        )
        return FinancialLexiconSentimentScorer()


# ---------------------------------------------------------------------------
# Point-in-Time Alignment & Trading-Day Math
# ---------------------------------------------------------------------------

def align_timestamp_to_trading_day(
    published_at: pd.Timestamp,
    cutoff_time: dt_time = NSE_MARKET_CLOSE_TIME,
    strict_daily_lag: bool = False,
    holidays: Sequence[date] | None = None,
) -> pd.Timestamp:
    """
    Align a publication timestamp to the applicable Indian trading date,
    enforcing Point-in-Time correctness and preventing future-information leakage.

    Rules:
      1. Converts timestamp to Indian Standard Time (IST, UTC+05:30).
      2. If `strict_daily_lag=True` (matching historical training pipeline where
         exact timestamps were absent and all news was shifted to >= t+1 trading day):
         The target date is set to at least calendar day + 1.
      3. If `strict_daily_lag=False`:
         - If published at or after `cutoff_time` (15:30 IST), the market session
           for that date is closed. This news cannot affect that day's closing price.
           It is shifted to at least calendar day + 1.
         - If published before 15:30 IST on a weekday, it is applicable to that date.
      4. If the resulting target date falls on a weekend (Saturday/Sunday) or an Indian
         market holiday, it rolls forward to the next valid trading day.

    Returns:
        Normalized pd.Timestamp (at midnight 00:00:00 IST) representing the applicable trading day.
    """
    if pd.isna(published_at):
        raise ValueError("Cannot align null or NaN publication timestamp.")

    # Timezone normalization to IST
    if published_at.tzinfo is None:
        ist_ts = published_at.tz_localize(IST_TIMEZONE)
    else:
        ist_ts = published_at.tz_convert(IST_TIMEZONE)

    holidays_set = set(holidays) if holidays else set()
    current_date = ist_ts.date()
    pub_time = ist_ts.time()

    if strict_daily_lag:
        # Strictly lag by +1 calendar day, rolling forward to next trading session
        target_date = current_date + timedelta(days=1)
    else:
        # Intraday cut-off logic
        if pub_time >= cutoff_time:
            # Post-market: impacts next trading session
            target_date = current_date + timedelta(days=1)
        else:
            target_date = current_date

    # Roll forward past weekends (Saturday=5, Sunday=6) and holidays
    while target_date.weekday() >= 5 or target_date in holidays_set:
        target_date = target_date + timedelta(days=1)

    # Return normalized timestamp (midnight)
    return pd.Timestamp(target_date)


def calculate_spcm_formulas(bull: int, bear: int, neutral: int) -> tuple[float, float, float]:
    """
    Exact implementation of paper equations (1)-(3) and training formula definitions:
      Sent = (N_bull - N_bear) / (N_bull + N_bear)
      PV   = ln(N_bull + N_bear + 1)
      AI   = 1 - sqrt(1 - Sent^2)

    Edge case handling matching `src/features/india_sentiment.py`:
      - If N_bull + N_bear == 0 (including neutral-only or no-news), returns (NaN, NaN, NaN).
    """
    denom = bull + bear
    if denom == 0:
        return np.nan, np.nan, np.nan

    sent = (bull - bear) / denom
    pv = math.log(denom + 1)
    # Clamp sent within [-1, 1] to avoid float precision domain errors in sqrt
    clamped_sent = max(-1.0, min(1.0, sent))
    ai = 1.0 - math.sqrt(max(0.0, 1.0 - clamped_sent ** 2))

    return sent, pv, ai


# ---------------------------------------------------------------------------
# Article Scoring & Daily Aggregation
# ---------------------------------------------------------------------------

def score_articles(
    articles: list[NewsArticle] | pd.DataFrame,
    scorer: BaseSentimentScorer | None = None,
) -> pd.DataFrame:
    """
    Score article headlines/text using the provided sentiment scorer.
    Matches schema of `news_sentiment.csv` from training.

    Returns:
        pd.DataFrame with columns:
          stock, symbol, title, summary, published_at, published_date,
          source, url, sentiment_label, positive_prob, negative_prob,
          neutral_prob, provider, is_live
    """
    if scorer is None:
        scorer = get_sentiment_scorer(backend="auto")

    if isinstance(articles, pd.DataFrame):
        df = articles.copy()
    elif isinstance(articles, list):
        if not articles:
            return pd.DataFrame(columns=[
                "stock", "symbol", "title", "summary", "published_at",
                "published_date", "source", "url", "sentiment_label",
                "positive_prob", "negative_prob", "neutral_prob",
                "provider", "is_live"
            ])
        df = pd.DataFrame([a.to_dict() for a in articles])
    else:
        raise ValueError("Articles must be a list of NewsArticle or a DataFrame.")

    if df.empty:
        return pd.DataFrame(columns=[
            "stock", "symbol", "title", "summary", "published_at",
            "published_date", "source", "url", "sentiment_label",
            "positive_prob", "negative_prob", "neutral_prob",
            "provider", "is_live"
        ])

    texts = df["title"].astype(str).tolist()
    scores = scorer.score_texts(texts)

    df["sentiment_label"] = [s["sentiment_label"] for s in scores]
    df["positive_prob"] = [s["positive_prob"] for s in scores]
    df["negative_prob"] = [s["negative_prob"] for s in scores]
    df["neutral_prob"] = [s["neutral_prob"] for s in scores]

    return df


def aggregate_point_in_time_sentiment(
    scored_articles: pd.DataFrame,
    trading_dates: Sequence[pd.Timestamp | date] | None = None,
    target_stocks: Sequence[str] | None = None,
    cutoff_time: dt_time = NSE_MARKET_CLOSE_TIME,
    strict_daily_lag: bool = False,
    holidays: Sequence[date] | None = None,
) -> pd.DataFrame:
    """
    Aggregates scored news articles into daily stock sentiment features
    enforcing Point-in-Time trading day alignment.

    Produces the exact training feature schema from `src/features/india_sentiment.py`:
      - headline_count, bullish_count, bearish_count, neutral_count
      - positive_mean_prob, negative_mean_prob, neutral_mean_prob
      - news_available
      - Sent, PV, AI

    Missing news handling:
      - If no news exists for a stock on a trading day, headline_count=0,
        news_available=0, and Sent, PV, AI are preserved as NaN.
      - Never silently replaces missing sentiment with zero values.
    """
    if scored_articles is None or scored_articles.empty:
        df_scored = pd.DataFrame(columns=[
            "stock", "title", "published_at", "sentiment_label",
            "positive_prob", "negative_prob", "neutral_prob"
        ])
    else:
        df_scored = scored_articles.copy()

    # Align each article to its applicable trading day
    if not df_scored.empty:
        df_scored["applicable_date"] = df_scored["published_at"].apply(
            lambda ts: align_timestamp_to_trading_day(
                ts,
                cutoff_time=cutoff_time,
                strict_daily_lag=strict_daily_lag,
                holidays=holidays,
            )
        )
        grp = df_scored.groupby(["stock", "applicable_date"])

        agg_records = []
        for (stk, app_date), group in grp:
            n_tot = len(group)
            n_bull = int((group["sentiment_label"] == "positive").sum())
            n_bear = int((group["sentiment_label"] == "negative").sum())
            n_neu = int((group["sentiment_label"] == "neutral").sum())

            mean_pos = float(group["positive_prob"].mean())
            mean_neg = float(group["negative_prob"].mean())
            mean_neu = float(group["neutral_prob"].mean())

            sent, pv, ai = calculate_spcm_formulas(n_bull, n_bear, n_neu)

            agg_records.append({
                "stock": stk,
                "date": pd.Timestamp(app_date),
                "headline_count": n_tot,
                "bullish_count": n_bull,
                "bearish_count": n_bear,
                "neutral_count": n_neu,
                "positive_mean_prob": mean_pos,
                "negative_mean_prob": mean_neg,
                "neutral_mean_prob": mean_neu,
                "news_available": 1 if n_tot > 0 else 0,
                "Sent": sent,
                "PV": pv,
                "AI": ai,
            })
        daily_df = pd.DataFrame(agg_records)
    else:
        daily_df = pd.DataFrame(columns=FULL_DAILY_SENTIMENT_COLUMNS)

    # If explicit trading calendar grid is provided (e.g. from price data),
    # outer merge to ensure every stock-trading day exists with correct NaN semantics
    if trading_dates is not None and target_stocks is not None:
        grid_rows = []
        norm_dates = [pd.Timestamp(d).normalize() for d in trading_dates]
        for s in target_stocks:
            s_code = extract_stock_code(s)
            for d in norm_dates:
                grid_rows.append({"stock": s_code, "date": d})
        grid_df = pd.DataFrame(grid_rows).drop_duplicates()

        if not daily_df.empty:
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()
            merged = pd.merge(grid_df, daily_df, on=["stock", "date"], how="left")
        else:
            merged = grid_df.copy()
            for col in [c for c in FULL_DAILY_SENTIMENT_COLUMNS if c not in ["stock", "date"]]:
                merged[col] = np.nan

        # Fill counts with 0, news_available with 0, and retain NaNs for probs, Sent, PV, AI
        merged["headline_count"] = merged["headline_count"].fillna(0).astype(int)
        merged["bullish_count"] = merged["bullish_count"].fillna(0).astype(int)
        merged["bearish_count"] = merged["bearish_count"].fillna(0).astype(int)
        merged["neutral_count"] = merged["neutral_count"].fillna(0).astype(int)
        merged["news_available"] = (merged["headline_count"] > 0).astype(int)

        # Ensure Sent, PV, AI are NaN where news_available == 0
        no_news = merged["news_available"] == 0
        merged.loc[no_news, ["Sent", "PV", "AI", "positive_mean_prob", "negative_mean_prob", "neutral_mean_prob"]] = np.nan

        daily_df = merged.sort_values(["stock", "date"]).reset_index(drop=True)

    return daily_df[FULL_DAILY_SENTIMENT_COLUMNS] if not daily_df.empty else pd.DataFrame(columns=FULL_DAILY_SENTIMENT_COLUMNS)


# ---------------------------------------------------------------------------
# High-Level News Sentiment Pipeline
# ---------------------------------------------------------------------------

class IndianNewsSentimentPipeline:
    """
    End-to-end news ingestion, scoring, and point-in-time daily sentiment pipeline.
    Connects `IndianNewsFetcher` -> `BaseSentimentScorer` -> Point-in-Time Alignment.
    """

    def __init__(
        self,
        scorer_backend: str = "auto",
        device: str = "auto",
        cutoff_time: dt_time = NSE_MARKET_CLOSE_TIME,
        strict_daily_lag: bool = False,
        news_fetcher: IndianNewsFetcher | None = None,
    ):
        self.news_fetcher = news_fetcher or IndianNewsFetcher()
        self.scorer = get_sentiment_scorer(backend=scorer_backend, device=device)
        self.cutoff_time = cutoff_time
        self.strict_daily_lag = strict_daily_lag

    def process_symbols(
        self,
        symbols: Sequence[str] | str,
        max_articles_per_symbol: int = 15,
        max_age_days: int | None = 14,
        trading_dates: Sequence[pd.Timestamp | date] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, NewsFetchMetadata]:
        """
        Fetch news, score articles, and compute daily point-in-time features.

        Returns:
            Tuple of:
              - Scored articles DataFrame
              - Aggregated daily features DataFrame
              - News fetch metadata
        """
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.replace(",", " ").split() if s.strip()]

        # 1. Fetch raw news
        fetch_res: NewsFetchResult = self.news_fetcher.fetch_news_for_symbols(
            symbols=symbols,
            max_articles_per_symbol=max_articles_per_symbol,
            max_age_days=max_age_days,
        )

        # 2. Score articles
        scored_df = score_articles(fetch_res.articles, scorer=self.scorer)

        # 3. Aggregate daily sentiment with point-in-time alignment
        target_stocks = [extract_stock_code(s) for s in symbols]
        daily_df = aggregate_point_in_time_sentiment(
            scored_articles=scored_df,
            trading_dates=trading_dates,
            target_stocks=target_stocks,
            cutoff_time=self.cutoff_time,
            strict_daily_lag=self.strict_daily_lag,
        )

        return scored_df, daily_df, fetch_res.metadata


# ---------------------------------------------------------------------------
# CLI & Demonstration
# ---------------------------------------------------------------------------

def cli_demonstration():
    """Command-line interface to demonstrate live news ingestion and sentiment scoring."""
    # Ensure UTF-8 output on Windows console
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Phase 3: Indian Stock News Ingestion & Point-in-Time Sentiment Integration."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE", "TCS", "INFY"],
        help="Indian stock symbols (e.g. RELIANCE TCS INFY or RELIANCE.NS TCS.NS INFY.NS)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Maximum articles per symbol (default: 5)"
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Maximum article age in days (default: 7)"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "finbert", "lexicon", "mock"],
        help="Sentiment model backend (default: auto)"
    )
    parser.add_argument(
        "--strict-daily-lag",
        action="store_true",
        help="Enforce conservative +1 calendar day lag matching historical training"
    )
    parser.add_argument(
        "--out-articles",
        type=str,
        default=None,
        help="Optional path to export scored articles CSV"
    )
    parser.add_argument(
        "--out-daily",
        type=str,
        default=None,
        help="Optional path to export daily sentiment features CSV"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" INDIAN STOCK NEWS INGESTION & POINT-IN-TIME SENTIMENT PIPELINE - PHASE 3")
    print("=" * 80)

    pipeline = IndianNewsSentimentPipeline(
        scorer_backend=args.backend,
        strict_daily_lag=args.strict_daily_lag,
    )

    scored_df, daily_df, meta = pipeline.process_symbols(
        symbols=args.symbols,
        max_articles_per_symbol=args.max_articles,
        max_age_days=args.max_age_days,
    )

    # 1. Print Data Source & Feed Information
    print("\n[1. DATA SOURCE & FEED STATUS]")
    print(f"  Primary News Provider:    {meta.provider}")
    print(f"  API Key Required:         {meta.api_key_required} (Free public RSS/Search)")
    print(f"  Rate Limits & Pacing:     {meta.rate_limit_info}")
    print(f"  Fetch Timestamp (IST):    {meta.fetch_timestamp}")
    print(f"  Symbols Requested:        {meta.symbols_requested}")
    print(f"  Symbols Successful:       {meta.symbols_successful}")
    if meta.symbols_failed:
        print(f"  Symbols Failed:           {meta.symbols_failed}")
    print(f"  Raw Articles Fetched:     {meta.total_articles_fetched}")
    print(f"  Articles Post-Dedup:      {meta.total_articles_deduplicated}")
    print(f"  Feed Classification:      LIVE Ingestion (network endpoint query)")

    # 2. Print Sentiment Model Information
    print("\n[2. SENTIMENT MODEL CONFIGURATION]")
    model_name = type(pipeline.scorer).__name__
    print(f"  Active Model Scorer:      {model_name}")
    if isinstance(pipeline.scorer, FinBertSentimentScorer):
        print(f"  Architecture:             BERT-base-uncased fine-tuned on Financial PhraseBank (ProsusAI/finbert)")
        print(f"  Inference Device:         {pipeline.scorer.device_str}")
        print(f"  Parameters/Weights:       ~438MB PyTorch weights")
    elif isinstance(pipeline.scorer, FinancialLexiconSentimentScorer):
        print(f"  Architecture:             Domain-specific Financial Lexicon with Calibrated Probabilities")
        print(f"  Dependency Requirements:  None (Zero heavy dependencies).")
        print(f"  Note:                     To enable full ProsusAI/finbert: pip install torch transformers")

    # 3. Print Scored Articles
    print("\n[3. INDIVIDUAL ARTICLE SENTIMENT SCORES]")
    if scored_df.empty:
        print("  No recent articles retrieved.")
    else:
        for stock, grp in scored_df.groupby("stock"):
            print(f"\n  --- {stock} ({len(grp)} articles) ---")
            for _, row in grp.iterrows():
                title_safe = str(row['title']).encode("ascii", errors="replace").decode()
                source_safe = str(row['source']).encode("ascii", errors="replace").decode()
                print(f"  * Published (IST): {row['published_at']}")
                print(f"    Source:          {source_safe}")
                print(f"    Headline:        {title_safe}")
                print(f"    Sentiment:       {row['sentiment_label'].upper()} "
                      f"[Pos: {row['positive_prob']:.3f}, Neg: {row['negative_prob']:.3f}, Neu: {row['neutral_prob']:.3f}]")
                print(f"    URL:             {row['url'][:80]}...")

    # 4. Print Daily Aggregation Results
    print("\n[4. POINT-IN-TIME DAILY SENTIMENT AGGREGATION]")
    if daily_df.empty:
        print("  No daily features generated.")
    else:
        display_cols = [
            "stock", "date", "headline_count", "bullish_count", "bearish_count",
            "neutral_count", "Sent", "PV", "AI", "news_available"
        ]
        print(daily_df[display_cols].to_string(index=False))

    # 5. Schema Validation against SPCM Model
    print("\n[5. SPCM MODEL SCHEMA COMPATIBILITY VERIFICATION]")
    missing_fusion = [c for c in EXPECTED_SENTIMENT_FUSION_COLUMNS if c not in daily_df.columns]
    if missing_fusion:
        print(f"  [FAIL] Missing required SPCM fusion columns: {missing_fusion}")
    else:
        print("  [PASS] All expected SPCM fusion columns present:")
        for c in EXPECTED_SENTIMENT_FUSION_COLUMNS:
            print(f"    - {c:<16}: OK (type: {daily_df[c].dtype if not daily_df.empty else 'unfilled'})")

    # Save output if requested
    if args.out_articles and not scored_df.empty:
        p_art = Path(args.out_articles)
        p_art.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(p_art, index=False)
        print(f"\n[OUTPUT] Saved scored articles to {p_art}")

    if args.out_daily and not daily_df.empty:
        p_day = Path(args.out_daily)
        p_day.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(p_day, index=False)
        print(f"\n[OUTPUT] Saved daily sentiment features to {p_day}")

    print("=" * 80)


if __name__ == "__main__":
    cli_demonstration()
