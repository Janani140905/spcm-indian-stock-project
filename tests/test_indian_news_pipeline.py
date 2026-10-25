"""
Unit Tests for Indian Stock News Ingestion & Point-in-Time Sentiment Integration (Phase 3).

Validates:
1. Timestamp parsing and IST timezone normalization.
2. Point-in-Time trading day alignment (pre-close, post-close, weekends, holidays).
3. Absence of future-information leakage.
4. Duplicate removal (exact URLs, normalized headlines within rolling window).
5. Handling of missing publication times, empty responses, and API failures.
6. Exact training formula calculations (Sent, PV, AI) matching `india_sentiment.py`.
7. Preservation of NaN semantics on no-news and neutral-only stock-days.
8. Compatibility with SPCM fusion matrix and feature column selection.

Note: All tests use deterministic mock providers and offline lexicon scorers;
no network access or paid API keys are required.
"""
import unittest
from datetime import date, datetime, time as dt_time, timedelta
import math
import numpy as np
import pandas as pd
import pytz

from src.utils.indian_news_fetcher import (
    IST_TIMEZONE,
    IndianNewsFetcher,
    MockNewsProvider,
    NewsArticle,
    deduplicate_articles,
    parse_rfc2822_to_ist,
    InvalidTimestampError,
)
from src.features.news_sentiment_inference import (
    NSE_MARKET_CLOSE_TIME,
    EXPECTED_SENTIMENT_FUSION_COLUMNS,
    FULL_DAILY_SENTIMENT_COLUMNS,
    FinancialLexiconSentimentScorer,
    MockSentimentScorer,
    align_timestamp_to_trading_day,
    calculate_spcm_formulas,
    score_articles,
    aggregate_point_in_time_sentiment,
    IndianNewsSentimentPipeline,
)
from src.models.spcm import build_fusion_matrix, get_feature_columns


class TestIndianNewsFetcher(unittest.TestCase):
    """Tests for news article ingestion, parsing, deduplication, and failure handling."""

    def test_parse_rfc2822_to_ist(self):
        """Verify RFC 2822 dates from RSS feeds correctly convert to IST."""
        # 08:35 GMT on Oct 21 -> 14:05 IST on Oct 21 (+5:30)
        rfc_str = "Wed, 21 Oct 2026 08:35:00 GMT"
        ts = parse_rfc2822_to_ist(rfc_str)
        self.assertEqual(ts.tzinfo.zone, "Asia/Kolkata")
        self.assertEqual(ts.hour, 14)
        self.assertEqual(ts.minute, 5)
        self.assertEqual(ts.day, 21)
        self.assertEqual(ts.month, 10)
        self.assertEqual(ts.year, 2026)

    def test_missing_and_invalid_timestamps(self):
        """Ensure missing or corrupt timestamp strings raise InvalidTimestampError."""
        with self.assertRaises(InvalidTimestampError):
            parse_rfc2822_to_ist("")
        with self.assertRaises(InvalidTimestampError):
            parse_rfc2822_to_ist(None)
        with self.assertRaises(InvalidTimestampError):
            parse_rfc2822_to_ist("Corrupted Date String 12345")

    def test_deduplication_exact_url(self):
        """Duplicate URLs must be discarded regardless of title or source."""
        ts = pd.Timestamp("2026-10-21 10:00:00", tz=IST_TIMEZONE)
        art1 = NewsArticle(
            stock="RELIANCE",
            symbol="RELIANCE.NS",
            title="Reliance announces major expansion",
            summary="",
            published_at=ts,
            source="SourceA",
            url="https://news.example.com/art1",
            provider="mock"
        )
        art2 = NewsArticle(
            stock="RELIANCE",
            symbol="RELIANCE.NS",
            title="Different Headline text for same article",
            summary="",
            published_at=ts - pd.Timedelta(minutes=5),
            source="SourceB",
            url="https://news.example.com/art1",  # Same URL
            provider="mock"
        )
        deduped = deduplicate_articles([art1, art2])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].source, "SourceA")  # Newest kept

    def test_deduplication_normalized_headline(self):
        """Near-identical headlines within 48 hours for the same stock must be deduplicated."""
        ts = pd.Timestamp("2026-10-21 11:00:00", tz=IST_TIMEZONE)
        art1 = NewsArticle(
            stock="TCS",
            symbol="TCS.NS",
            title="TCS Signs $500M Mega Deal with European Retailer!",
            summary="",
            published_at=ts,
            source="SourceA",
            url="https://news.example.com/tcs1",
            provider="mock"
        )
        art2 = NewsArticle(
            stock="TCS",
            symbol="TCS.NS",
            title="tcs signs 500m mega deal with european retailer",  # Normalized duplicate
            summary="",
            published_at=ts - pd.Timedelta(hours=4),
            source="SourceB",
            url="https://news.example.com/tcs2",  # Different URL
            provider="mock"
        )
        deduped = deduplicate_articles([art1, art2])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].title, art1.title)

    def test_deduplication_preserves_distinct_stocks(self):
        """Articles with similar headlines for different stocks must NOT be deduplicated."""
        ts = pd.Timestamp("2026-10-21 10:00:00", tz=IST_TIMEZONE)
        art1 = NewsArticle(
            stock="TCS",
            symbol="TCS.NS",
            title="IT stocks rally as quarterly earnings beat expectations",
            summary="",
            published_at=ts,
            source="SourceA",
            url="https://news.example.com/tcs",
            provider="mock"
        )
        art2 = NewsArticle(
            stock="INFY",
            symbol="INFY.NS",
            title="IT stocks rally as quarterly earnings beat expectations",
            summary="",
            published_at=ts,
            source="SourceB",
            url="https://news.example.com/infy",
            provider="mock"
        )
        deduped = deduplicate_articles([art1, art2])
        self.assertEqual(len(deduped), 2)

    def test_empty_news_response_handling(self):
        """An empty provider response must cleanly return an empty list without raising."""
        mock_provider = MockNewsProvider(canned_articles=[])
        fetcher = IndianNewsFetcher(custom_provider=mock_provider)
        res = fetcher.fetch_news_for_symbols(["RELIANCE", "TCS"])
        self.assertEqual(len(res.articles), 0)
        self.assertEqual(res.metadata.total_articles_fetched, 0)
        self.assertIn("RELIANCE.NS", res.metadata.symbols_successful)
        self.assertIn("TCS.NS", res.metadata.symbols_successful)

    def test_provider_failure_resilience(self):
        """A failure on one symbol must be recorded in symbols_failed without halting others."""
        class FailingMockProvider:
            def fetch_news(self, stock_code, symbol, max_articles=20, max_age_days=None):
                if stock_code == "FAIL":
                    raise ConnectionResetError("Remote server disconnected")
                return [
                    NewsArticle(
                        stock=stock_code,
                        symbol=symbol,
                        title=f"{stock_code} normal headline",
                        summary="",
                        published_at=pd.Timestamp("2026-10-21 10:00:00", tz=IST_TIMEZONE),
                        source="Mock",
                        url=f"https://mock.com/{stock_code}",
                        provider="mock"
                    )
                ]

        fetcher = IndianNewsFetcher(custom_provider=FailingMockProvider(), request_delay=0.0)
        res = fetcher.fetch_news_for_symbols(["RELIANCE", "FAIL", "TCS"])
        self.assertEqual(len(res.articles), 2)
        self.assertIn("RELIANCE.NS", res.metadata.symbols_successful)
        self.assertIn("TCS.NS", res.metadata.symbols_successful)
        self.assertIn("FAIL.NS", res.metadata.symbols_failed)


class TestPointInTimeAlignment(unittest.TestCase):
    """Tests for Point-in-Time trading day alignment and absence of future information leakage."""

    def test_intraday_before_close_alignment(self):
        """News published on a weekday before 15:30 IST is applicable to that day."""
        # Wednesday 2026-10-21 at 11:30 AM IST
        ts = pd.Timestamp("2026-10-21 11:30:00", tz=IST_TIMEZONE)
        trade_date = align_timestamp_to_trading_day(ts, cutoff_time=NSE_MARKET_CLOSE_TIME)
        self.assertEqual(trade_date.date(), date(2026, 10, 21))

    def test_intraday_after_close_alignment(self):
        """News published at or after 15:30 IST shifts to next trading day."""
        # Wednesday 2026-10-21 at 15:31:00 IST (market closed) -> Thursday 2026-10-22
        ts = pd.Timestamp("2026-10-21 15:31:00", tz=IST_TIMEZONE)
        trade_date = align_timestamp_to_trading_day(ts, cutoff_time=NSE_MARKET_CLOSE_TIME)
        self.assertEqual(trade_date.date(), date(2026, 10, 22))

    def test_friday_post_market_rolls_to_monday(self):
        """News published on Friday evening must roll past the weekend to Monday."""
        # Friday 2026-10-23 at 18:00:00 IST -> Monday 2026-10-26
        ts = pd.Timestamp("2026-10-23 18:00:00", tz=IST_TIMEZONE)
        trade_date = align_timestamp_to_trading_day(ts, cutoff_time=NSE_MARKET_CLOSE_TIME)
        self.assertEqual(trade_date.date(), date(2026, 10, 26))

    def test_weekend_news_rolls_to_monday(self):
        """News published on Saturday or Sunday must roll to Monday."""
        # Saturday 2026-10-24 at 10:00:00 IST -> Monday 2026-10-26
        ts_sat = pd.Timestamp("2026-10-24 10:00:00", tz=IST_TIMEZONE)
        self.assertEqual(align_timestamp_to_trading_day(ts_sat).date(), date(2026, 10, 26))

        # Sunday 2026-10-25 at 20:00:00 IST -> Monday 2026-10-26
        ts_sun = pd.Timestamp("2026-10-25 20:00:00", tz=IST_TIMEZONE)
        self.assertEqual(align_timestamp_to_trading_day(ts_sun).date(), date(2026, 10, 26))

    def test_holiday_rolling(self):
        """News falling on or shifting to an official market holiday must roll forward."""
        # Wednesday 2026-10-21 post-market -> rolls to Oct 22. But if Oct 22 is a holiday, rolls to Oct 23.
        ts = pd.Timestamp("2026-10-21 16:00:00", tz=IST_TIMEZONE)
        holiday = date(2026, 10, 22)
        trade_date = align_timestamp_to_trading_day(ts, holidays=[holiday])
        self.assertEqual(trade_date.date(), date(2026, 10, 23))

    def test_strict_daily_lag(self):
        """When strict_daily_lag is True, even pre-market news shifts to next trading day."""
        # Wednesday 2026-10-21 at 08:00:00 IST (pre-market)
        ts = pd.Timestamp("2026-10-21 08:00:00", tz=IST_TIMEZONE)
        trade_date = align_timestamp_to_trading_day(ts, strict_daily_lag=True)
        self.assertEqual(trade_date.date(), date(2026, 10, 22))

    def test_absence_of_future_leakage(self):
        """A news item published at timestamp T must never be assigned to any date < T."""
        timestamps = [
            pd.Timestamp("2026-10-19 14:00:00", tz=IST_TIMEZONE),
            pd.Timestamp("2026-10-19 16:00:00", tz=IST_TIMEZONE),
            pd.Timestamp("2026-10-23 19:00:00", tz=IST_TIMEZONE),
            pd.Timestamp("2026-10-25 11:00:00", tz=IST_TIMEZONE),
        ]
        for ts in timestamps:
            aligned_date = align_timestamp_to_trading_day(ts).date()
            self.assertGreaterEqual(aligned_date, ts.date())


class TestSentimentScoringAndFormulas(unittest.TestCase):
    """Tests for sentiment scoring models and exact formula reproduction."""

    def test_lexicon_deterministic_scoring(self):
        """Verify FinancialLexiconSentimentScorer correctly identifies bullish, bearish, and neutral."""
        scorer = FinancialLexiconSentimentScorer()
        texts = [
            "Reliance reports record profit growth and strong revenue beat",
            "TCS shares plunge following huge quarterly loss and debt downgrade",
            "Company schedules annual general meeting for next month",
        ]
        scores = scorer.score_texts(texts)
        self.assertEqual(scores[0]["sentiment_label"], "positive")
        self.assertGreater(scores[0]["positive_prob"], 0.6)

        self.assertEqual(scores[1]["sentiment_label"], "negative")
        self.assertGreater(scores[1]["negative_prob"], 0.6)

        self.assertEqual(scores[2]["sentiment_label"], "neutral")
        self.assertGreater(scores[2]["neutral_prob"], 0.4)

    def test_probability_sums_to_one(self):
        """All probabilities must sum to 1.0 within floating tolerance."""
        scorer = FinancialLexiconSentimentScorer()
        test_headlines = [
            "Quarterly results announced",
            "Profits surge 25%",
            "Severe fraud probe launched by regulators",
            "Market closes flat in quiet session",
        ]
        scores = scorer.score_texts(test_headlines)
        for s in scores:
            total = s["positive_prob"] + s["negative_prob"] + s["neutral_prob"]
            self.assertAlmostEqual(total, 1.0, places=4)
            self.assertGreaterEqual(s["positive_prob"], 0.0)
            self.assertGreaterEqual(s["negative_prob"], 0.0)
            self.assertGreaterEqual(s["neutral_prob"], 0.0)

    def test_spcm_formula_calculations_exact_cases(self):
        """
        Spot check exact mathematical values of formulas:
          Sent = (bull - bear) / (bull + bear)
          PV   = ln(bull + bear + 1)
          AI   = 1 - sqrt(1 - Sent^2)
        """
        # Case 1: bull=5, bear=3, neutral=0
        s, p, a = calculate_spcm_formulas(5, 3, 0)
        self.assertAlmostEqual(s, 0.25)
        self.assertAlmostEqual(p, math.log(9))
        self.assertAlmostEqual(a, 1.0 - math.sqrt(1.0 - 0.25 ** 2))

        # Case 2: All bullish (10, 0, 0)
        s, p, a = calculate_spcm_formulas(10, 0, 0)
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(p, math.log(11))
        self.assertAlmostEqual(a, 1.0)

        # Case 3: All bearish (0, 10, 0)
        s, p, a = calculate_spcm_formulas(0, 10, 0)
        self.assertAlmostEqual(s, -1.0)
        self.assertAlmostEqual(p, math.log(11))
        self.assertAlmostEqual(a, 1.0)

        # Case 4: Equal sentiment (5, 5, 0)
        s, p, a = calculate_spcm_formulas(5, 5, 0)
        self.assertAlmostEqual(s, 0.0)
        self.assertAlmostEqual(p, math.log(11))
        self.assertAlmostEqual(a, 0.0)

        # Case 5: Neutral-only news (0, 0, 10)
        # Sent, PV, AI must be NaN; must NOT be silently treated as 0
        s, p, a = calculate_spcm_formulas(0, 0, 10)
        self.assertTrue(math.isnan(s))
        self.assertTrue(math.isnan(p))
        self.assertTrue(math.isnan(a))

        # Case 6: No news (0, 0, 0)
        s, p, a = calculate_spcm_formulas(0, 0, 0)
        self.assertTrue(math.isnan(s))
        self.assertTrue(math.isnan(p))
        self.assertTrue(math.isnan(a))


class TestSentimentAggregationAndSchema(unittest.TestCase):
    """Tests for daily aggregation, NaN semantics, and SPCM model schema compatibility."""

    def test_daily_aggregation_and_nan_preservation(self):
        """Verify that on no-news days, news_available=0 and Sent/PV/AI remain NaN."""
        # Simulated 2 trading dates for STK1 and STK2
        trading_dates = [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")]
        target_stocks = ["STK1", "STK2"]

        # Only STK1 has news on 2026-10-20 (positive)
        scored_df = pd.DataFrame([
            {
                "stock": "STK1",
                "symbol": "STK1.NS",
                "title": "STK1 reports great earnings",
                "published_at": pd.Timestamp("2026-10-20 10:00:00", tz=IST_TIMEZONE),
                "sentiment_label": "positive",
                "positive_prob": 0.9,
                "negative_prob": 0.05,
                "neutral_prob": 0.05,
            }
        ])

        daily_df = aggregate_point_in_time_sentiment(
            scored_articles=scored_df,
            trading_dates=trading_dates,
            target_stocks=target_stocks,
        )

        self.assertEqual(len(daily_df), 4)  # 2 stocks x 2 dates

        # Check STK1 on 2026-10-20 (has news)
        stk1_d1 = daily_df[(daily_df["stock"] == "STK1") & (daily_df["date"] == pd.Timestamp("2026-10-20"))].iloc[0]
        self.assertEqual(stk1_d1["news_available"], 1)
        self.assertEqual(stk1_d1["headline_count"], 1)
        self.assertEqual(stk1_d1["bullish_count"], 1)
        self.assertAlmostEqual(stk1_d1["Sent"], 1.0)
        self.assertAlmostEqual(stk1_d1["PV"], math.log(2))
        self.assertAlmostEqual(stk1_d1["AI"], 1.0)

        # Check STK1 on 2026-10-21 (no news)
        stk1_d2 = daily_df[(daily_df["stock"] == "STK1") & (daily_df["date"] == pd.Timestamp("2026-10-21"))].iloc[0]
        self.assertEqual(stk1_d2["news_available"], 0)
        self.assertEqual(stk1_d2["headline_count"], 0)
        self.assertTrue(np.isnan(stk1_d2["Sent"]))
        self.assertTrue(np.isnan(stk1_d2["PV"]))
        self.assertTrue(np.isnan(stk1_d2["AI"]))

        # Check STK2 on 2026-10-20 (no news)
        stk2_d1 = daily_df[(daily_df["stock"] == "STK2") & (daily_df["date"] == pd.Timestamp("2026-10-20"))].iloc[0]
        self.assertEqual(stk2_d1["news_available"], 0)
        self.assertTrue(np.isnan(stk2_d1["Sent"]))

    def test_neutral_only_day_behavior(self):
        """When only neutral news is published, news_available=1 but Sent/PV/AI must be NaN."""
        scored_df = pd.DataFrame([
            {
                "stock": "INFY",
                "symbol": "INFY.NS",
                "title": "Infosys holds routine board meeting",
                "published_at": pd.Timestamp("2026-10-21 11:00:00", tz=IST_TIMEZONE),
                "sentiment_label": "neutral",
                "positive_prob": 0.1,
                "negative_prob": 0.1,
                "neutral_prob": 0.8,
            }
        ])

        daily_df = aggregate_point_in_time_sentiment(scored_df)
        self.assertEqual(len(daily_df), 1)
        row = daily_df.iloc[0]
        self.assertEqual(row["news_available"], 1)
        self.assertEqual(row["headline_count"], 1)
        self.assertEqual(row["neutral_count"], 1)
        self.assertEqual(row["bullish_count"], 0)
        self.assertEqual(row["bearish_count"], 0)
        self.assertTrue(np.isnan(row["Sent"]))
        self.assertTrue(np.isnan(row["PV"]))
        self.assertTrue(np.isnan(row["AI"]))

    def test_spcm_fusion_matrix_compatibility(self):
        """Verify aggregated sentiment seamlessly merges with SPCM price features."""
        # Mock price features dataframe (such as returned by build_price_features)
        price_feat = pd.DataFrame({
            "stock": ["RELIANCE", "RELIANCE"],
            "date": [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")],
            "open": [2500.0, 2520.0],
            "close": [2515.0, 2530.0],
            "volume": [100000, 120000],
            "ret_1d": [0.005, 0.006],
        })

        sentiment_ind = pd.DataFrame({
            "stock": ["RELIANCE", "RELIANCE"],
            "date": [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")],
            "Sent": [0.5, np.nan],
            "PV": [math.log(3), np.nan],
            "AI": [1 - math.sqrt(1 - 0.25), np.nan],
            "news_available": [1, 0],
            "headline_count": [2, 0],
        })

        labels = pd.DataFrame({
            "stock": ["RELIANCE", "RELIANCE"],
            "date": [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")],
            "label": [1.0, 0.0],
        })

        fusion = build_fusion_matrix(
            price_feat=price_feat,
            sentiment_ind=sentiment_ind,
            homophily_ind=None,
            labels=labels,
        )

        # Check required columns are present in fusion
        for col in ["Sent", "PV", "AI", "news_available", "headline_count"]:
            self.assertIn(col, fusion.columns)

        # Ensure feature column retrieval works as expected
        feat_cols = get_feature_columns(fusion, include_sentiment=True)
        for col in ["Sent", "PV", "AI", "news_available", "headline_count"]:
            self.assertIn(col, feat_cols)

        # Check that NaN in Sent is preserved without silent zero-fill in fusion
        self.assertTrue(np.isnan(fusion.loc[1, "Sent"]))
        self.assertEqual(fusion.loc[0, "Sent"], 0.5)


if __name__ == "__main__":
    unittest.main()
