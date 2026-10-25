"""
Indian Stock News Fetcher for SPCM Pipeline (Phase 3).

Provides live and historical financial news ingestion for supported Indian stock symbols
(NSE/BSE) using accessible public sources (Google News RSS and Yahoo Finance news).
Enforces strict timestamp extraction in Indian Standard Time (IST), deduplication,
rate-limit polite intervals, error resilience, and source metadata tracking.

DATA SOURCE & FEED INFORMATION:
- Primary Provider: Google News RSS (focused on Indian business/finance queries)
  - Endpoint: https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en
  - API Key: Not required (free public RSS feed)
  - Rate Limits: No published strict limit; polite pacing (0.5s-1.0s delay between queries)
    and browser User-Agent headers are enforced to prevent HTTP 429 throttling.
  - Timestamps: RFC 2822 format (e.g., 'Wed, 21 Oct 2026 08:35:00 GMT') converted to IST (Asia/Kolkata).
- Secondary / Fallback Provider: Yahoo Finance Search News (via yfinance)
  - Endpoint: Yahoo Finance Search API
  - API Key: Not required
  - Rate Limits: Subject to Yahoo Finance scraping rate limits.
  - Timestamps: Unix epoch seconds (providerPublishTime) converted to IST (Asia/Kolkata).
- Classification: LIVE when fetched from online endpoints during the current session,
  or CACHED / HISTORICAL when reading persisted snapshots.
"""
from __future__ import annotations

import argparse
import email.utils
import html
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pytz
import requests

from src.utils.indian_market_data import (
    DEFAULT_INDIAN_SYMBOLS,
    extract_stock_code,
    normalize_symbol,
)

# Logger setup
logger = logging.getLogger("indian_news_fetcher")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

IST_TIMEZONE = pytz.timezone("Asia/Kolkata")

# Default headers simulating a standard browser client
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


class NewsFetchError(Exception):
    """Base exception for news ingestion errors."""
    pass


class ProviderUnavailableError(NewsFetchError):
    """Raised when news provider fails to respond or returns HTTP error."""
    pass


class InvalidTimestampError(NewsFetchError):
    """Raised when publication timestamp is missing or cannot be parsed."""
    pass


@dataclass
class NewsArticle:
    """Represents a single parsed news article for an Indian stock."""
    stock: str                         # Base ticker (e.g. 'RELIANCE')
    symbol: str                        # Exchange symbol (e.g. 'RELIANCE.NS')
    title: str                         # Article headline
    summary: str                       # Snippet or summary
    published_at: pd.Timestamp         # IST timezone-aware Timestamp
    source: str                        # Publisher (e.g. 'Reuters', 'Livemint')
    url: str                           # Article web URL
    provider: str                      # 'google_news', 'yahoo_finance', 'mock', 'cached'
    is_live: bool = True               # True for live network fetch
    raw_timestamp: str = ""            # Original raw timestamp string

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock": self.stock,
            "symbol": self.symbol,
            "title": self.title,
            "summary": self.summary,
            "published_at": self.published_at,
            "published_date": self.published_at.date() if pd.notna(self.published_at) else None,
            "source": self.source,
            "url": self.url,
            "provider": self.provider,
            "is_live": self.is_live,
        }


@dataclass
class NewsFetchMetadata:
    """Metadata regarding news provider, fetch timestamp, and execution status."""
    provider: str = "google_news"
    fetch_timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz=IST_TIMEZONE))
    symbols_requested: list[str] = field(default_factory=list)
    symbols_successful: list[str] = field(default_factory=list)
    symbols_failed: dict[str, str] = field(default_factory=dict)
    total_articles_fetched: int = 0
    total_articles_deduplicated: int = 0
    rate_limit_info: str = "Polite request delays (0.5s) enforced; no API key required."
    api_key_required: bool = False
    notes: str = ""


@dataclass
class NewsFetchResult:
    """Encapsulates list of parsed articles and corresponding fetch metadata."""
    articles: list[NewsArticle]
    metadata: NewsFetchMetadata

    def to_dataframe(self) -> pd.DataFrame:
        if not self.articles:
            return pd.DataFrame(columns=[
                "stock", "symbol", "title", "summary", "published_at",
                "published_date", "source", "url", "provider", "is_live"
            ])
        rows = [a.to_dict() for a in self.articles]
        df = pd.DataFrame(rows)
        return df.sort_values(["stock", "published_at"], ascending=[True, False]).reset_index(drop=True)


def clean_html_text(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html.unescape(clean)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", clean).strip()


def normalize_title_for_dedup(title: str) -> str:
    """Normalize headline string for near-duplicate detection."""
    clean = clean_html_text(title).lower()
    # Remove non-alphanumeric characters
    clean = re.sub(r"[^a-z0-9\s]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


def parse_rfc2822_to_ist(date_str: str) -> pd.Timestamp:
    """
    Parse RFC 2822 date string (from RSS) to IST timezone-aware pd.Timestamp.
    Raises InvalidTimestampError if missing or unparseable.
    """
    if not date_str or not isinstance(date_str, str) or not date_str.strip():
        raise InvalidTimestampError("Empty or missing date string.")
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist_dt = dt.astimezone(IST_TIMEZONE)
        return pd.Timestamp(ist_dt)
    except Exception as exc:
        raise InvalidTimestampError(f"Could not parse date '{date_str}': {exc}") from exc


def deduplicate_articles(
    articles: list[NewsArticle],
    similarity_window_hours: int = 48
) -> list[NewsArticle]:
    """
    Deduplicates news articles using a three-tier safeguard:
      1. Exact URL deduplication.
      2. Exact headline matching per stock.
      3. Normalized headline matching per stock within a rolling time window.
    """
    if not articles:
        return []

    seen_urls: set[str] = set()
    seen_exact_titles: set[tuple[str, str]] = set()  # (stock, clean_title)
    retained_articles: list[NewsArticle] = []

    # Sort newest first for priority
    sorted_articles = sorted(
        articles,
        key=lambda a: a.published_at if pd.notna(a.published_at) else pd.Timestamp.min.tz_localize(IST_TIMEZONE),
        reverse=True
    )

    for art in sorted_articles:
        # Tier 1: Check URL
        url = (art.url or "").strip()
        if url and url in seen_urls:
            continue

        # Tier 2: Check exact headline for stock
        title_norm = normalize_title_for_dedup(art.title)
        if not title_norm:
            continue
        key_exact = (art.stock, title_norm)
        if key_exact in seen_exact_titles:
            continue

        # Tier 3: Check near-duplicate in time window
        is_dup = False
        for retained in retained_articles:
            if retained.stock == art.stock:
                other_norm = normalize_title_for_dedup(retained.title)
                # If titles match normalized
                if other_norm == title_norm:
                    time_diff = abs((art.published_at - retained.published_at).total_seconds())
                    if time_diff <= similarity_window_hours * 3600:
                        is_dup = True
                        break

        if is_dup:
            continue

        if url:
            seen_urls.add(url)
        seen_exact_titles.add(key_exact)
        retained_articles.append(art)

    return retained_articles


class GoogleNewsRSSProvider:
    """
    Ingests live Indian financial news via Google News RSS search.
    Requires no API keys, provides live updates with exact RFC 2822 timestamps,
    and returns source attribution.
    """

    BASE_URL = "https://news.google.com/rss/search"

    def __init__(self, timeout: float = 10.0, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    def fetch_news(
        self,
        stock_code: str,
        symbol: str,
        max_articles: int = 20,
        max_age_days: int | None = 14,
    ) -> list[NewsArticle]:
        """
        Fetch news articles for a given stock code and exchange symbol.
        """
        # Tailor query to Indian business/stock news
        query = f"{stock_code} stock NSE India"
        params = {
            "q": query,
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en",
        }

        articles: list[NewsArticle] = []
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    headers=DEFAULT_HEADERS,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    articles = self._parse_rss(response.content, stock_code, symbol)
                    break
                elif response.status_code == 429:
                    logger.warning(f"Google News rate limited (429) for '{stock_code}', retry {attempt}/{self.max_retries}")
                    time.sleep(1.0 * attempt)
                    last_error = ProviderUnavailableError("Google News rate limit (HTTP 429).")
                else:
                    last_error = ProviderUnavailableError(f"Google News returned HTTP {response.status_code}")
                    time.sleep(0.5)
            except Exception as exc:
                last_error = exc
                logger.warning(f"Error fetching Google News for '{stock_code}' (attempt {attempt}): {exc}")
                time.sleep(0.5 * attempt)

        if not articles and last_error and not isinstance(last_error, ProviderUnavailableError):
            logger.warning(f"Google News failed for '{stock_code}': {last_error}")

        # Filter by max_age_days if provided
        if max_age_days is not None and articles:
            cutoff_time = pd.Timestamp.now(tz=IST_TIMEZONE) - pd.Timedelta(days=max_age_days)
            articles = [a for a in articles if a.published_at >= cutoff_time]

        return articles[:max_articles]

    def _parse_rss(self, content: bytes, stock_code: str, symbol: str) -> list[NewsArticle]:
        try:
            root = ET.fromstring(content)
        except Exception as exc:
            logger.error(f"Failed to parse XML response for '{stock_code}': {exc}")
            return []

        items = root.findall("./channel/item")
        articles: list[NewsArticle] = []

        for item in items:
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_elem = item.find("pubDate")
            source_elem = item.find("source")
            desc_elem = item.find("description")

            title = clean_html_text(title_elem.text if title_elem is not None else "")
            link = (link_elem.text or "").strip() if link_elem is not None else ""
            pub_str = (pub_elem.text or "").strip() if pub_elem is not None else ""
            source = clean_html_text(source_elem.text if source_elem is not None else "Google News")
            summary = clean_html_text(desc_elem.text if desc_elem is not None else "")

            if not title:
                continue

            # Strip publisher from Google News title suffix if formatted as "Headline - Source"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                if source == "Google News" and len(parts) > 1:
                    source = parts[1].strip()

            if not pub_str:
                # Requirement: Do not fabricate timestamps. Skip articles with missing timestamps.
                logger.debug(f"Skipping article '{title[:40]}' due to missing publication time.")
                continue

            try:
                published_at = parse_rfc2822_to_ist(pub_str)
            except InvalidTimestampError as exc:
                logger.debug(f"Skipping article due to invalid timestamp '{pub_str}': {exc}")
                continue

            article = NewsArticle(
                stock=stock_code,
                symbol=symbol,
                title=title,
                summary=summary,
                published_at=published_at,
                source=source,
                url=link,
                provider="google_news",
                is_live=True,
                raw_timestamp=pub_str,
            )
            articles.append(article)

        return articles


class YahooFinanceNewsProvider:
    """
    Fetches news via yfinance search API as an alternate/fallback provider.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_news(
        self,
        stock_code: str,
        symbol: str,
        max_articles: int = 20,
        max_age_days: int | None = 14,
    ) -> list[NewsArticle]:
        try:
            import yfinance as yf
            # Try yf.Search first
            search = yf.Search(stock_code)
            raw_news = getattr(search, "news", [])
            if not raw_news:
                ticker = yf.Ticker(symbol)
                raw_news = getattr(ticker, "news", [])
        except Exception as exc:
            logger.warning(f"Yahoo Finance news fetch failed for '{symbol}': {exc}")
            return []

        if not raw_news:
            return []

        articles: list[NewsArticle] = []
        for item in raw_news:
            if not isinstance(item, dict):
                continue
            title = clean_html_text(item.get("title", ""))
            link = item.get("link", "")
            publisher = item.get("publisher", "Yahoo Finance")
            epoch_time = item.get("providerPublishTime")

            if not title or epoch_time is None:
                continue

            try:
                dt_utc = datetime.fromtimestamp(int(epoch_time), tz=timezone.utc)
                published_at = pd.Timestamp(dt_utc.astimezone(IST_TIMEZONE))
            except Exception:
                continue

            article = NewsArticle(
                stock=stock_code,
                symbol=symbol,
                title=title,
                summary=title,
                published_at=published_at,
                source=publisher,
                url=link,
                provider="yahoo_finance",
                is_live=True,
                raw_timestamp=str(epoch_time),
            )
            articles.append(article)

        if max_age_days is not None and articles:
            cutoff = pd.Timestamp.now(tz=IST_TIMEZONE) - pd.Timedelta(days=max_age_days)
            articles = [a for a in articles if a.published_at >= cutoff]

        return articles[:max_articles]


class MockNewsProvider:
    """
    Deterministic mock provider for unit testing without network or external dependencies.
    """

    def __init__(self, canned_articles: list[dict[str, Any]] | None = None):
        self.canned_articles = canned_articles or []

    def fetch_news(
        self,
        stock_code: str,
        symbol: str,
        max_articles: int = 20,
        max_age_days: int | None = None,
    ) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for d in self.canned_articles:
            if d.get("stock") == stock_code or d.get("symbol") == symbol:
                ts = d.get("published_at")
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(IST_TIMEZONE)
                else:
                    ts = ts.tz_convert(IST_TIMEZONE)

                art = NewsArticle(
                    stock=stock_code,
                    symbol=symbol,
                    title=d.get("title", ""),
                    summary=d.get("summary", d.get("title", "")),
                    published_at=pd.Timestamp(ts),
                    source=d.get("source", "MockNews"),
                    url=d.get("url", f"https://mocknews.local/{len(articles)}"),
                    provider="mock",
                    is_live=False,
                    raw_timestamp=str(d.get("published_at", "")),
                )
                articles.append(art)
        return articles[:max_articles]


class IndianNewsFetcher:
    """
    Reusable, resilient news ingestion manager for Indian equities.
    Integrates primary (Google News RSS) and secondary (Yahoo Finance) providers,
    enforcing deduplication, rate limiting, and Indian Standard Time validation.
    """

    def __init__(
        self,
        primary_provider: str = "google_news",
        enable_fallback: bool = True,
        request_delay: float = 0.5,
        timeout: float = 10.0,
        custom_provider: Any = None,
    ):
        self.primary_provider = primary_provider
        self.enable_fallback = enable_fallback
        self.request_delay = request_delay
        self.timeout = timeout
        self.custom_provider = custom_provider

        self._google_provider = GoogleNewsRSSProvider(timeout=timeout)
        self._yahoo_provider = YahooFinanceNewsProvider(timeout=timeout)

    def fetch_news_for_symbol(
        self,
        symbol: str,
        max_articles: int = 20,
        max_age_days: int | None = 14,
    ) -> list[NewsArticle]:
        """
        Fetch news articles for a single Indian stock symbol.
        """
        stock_code = extract_stock_code(symbol)
        norm_sym = normalize_symbol(symbol)

        if self.custom_provider is not None:
            return self.custom_provider.fetch_news(
                stock_code=stock_code,
                symbol=norm_sym,
                max_articles=max_articles,
                max_age_days=max_age_days,
            )

        articles: list[NewsArticle] = []

        if self.primary_provider == "google_news":
            articles = self._google_provider.fetch_news(
                stock_code=stock_code,
                symbol=norm_sym,
                max_articles=max_articles,
                max_age_days=max_age_days,
            )
            # If Google News returned nothing and fallback enabled, try Yahoo Finance
            if not articles and self.enable_fallback:
                logger.info(f"Trying Yahoo Finance fallback for '{stock_code}'...")
                articles = self._yahoo_provider.fetch_news(
                    stock_code=stock_code,
                    symbol=norm_sym,
                    max_articles=max_articles,
                    max_age_days=max_age_days,
                )
        elif self.primary_provider == "yahoo_finance":
            articles = self._yahoo_provider.fetch_news(
                stock_code=stock_code,
                symbol=norm_sym,
                max_articles=max_articles,
                max_age_days=max_age_days,
            )
            if not articles and self.enable_fallback:
                articles = self._google_provider.fetch_news(
                    stock_code=stock_code,
                    symbol=norm_sym,
                    max_articles=max_articles,
                    max_age_days=max_age_days,
                )
        else:
            raise ValueError(f"Unknown news provider: {self.primary_provider}")

        return deduplicate_articles(articles)

    def fetch_news_for_symbols(
        self,
        symbols: Sequence[str] | str,
        max_articles_per_symbol: int = 20,
        max_age_days: int | None = 14,
    ) -> NewsFetchResult:
        """
        Fetch news articles for multiple Indian stock symbols with polite pacing and deduplication.
        """
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.replace(",", " ").split() if s.strip()]

        symbols_list = list(symbols) if symbols else list(DEFAULT_INDIAN_SYMBOLS)
        formatted_symbols = [normalize_symbol(s) for s in symbols_list]

        metadata = NewsFetchMetadata(
            provider=self.primary_provider,
            fetch_timestamp=pd.Timestamp.now(tz=IST_TIMEZONE),
            symbols_requested=formatted_symbols,
            symbols_successful=[],
            symbols_failed={},
            api_key_required=False,
            notes=(
                "Live Indian news fetched without API keys. "
                "Publication timestamps normalized to Indian Standard Time (IST, UTC+05:30)."
            )
        )

        all_articles: list[NewsArticle] = []
        raw_count = 0

        for i, sym in enumerate(formatted_symbols):
            try:
                arts = self.fetch_news_for_symbol(
                    symbol=sym,
                    max_articles=max_articles_per_symbol,
                    max_age_days=max_age_days,
                )
                raw_count += len(arts)
                all_articles.extend(arts)
                metadata.symbols_successful.append(sym)
                logger.info(f"[{sym}] Retrieved {len(arts)} news articles.")
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(f"[{sym}] Failed to fetch news: {err_msg}")
                metadata.symbols_failed[sym] = err_msg

            # Polite request delay between symbols
            if i < len(formatted_symbols) - 1 and self.request_delay > 0:
                time.sleep(self.request_delay)

        # Global deduplication across all items
        deduped = deduplicate_articles(all_articles)
        metadata.total_articles_fetched = raw_count
        metadata.total_articles_deduplicated = len(deduped)

        return NewsFetchResult(articles=deduped, metadata=metadata)


def cli_test_news():
    """Command-line utility to test and inspect Indian stock news ingestion."""
    # Ensure Windows console can safely print UTF-8 characters (like quotes or currency symbols)
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Phase 3: Reusable Indian Stock News Fetcher (Google News RSS / Yahoo Finance)."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE", "TCS", "INFY"],
        help="Stock symbols to fetch (e.g. RELIANCE TCS INFY or RELIANCE.NS TCS.NS INFY.NS)"
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
        "--out",
        type=str,
        default=None,
        help="Optional path to save fetched articles to CSV"
    )

    args = parser.parse_args()

    print("=" * 75)
    print(" INDIAN STOCK NEWS INGESTION - PHASE 3")
    print("=" * 75)

    fetcher = IndianNewsFetcher()
    result = fetcher.fetch_news_for_symbols(
        symbols=args.symbols,
        max_articles_per_symbol=args.max_articles,
        max_age_days=args.max_age_days
    )

    meta = result.metadata
    print("\n[DATA SOURCE & FEED STATUS]")
    print(f"  Primary Provider:       {meta.provider}")
    print(f"  API Key Required:       {meta.api_key_required} (Free public RSS/Search)")
    print(f"  Rate Limit Safeguards:  {meta.rate_limit_info}")
    print(f"  Fetch Timestamp (IST):  {meta.fetch_timestamp}")
    print(f"  Symbols Requested:      {meta.symbols_requested}")
    print(f"  Symbols Successful:     {meta.symbols_successful}")
    if meta.symbols_failed:
        print(f"  Symbols Failed:         {meta.symbols_failed}")
    print(f"  Raw Articles Fetched:   {meta.total_articles_fetched}")
    print(f"  After Deduplication:    {meta.total_articles_deduplicated}")

    df = result.to_dataframe()
    if df.empty:
        print("\n[!] No articles retrieved for the requested symbols.")
    else:
        print("\n[FETCHED HEADLINES]")
        for stock, grp in df.groupby("stock"):
            print(f"\n--- {stock} ({len(grp)} articles) ---")
            for _, row in grp.head(args.max_articles).iterrows():
                # Safe print handling for Windows consoles
                title_safe = str(row['title']).encode("ascii", errors="replace").decode()
                source_safe = str(row['source']).encode("ascii", errors="replace").decode()
                print(f"  [{row['published_at']}] [{source_safe}] {title_safe}")
                print(f"    URL: {row['url'][:80]}...")

    if args.out and not df.empty:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n[OUTPUT] Saved articles to {out_path}")

    print("=" * 75)


if __name__ == "__main__":
    cli_test_news()
