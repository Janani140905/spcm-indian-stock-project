"""
Test the exact news endpoint logic with row iteration and proper columns.
"""
import pandas as pd
from src.utils.indian_news_fetcher import IndianNewsFetcher
from src.features.news_sentiment_inference import score_articles

def test_fixed_news():
    fetcher = IndianNewsFetcher()
    target_symbols = ["RELIANCE", "TCS"]
    fetch_res = fetcher.fetch_news_for_symbols(target_symbols, max_articles_per_symbol=2)
    scored = score_articles(fetch_res.articles)
    
    formatted_articles = []
    for _, row in scored.head(10).iterrows():
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
        
    print(f"Success! Formatted {len(formatted_articles)} articles:")
    for a in formatted_articles:
        print(" ", a["symbol"], ":", a["title"][:50], "->", a["sentiment_label"], "(", a["sentiment_score"], ")")

if __name__ == "__main__":
    test_fixed_news()
