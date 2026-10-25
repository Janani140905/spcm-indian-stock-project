"""
Data loading utilities for the SPCM pipeline.

Fill in the TODO sections with your actual data sources (Guba-equivalent scraper,
Wind/Tushare/AkShare/yfinance for prices, CSMAR/Ken-French library for FF5 factors).
Everything downstream (features, models, backtest) only depends on the DataFrame
*schemas* documented below, not on where the data came from.

Standard schemas
-----------------
posts_df (raw social media posts):
    thread_id : str
    message_id : str
    stock : str        # ticker / stock code
    author : str        # user id
    post_time : datetime64
    text : str

prices_df (daily OHLCV + fundamentals):
    stock : str
    date : datetime64
    open, high, low, close : float
    volume : float                 # shares traded
    turnover : float                # transaction_amount / market_cap
    market_cap : float
    shares_outstanding : float
    pe, pcf, nav, income_per_share : float   # optional, used for Appendix Table 6 features
    liquidity : float               # current assets / current liabilities

ff5_df (Fama-French 5 factor daily/monthly series):
    date : datetime64
    mkt_rf, smb, hml, rmw, cma, rf : float

fundamentals_df (firm characteristics, e.g. quarterly, forward-filled to daily):
    stock : str
    date : datetime64
    size : float        # ln(market cap) typically
    bm : float           # book-to-market
    roa : float
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TARGET_STOCKS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "KOTAKBANK",
]
INDIAN_STOCKS = TARGET_STOCKS


def normalize_indian_symbol(symbol: str) -> str:
    """Normalize NSE/Yahoo ticker symbols like 'RELIANCE.NS' -> 'RELIANCE'."""
    if pd.isna(symbol):
        return symbol
    s = str(symbol).strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s


def load_posts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["post_time"])
    # Relax validation for India (where thread/author might be missing)
    required_base = {"stock", "post_time", "text"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"posts file missing critical columns: {missing}")
    df["stock"] = df["stock"].apply(normalize_indian_symbol)
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"stock", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices file missing columns: {missing}")
    
    df["stock"] = df["stock"].apply(normalize_indian_symbol)
    
    # Validation checks
    if df.duplicated(["stock", "date"]).any():
        raise ValueError("Duplicate stock-date entries found in prices.")
    if df[["open", "high", "low", "close"]].isnull().any().any():
        raise ValueError("Missing OHLC values found.")
    
    return df


def load_ff5_factors(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() and p.name == "ff5_factors.csv":
        alt = p.parent / "ff4_factors.csv"
        if alt.exists():
            p = alt
    df = pd.read_csv(p, parse_dates=["date"])
    # Require at least MKT, SMB, HML (allow 3-factor, 4-factor IIMA, or 5-factor adaptations)
    required = {"date", "mkt_rf", "smb", "hml", "rf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"factors file missing base columns: {missing}")
    if df.duplicated("date").any():
        raise ValueError("Duplicate dates found in factors.")
    return df


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame() # Optional for baseline predictive model
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"stock", "date", "size", "bm", "roa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fundamentals file missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Synthetic data generator, used by `python main.py --demo` so the pipeline is
# runnable end-to-end without real data.
# ---------------------------------------------------------------------------
def make_synthetic_dataset(
    n_stocks: int = 15,
    start: str = "2018-01-01",
    end: str = "2020-12-31",
    posts_per_stock_day: float = 8.0,
    seed: int = 42,
    out_dir: str | Path = DATA_DIR / "raw",
):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(start, end)
    stocks = [f"STK{i:03d}" for i in range(n_stocks)]

    # ---- prices: correlated random walk with drift ----
    price_rows = []
    for s in stocks:
        drift = rng.normal(0.0002, 0.0003)
        vol = rng.uniform(0.01, 0.03)
        px = 50 + rng.normal(0, 5)
        mcap0 = rng.uniform(5e9, 5e11)
        for d in dates:
            ret = rng.normal(drift, vol)
            px = max(px * (1 + ret), 1.0)
            o = px * (1 + rng.normal(0, 0.002))
            h = max(o, px) * (1 + abs(rng.normal(0, 0.003)))
            l = min(o, px) * (1 - abs(rng.normal(0, 0.003)))
            vol_traded = rng.uniform(1e5, 5e6)
            price_rows.append(dict(
                stock=s, date=d, open=o, high=h, low=l, close=px,
                volume=vol_traded, turnover=vol_traded * px / mcap0,
                market_cap=mcap0, shares_outstanding=mcap0 / px,
                pe=rng.uniform(5, 40), pcf=rng.uniform(2, 20),
                nav=rng.uniform(1, 10), income_per_share=rng.uniform(0.1, 3),
                liquidity=rng.uniform(0.8, 3.0),
            ))
    prices_df = pd.DataFrame(price_rows)
    prices_df.to_csv(out_dir / "prices.csv", index=False)

    # ---- posts: sentiment-laden text correlated with next-day return sign ----
    bullish_templates = [
        "This stock is going to the moon!", "Strong buy, great fundamentals.",
        "Loading up on shares today.", "Earnings beat expectations, bullish!",
    ]
    bearish_templates = [
        "This stock is crashing, sell now.", "Terrible earnings, avoid.",
        "I'm dumping my shares.", "Overvalued, expect a correction.",
    ]
    neutral_templates = [
        "Watching this stock closely.", "Any thoughts on this company?",
        "Volume looks average today.", "Waiting for more information.",
    ]
    prices_indexed = prices_df.set_index(["stock", "date"])["close"]
    post_rows = []
    msg_id = 0
    n_users = 500
    users = [f"user_{u}" for u in range(n_users)]
    for s in stocks:
        for d in dates:
            n_posts = rng.poisson(posts_per_stock_day)
            try:
                nxt_idx = dates.get_loc(d) + 1
                fwd_ret = (prices_indexed[(s, dates[nxt_idx])] / prices_indexed[(s, d)] - 1) if nxt_idx < len(dates) else 0
            except KeyError:
                fwd_ret = 0
            bullish_bias = 0.5 + np.clip(fwd_ret * 20, -0.3, 0.3)
            for _ in range(n_posts):
                r = rng.random()
                if r < bullish_bias * 0.7:
                    text = rng.choice(bullish_templates)
                elif r < bullish_bias * 0.7 + (1 - bullish_bias) * 0.7:
                    text = rng.choice(bearish_templates)
                else:
                    text = rng.choice(neutral_templates)
                msg_id += 1
                thread_bucket = rng.integers(0, 5)  # multiple threads per stock-day so homophily is computable
                post_rows.append(dict(
                    thread_id=f"T{s}_{d.date()}_{thread_bucket}", message_id=f"M{msg_id}",
                    stock=s, author=rng.choice(users), post_time=d, text=text,
                ))
    posts_df = pd.DataFrame(post_rows)
    posts_df.to_csv(out_dir / "posts.csv", index=False)

    # ---- FF5 factors: synthetic daily series ----
    ff5_rows = []
    for d in dates:
        ff5_rows.append(dict(
            date=d,
            mkt_rf=rng.normal(0.0003, 0.01),
            smb=rng.normal(0.0, 0.005),
            hml=rng.normal(0.0, 0.005),
            rmw=rng.normal(0.0, 0.004),
            cma=rng.normal(0.0, 0.004),
            rf=0.00008,
        ))
    ff5_df = pd.DataFrame(ff5_rows)
    ff5_df.to_csv(out_dir / "ff5_factors.csv", index=False)

    # ---- fundamentals: quarterly, forward filled conceptually (here just daily synthetic) ----
    fund_rows = []
    for s in stocks:
        size0 = rng.uniform(20, 26)
        bm0 = rng.uniform(0.2, 1.5)
        roa0 = rng.uniform(0.01, 0.15)
        for d in dates:
            fund_rows.append(dict(
                stock=s, date=d,
                size=size0 + rng.normal(0, 0.01),
                bm=bm0 + rng.normal(0, 0.01),
                roa=roa0 + rng.normal(0, 0.001),
            ))
    fundamentals_df = pd.DataFrame(fund_rows)
    fundamentals_df.to_csv(out_dir / "fundamentals.csv", index=False)

    # ---- small labeled subsample for BERT fine-tuning ----
    label_map = {}
    for t in bullish_templates:
        label_map[t] = 1
    for t in bearish_templates:
        label_map[t] = -1
    for t in neutral_templates:
        label_map[t] = 0
    sample = posts_df.sample(min(2000, len(posts_df)), random_state=seed)
    labeled = sample[["text"]].copy()
    labeled["label"] = labeled["text"].map(label_map)
    (out_dir.parent / "labeled").mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_dir.parent / "labeled" / "annotated_posts.csv", index=False)

    print(f"Synthetic dataset written to {out_dir} (+ labeled/annotated_posts.csv)")
    return dict(prices=prices_df, posts=posts_df, ff5=ff5_df, fundamentals=fundamentals_df, labeled=labeled)


def make_indian_synthetic_dataset(
    n_stocks: int = 15,
    start: str = "2018-01-01",
    end: str = "2020-12-31",
    posts_per_stock_day: float = 8.0,
    seed: int = 42,
    out_dir: str | Path = DATA_DIR / "raw" / "india_synthetic",
):
    """
    Generate synthetic dataset mimicking Indian market conditions to validate the pipeline:
    - NIFTY 50 universe (symbols end in .NS)
    - Social data lacks author/thread_id (forces pipeline to skip Homophily)
    - Price data lacks pe, pcf, turnover, liquidity (mimics Yahoo Finance basic OHLCV limitation)
    - Factors simulate IIMA 4-factor model (MKT, SMB, HML, WML) rather than FF5.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(start, end)
    stocks = [normalize_indian_symbol(s) for s in TARGET_STOCKS[:n_stocks]]

    # ---- prices: basic OHLCV only ----
    price_rows = []
    for s in stocks:
        px = 1000 + rng.normal(0, 50)
        mcap0 = rng.uniform(5e10, 5e11)
        for d in dates:
            ret = rng.normal(0.0005, 0.02)
            px = max(px * (1 + ret), 1.0)
            o = px * (1 + rng.normal(0, 0.005))
            h = max(o, px) * (1 + abs(rng.normal(0, 0.005)))
            l = min(o, px) * (1 - abs(rng.normal(0, 0.005)))
            vol_traded = rng.uniform(5e5, 2e7)
            price_rows.append(dict(
                stock=s, date=d, open=o, high=h, low=l, close=px, volume=vol_traded
            ))
    prices_df = pd.DataFrame(price_rows)
    prices_df.to_csv(out_dir / "prices.csv", index=False)

    # ---- posts: no author or thread id ----
    bullish_templates = ["Great earnings by company!", "Buying more today.", "Market looks positive."]
    bearish_templates = ["Selling my shares.", "Overvalued stock.", "Expect a crash soon."]
    neutral_templates = ["Watching this one.", "Volume is low.", "Any news?"]
    
    post_rows = []
    for s in stocks:
        for d in dates:
            n_posts = rng.poisson(posts_per_stock_day)
            for _ in range(n_posts):
                r = rng.random()
                text = rng.choice(bullish_templates) if r < 0.4 else (rng.choice(bearish_templates) if r < 0.8 else rng.choice(neutral_templates))
                post_rows.append(dict(stock=s, post_time=d, text=text))
    posts_df = pd.DataFrame(post_rows)
    posts_df.to_csv(out_dir / "posts.csv", index=False)

    # ---- 4-factor series (IIMA style) ----
    ff5_rows = []
    for d in dates:
        ff5_rows.append(dict(
            date=d,
            mkt_rf=rng.normal(0.0005, 0.015),
            smb=rng.normal(0.0, 0.008),
            hml=rng.normal(0.0, 0.008),
            wml=rng.normal(0.0, 0.008), # Momentum instead of RMW/CMA
            rf=0.00015,
        ))
    ff5_df = pd.DataFrame(ff5_rows)
    ff5_df.to_csv(out_dir / "ff5_factors.csv", index=False)

    print(f"Synthetic Indian dataset written to {out_dir}")
    return dict(prices=prices_df, posts=posts_df, ff5=ff5_df)


# ---------------------------------------------------------------------------
# Data Quality & Validation Reports
# ---------------------------------------------------------------------------
def find_common_date_range(prices: pd.DataFrame, posts: pd.DataFrame, ff5: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    p_dates = set(prices["date"].dt.normalize())
    s_dates = set(pd.to_datetime(posts["post_time"]).dt.normalize())
    f_dates = set(ff5["date"].dt.normalize())
    
    common = sorted(list(p_dates & s_dates & f_dates))
    if not common:
        raise ValueError("CRITICAL: No common dates found across prices, posts, and factors. Cannot proceed.")
    
    return common[0], common[-1], len(common)


def validate_data_quality(prices: pd.DataFrame, posts: pd.DataFrame, ff5: pd.DataFrame):
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    
    # Prices
    p_stocks = prices["stock"].nunique()
    p_days = prices["date"].nunique()
    p_missing_ohlcv = prices[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
    p_dupes = prices.duplicated(["stock", "date"]).sum()
    print(f"[Prices] Stocks: {p_stocks}, Trading Days: {p_days}, Missing OHLCV: {p_missing_ohlcv}, Dupes: {p_dupes}")
    
    # Posts
    s_rows = len(posts)
    s_stocks = posts["stock"].nunique()
    s_authors = posts["author"].nunique() if "author" in posts.columns else 0
    s_missing_text = posts["text"].isnull().sum()
    s_dupes = posts.duplicated(["stock", "post_time", "text"]).sum()
    print(f"[Posts] Rows: {s_rows}, Unique Stocks: {s_stocks}, Authors: {s_authors}")
    print(f"[Posts] Missing Text: {s_missing_text}, Duplicates: {s_dupes}")
    
    if "author" not in posts.columns or "thread_id" not in posts.columns:
        print("[!] Homophily unavailable: social dataset lacks user/thread-level information.")
        
    # Factors
    f_days = ff5["date"].nunique()
    f_dupes = ff5.duplicated("date").sum()
    print(f"[Factors] Days: {f_days}, Dupes: {f_dupes}")
    
    # Check factor units (decimals vs percentages)
    numeric_cols = [c for c in ff5.columns if c != "date"]
    max_val = ff5[numeric_cols].abs().max().max()
    if max_val > 10.0:
        print("[WARNING] Factors appear to be in percentages (max > 10). They may need conversion to decimals.")
    elif max_val > 1.0:
        print("[WARNING] Factors max value > 1.0 but < 10. Please verify if they are percentages or decimals.")
    else:
        print("[Factors] Units appear to be decimals (expected).")
        
    # Dates
    start, end, n_common = find_common_date_range(prices, posts, ff5)
    print(f"[Intersection] Common Date Range: {start.date()} to {end.date()} ({n_common} days)")
    print("="*50 + "\n")

