"""
Construct the homophily indicator (paper Sec III.C.2, eq. 4):

    Homo_{i,s,t} = OC^NC_{i,s,t} - OC^IC_{i,s,t}

where OC ("opinion distance") is the absolute difference between an investor's own
sentiment and the average sentiment of a discussion thread; OC^IC is computed over the
threads the investor actually replied to ("in choice"), OC^NC over the threads on their
screen that they did NOT reply to ("not chosen").

NOTE ON APPROXIMATION: the paper defines the choice set as the ~80 threads shown on Guba's
first screen at the moment of posting. Most public datasets don't preserve "what was on
screen but not clicked" — only what was posted. We approximate the choice set, per
stock-day, as all *active* threads that day (i.e., threads with >=1 post from any user);
"chosen" = threads the user posted in that day, "not chosen" = the remaining active threads.
If you have real impression/view-list data, replace `_daily_choice_sets` accordingly.

Homo is computed at the user-stock-day level then aggregated to stock-day level
(Homo_{s,t}) as the paper's regression uses a stock-horizon aggregate, and further
smoothed with a trailing 2-week (10 trading day) window, per the paper's stated design
("we focus on two weeks before the target day to observe the recent homophily").
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _thread_avg_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment per thread per day (thread-level opinion)."""
    return (
        posts.groupby(["stock", "date", "thread_id"])["sentiment"]
        .mean()
        .rename("thread_sent")
        .reset_index()
    )


def compute_user_stock_day_homophily(posts_with_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    posts_with_sentiment : DataFrame with columns
        stock, thread_id, author, post_time, sentiment (in {-1,0,1})

    Returns
    -------
    DataFrame with columns: stock, date, author, Homo (user-stock-day level)
    """
    df = posts_with_sentiment.copy()
    df["date"] = pd.to_datetime(df["post_time"]).dt.normalize()
    df["sentiment"] = df["sentiment"].astype(float)

    thread_sent = _thread_avg_sentiment(df)

    records = []
    for (stock, date), day_df in df.groupby(["stock", "date"]):
        threads_today = thread_sent[(thread_sent["stock"] == stock) & (thread_sent["date"] == date)]
        if threads_today.empty:
            continue
        thread_map = dict(zip(threads_today["thread_id"], threads_today["thread_sent"]))
        all_threads = set(thread_map.keys())

        # each user's own sentiment that day = mean of their own posts' sentiment
        user_sent = day_df.groupby("author")["sentiment"].mean()
        user_threads = day_df.groupby("author")["thread_id"].apply(lambda x: set(x))

        for author, own_sent in user_sent.items():
            chosen = user_threads.get(author, set())
            not_chosen = all_threads - chosen
            if not chosen or not not_chosen:
                continue  # need both sets non-empty to define Homo
            oc_ic = np.mean([abs(own_sent - thread_map[t]) for t in chosen])
            oc_nc = np.mean([abs(own_sent - thread_map[t]) for t in not_chosen])
            homo = oc_nc - oc_ic
            records.append(dict(stock=stock, date=date, author=author, Homo=homo))

    return pd.DataFrame(records)


def aggregate_stock_homophily(user_homo_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Aggregate user-stock-day homophily to a stock-day level Homo_{s,t}, then apply a
    trailing `window`-trading-day rolling mean (paper: two weeks ~= 10 trading days).
    """
    daily = (
        user_homo_df.groupby(["stock", "date"])["Homo"]
        .mean()
        .rename("Homo_raw")
        .reset_index()
        .sort_values(["stock", "date"])
    )
    daily["Homo"] = (
        daily.groupby("stock")["Homo_raw"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return daily[["stock", "date", "Homo"]]


def _cli():
    from src.utils.data_loader import DATA_DIR
    posts_path = DATA_DIR / "processed" / "posts_with_sentiment.csv"
    if not posts_path.exists():
        raise FileNotFoundError(f"{posts_path} not found. Run bert_sentiment.py predict first.")
    posts = pd.read_csv(posts_path, parse_dates=["post_time"])

    user_homo = compute_user_stock_day_homophily(posts)
    stock_homo = aggregate_stock_homophily(user_homo)

    out_path = DATA_DIR / "processed" / "homophily.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stock_homo.to_csv(out_path, index=False)
    print(f"Wrote homophily indicators to {out_path} ({len(stock_homo)} rows)")


if __name__ == "__main__":
    _cli()
