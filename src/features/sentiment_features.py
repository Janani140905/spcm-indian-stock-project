"""
Construct daily stock-level sentiment indicators from BERT-labeled posts.
Implements paper equations (1)-(3):

    Sent_{s,t} = (N_bull - N_bear) / (N_bull + N_bear)                      (1)
    PV_{s,t}   = ln(N_bull + N_bear + 1)                                    (2)
    AI_{s,t}   = 1 - sqrt(1 - Sent_{s,t}^2)                                 (3)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def build_sentiment_indicators(posts_with_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    posts_with_sentiment : DataFrame with columns
        stock, post_time (datetime), sentiment (in {-1, 0, 1})

    Returns
    -------
    DataFrame indexed by (stock, date) with columns: Sent, PV, AI, N_bull, N_bear, N_neutral
    """
    df = posts_with_sentiment.copy()
    df["date"] = pd.to_datetime(df["post_time"]).dt.normalize()

    grp = df.groupby(["stock", "date"])
    n_bull = grp.apply(lambda g: (g["sentiment"] == 1).sum())
    n_bear = grp.apply(lambda g: (g["sentiment"] == -1).sum())
    n_neutral = grp.apply(lambda g: (g["sentiment"] == 0).sum())

    out = pd.DataFrame({"N_bull": n_bull, "N_bear": n_bear, "N_neutral": n_neutral}).reset_index()

    denom = (out["N_bull"] + out["N_bear"]).replace(0, np.nan)
    out["Sent"] = (out["N_bull"] - out["N_bear"]) / denom
    out["Sent"] = out["Sent"].fillna(0.0)                       # no bullish/bearish posts -> neutral sentiment
    out["PV"] = np.log(out["N_bull"] + out["N_bear"] + 1)
    out["AI"] = 1 - np.sqrt(np.clip(1 - out["Sent"] ** 2, 0, 1))

    return out.sort_values(["stock", "date"]).reset_index(drop=True)


def _cli():
    from src.utils.data_loader import DATA_DIR
    posts_path = DATA_DIR / "processed" / "posts_with_sentiment.csv"
    if not posts_path.exists():
        raise FileNotFoundError(
            f"{posts_path} not found. Run bert_sentiment.py predict first."
        )
    posts = pd.read_csv(posts_path, parse_dates=["post_time"])
    indicators = build_sentiment_indicators(posts)
    out_path = DATA_DIR / "processed" / "sentiment_indicators.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(out_path, index=False)
    print(f"Wrote sentiment indicators to {out_path} ({len(indicators)} rows)")


if __name__ == "__main__":
    _cli()
