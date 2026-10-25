"""
Rolling-window backtest reproducing paper Sec V.B / Figure 5 (AUC by half-year) and
Figure 6 (cumulative return of recommended portfolio vs. CSI300-style benchmark).

Backtest design (paper):
- Training window: trailing 6 months of (features, label) pairs
- Test point: the current backtesting day
- Recommend: top-K stocks by predicted P(label=1)
- Portfolio return: equal-weighted, held for the recommendation interval
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.spcm import (
    build_price_features, build_labels, build_fusion_matrix, get_feature_columns,
)
from src.models.baseline_models import MODEL_REGISTRY, scale_to_unit_range
from src.utils.data_loader import TARGET_STOCKS, normalize_indian_symbol

TRAIN_WINDOW_DAYS = 182  # ~6 months of trading days, paper: "past six months"


def rolling_backtest(
    fusion_df: pd.DataFrame,
    backtest_dates: pd.DatetimeIndex,
    classifier: str = "nn",
    include_sentiment: bool = True,
    top_k: int = 50,
    train_window_days: int = TRAIN_WINDOW_DAYS,
):
    """
    For each date in `backtest_dates`, train on the trailing `train_window_days` window
    (excluding the current date to avoid lookahead) and predict/rank stocks for that date.

    Returns
    -------
    recs : DataFrame (date, stock, proba) -- the top-K recommended stocks per date
    daily_auc : DataFrame (date, auc) -- classification AUC on that date's cross-section
    """
    feature_cols = get_feature_columns(fusion_df, include_sentiment=include_sentiment)
    fusion_df = fusion_df.sort_values("date").reset_index(drop=True)
    all_dates = np.sort(fusion_df["date"].unique())

    recs_rows = []
    auc_rows = []

    for target_date in backtest_dates:
        target_date = pd.Timestamp(target_date)
        window_start = target_date - pd.Timedelta(days=train_window_days)
        train_df = fusion_df[(fusion_df["date"] >= window_start) & (fusion_df["date"] < target_date)]
        test_df = fusion_df[fusion_df["date"] == target_date]

        train_df = train_df.dropna(subset=feature_cols + ["label"])
        test_df = test_df.dropna(subset=feature_cols)
        if len(train_df) < 50 or len(test_df) < 5 or train_df["label"].nunique() < 2:
            continue

        X_train = train_df[feature_cols].values.astype(float)
        y_train = train_df["label"].values.astype(int)
        X_test = test_df[feature_cols].values.astype(float)

        X_train_s, scaler = scale_to_unit_range(X_train)
        X_test_s, _ = scale_to_unit_range(X_test, scaler=scaler)

        model = MODEL_REGISTRY[classifier]()
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]

        test_df = test_df.copy()
        test_df["proba"] = proba

        top = test_df.sort_values("proba", ascending=False).head(top_k)
        for _, row in top.iterrows():
            recs_rows.append(dict(date=target_date, stock=row["stock"], proba=row["proba"]))

        if test_df["label"].notna().any() and test_df["label"].nunique() > 1:
            auc = roc_auc_score(test_df["label"], test_df["proba"])
            auc_rows.append(dict(date=target_date, auc=auc))

    return pd.DataFrame(recs_rows), pd.DataFrame(auc_rows)


def compute_portfolio_cumulative_return(recs: pd.DataFrame, prices: pd.DataFrame, hold_days: int = 10) -> pd.DataFrame:
    """
    Equal-weighted portfolio return: on each recommendation date, buy the recommended
    top-K stocks, hold `hold_days`, then rebalance to the next recommendation date's list.
    Returns a DataFrame (date, portfolio_return, cumulative_return).
    """
    prices = prices.sort_values(["stock", "date"]).copy()
    prices["stock"] = prices["stock"].apply(normalize_indian_symbol)
    recs = recs.copy()
    recs["stock"] = recs["stock"].apply(normalize_indian_symbol)
    px = prices.pivot(index="date", columns="stock", values="close")
    fwd_ret = px.pct_change(hold_days).shift(-hold_days)

    rec_dates = sorted(recs["date"].unique())
    port_returns = []
    for d in rec_dates:
        stocks_today = recs[recs["date"] == d]["stock"].tolist()
        if d not in fwd_ret.index:
            continue
        rets = fwd_ret.loc[d, [s for s in stocks_today if s in fwd_ret.columns]]
        port_returns.append(dict(date=d, portfolio_return=rets.mean()))

    port_df = pd.DataFrame(port_returns).sort_values("date")
    port_df["portfolio_return"] = port_df["portfolio_return"].fillna(0.0)
    port_df["cumulative_return"] = (1 + port_df["portfolio_return"]).cumprod() - 1
    return port_df


def compute_benchmark_cumulative_return(prices: pd.DataFrame, hold_days: int = 10) -> pd.DataFrame:
    """Equal-weighted-across-universe benchmark, as a stand-in for the market index."""
    prices = prices.sort_values(["stock", "date"]).copy()
    prices["stock"] = prices["stock"].apply(normalize_indian_symbol)
    px = prices.pivot(index="date", columns="stock", values="close")
    index_level = px.mean(axis=1)
    bench_ret = index_level.pct_change(hold_days)
    out = bench_ret.reset_index().rename(columns={0: "bench_return"})
    out.columns = ["date", "bench_return"]
    out["bench_return"] = out["bench_return"].fillna(0.0)
    out["bench_cumulative_return"] = (1 + out["bench_return"]).cumprod() - 1
    return out


def _cli():
    from src.utils.data_loader import DATA_DIR

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--backtest_start", default="2018-07-01")
    p_run.add_argument("--backtest_end", default="2020-12-31")
    p_run.add_argument("--top_n", type=int, default=50)
    p_run.add_argument("--classifier", default="nn", choices=list(MODEL_REGISTRY.keys()))
    p_run.add_argument("--rebalance_every", type=int, default=10, help="trading days between rebalances")
    args = parser.parse_args()

    prices = pd.read_csv(DATA_DIR / "raw" / "prices.csv", parse_dates=["date"])
    sentiment_ind = pd.read_csv(DATA_DIR / "processed" / "sentiment_indicators.csv", parse_dates=["date"])
    homophily_ind = pd.read_csv(DATA_DIR / "processed" / "homophily.csv", parse_dates=["date"])

    price_feat = build_price_features(prices)
    labels = build_labels(prices)
    fusion = build_fusion_matrix(price_feat, sentiment_ind, homophily_ind, labels)

    all_dates = sorted(fusion["date"].unique())
    bt_dates = [d for d in all_dates if args.backtest_start <= str(pd.Timestamp(d).date()) <= args.backtest_end]
    bt_dates = bt_dates[::args.rebalance_every]  # rebalance every N trading days

    recs, daily_auc = rolling_backtest(fusion, bt_dates, classifier=args.classifier, top_k=args.top_n)
    port = compute_portfolio_cumulative_return(recs, prices, hold_days=args.rebalance_every)
    bench = compute_benchmark_cumulative_return(prices, hold_days=args.rebalance_every)

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_dir / "backtest_recommendations.csv", index=False)
    daily_auc.to_csv(out_dir / "backtest_auc.csv", index=False)
    port.to_csv(out_dir / "backtest_portfolio_returns.csv", index=False)
    bench.to_csv(out_dir / "backtest_benchmark_returns.csv", index=False)

    print(f"Mean AUC: {daily_auc['auc'].mean():.4f}" if len(daily_auc) else "No AUC computed")
    if len(port):
        print(f"Final portfolio cumulative return: {port['cumulative_return'].iloc[-1]:.2%}")
    if len(bench):
        print(f"Final benchmark cumulative return: {bench['bench_cumulative_return'].iloc[-1]:.2%}")


if __name__ == "__main__":
    _cli()
