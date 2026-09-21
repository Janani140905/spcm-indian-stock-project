"""
End-to-end SPCM pipeline orchestration.

  python main.py --demo                 # synthetic data, fast sklearn sentiment model (no GPU needed)
  python main.py --demo --use_bert       # synthetic data, real BERT fine-tuning (slower, needs transformers/torch)
  python main.py                         # real data pipeline (expects data/raw/*.csv already populated,
                                          #  and models/bert_sentiment/ already fine-tuned)

This script is intentionally a thin wrapper around the `src/` modules described in
README.md -- read that file for the step-by-step explanation of what each stage does.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from src.utils.data_loader import (
    DATA_DIR, make_synthetic_dataset, make_indian_synthetic_dataset,
    load_prices, load_posts, load_ff5_factors, load_fundamentals,
    validate_data_quality
)
from src.features.sentiment_features import build_sentiment_indicators
from src.features.homophily import compute_user_stock_day_homophily, aggregate_stock_homophily
from src.features.financial_features import compute_abnormal_returns
from src.models.spcm import (
    build_price_features, build_labels, build_fusion_matrix, train_spcm,
    predict_proba, compute_auc, get_feature_columns,
)
from src.backtest.backtest import rolling_backtest, compute_portfolio_cumulative_return, compute_benchmark_cumulative_return


def fast_sentiment_labels(posts: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight TF-IDF + Logistic Regression sentiment classifier, used only for the
    quick --demo path so the whole pipeline runs in seconds without a GPU.
    For real projects, use src/sentiment/bert_sentiment.py (BERT fine-tuning), which
    is what the paper actually uses and reports 84.12% accuracy with.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    labeled = pd.read_csv(DATA_DIR / "labeled" / "annotated_posts.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        labeled["text"], labeled["label"], test_size=0.3, random_state=42, stratify=labeled["label"]
    )
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    acc = clf.score(Xte, y_test)
    print(f"[demo sentiment model] held-out accuracy: {acc:.4f}")

    X_all = vec.transform(posts["text"].fillna(""))
    posts = posts.copy()
    posts["sentiment"] = clf.predict(X_all)
    return posts


def run_pipeline(use_bert: bool = False, demo: bool = True, india: bool = False, synthetic_india: bool = False):
    if india and synthetic_india:
        print("== Generating synthetic INDIAN dataset ==")
        make_indian_synthetic_dataset()
        data_path = DATA_DIR / "raw" / "india_synthetic"
        out_path = DATA_DIR / "processed" / "india_synthetic"
        out_path.mkdir(parents=True, exist_ok=True)
    elif india:
        print("== Running REAL Indian Data Pipeline ==")
        data_path = DATA_DIR / "raw" / "india"
        out_path = DATA_DIR / "processed" / "india"
        out_path.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            raise FileNotFoundError(f"Missing real data directory: {data_path}. Please place posts.csv, prices.csv, and ff5_factors.csv there.")
    elif demo:
        print("== Generating synthetic dataset ==")
        make_synthetic_dataset()
        data_path = DATA_DIR / "raw"
        out_path = DATA_DIR / "processed"
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        data_path = DATA_DIR / "raw"
        out_path = DATA_DIR / "processed"

    # Use robust loaders instead of direct pd.read_csv to trigger validations
    posts_raw = load_posts(data_path / "posts.csv")
    prices = load_prices(data_path / "prices.csv")
    ff5 = load_ff5_factors(data_path / "ff5_factors.csv")
    
    # Pre-execution Data Quality & Alignment Report
    if india:
        validate_data_quality(prices, posts_raw, ff5)

    print("== Step 2: sentiment classification ==")
    if use_bert:
        from src.sentiment.bert_sentiment import train as bert_train, predict as bert_predict
        bert_train(
            labeled_csv=str(data_path.parent / "labeled" / "annotated_posts.csv"),
            output_dir=str(Path("models") / "bert_sentiment"),
            base_model="ProsusAI/finbert" if india else "bert-base-chinese"
        )
        posts = bert_predict(
            posts_csv=str(data_path / "posts.csv"),
            model_dir=str(Path("models") / "bert_sentiment"),
            output_csv=str(out_path / "posts_with_sentiment.csv"),
        )
    else:
        posts = fast_sentiment_labels(posts_raw)
        posts.to_csv(out_path / "posts_with_sentiment.csv", index=False)

    print("== Step 3: sentiment & homophily indicators ==")
    sentiment_ind = build_sentiment_indicators(posts)
    sentiment_ind.to_csv(out_path / "sentiment_indicators.csv", index=False)
    
    # Check if thread_id/author exists to compute homophily
    if "thread_id" in posts.columns and "author" in posts.columns and not india:
        user_homo = compute_user_stock_day_homophily(posts)
        homophily_ind = aggregate_stock_homophily(user_homo)
        homophily_ind.to_csv(out_path / "homophily.csv", index=False)
    else:
        print("[!] Skipping Homophily (thread_id or author missing in dataset)")
        homophily_ind = None

    print("== Step 4: abnormal returns ==")
    ar_df = compute_abnormal_returns(prices, ff5, window=60 if demo or india else 250)
    ar_df.to_csv(out_path / "abnormal_returns.csv", index=False)

    print("== Step 6: build fusion matrix + train SPCM ==")
    price_feat = build_price_features(prices)
    labels = build_labels(prices)
    fusion = build_fusion_matrix(price_feat, sentiment_ind, homophily_ind, labels)

    model, scaler, feature_cols = train_spcm(fusion, classifier="nn")
    proba = predict_proba(model, scaler, feature_cols, fusion)
    auc = compute_auc(fusion, proba)
    print(f"In-sample SPCM AUC: {auc:.4f}")

    print("== Step 7: rolling backtest ==")
    all_dates = sorted(fusion["date"].unique())
    bt_dates = all_dates[len(all_dates) // 2::5]  # start halfway through, rebalance every 5 days
    recs, daily_auc = rolling_backtest(fusion, bt_dates, classifier="nn", top_k=5)
    if len(daily_auc):
        print(f"Backtest mean AUC: {daily_auc['auc'].mean():.4f}")

    port = compute_portfolio_cumulative_return(recs, prices, hold_days=5)
    bench = compute_benchmark_cumulative_return(prices, hold_days=5)
    if len(port):
        print(f"Backtest portfolio cumulative return: {port['cumulative_return'].iloc[-1]:.2%}")
    if len(bench):
        print(f"Backtest benchmark cumulative return: {bench['bench_cumulative_return'].iloc[-1]:.2%}")

    out_dir = out_path
    recs.to_csv(out_dir / "backtest_recommendations.csv", index=False)
    port.to_csv(out_dir / "backtest_portfolio_returns.csv", index=False)
    bench.to_csv(out_dir / "backtest_benchmark_returns.csv", index=False)
    print(f"\nAll processed outputs written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="generate synthetic data first")
    parser.add_argument("--use_bert", action="store_true", help="fine-tune & use real BERT instead of the fast demo classifier")
    parser.add_argument("--india", action="store_true", help="run REAL Indian market adaptation")
    parser.add_argument("--synthetic", action="store_true", help="if combined with --india, generates and runs synthetic Indian data")
    args = parser.parse_args()
    run_pipeline(use_bert=args.use_bert, demo=args.demo, india=args.india, synthetic_india=args.synthetic)
