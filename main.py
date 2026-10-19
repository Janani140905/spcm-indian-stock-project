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

    print("== Step 2 & 3: sentiment classification & indicators ==")
    if india and not synthetic_india:
        sentiment_path = out_path / "daily_sentiment_features.csv"
        if not sentiment_path.exists():
            from src.features.india_sentiment import run_sentiment_pipeline
            run_sentiment_pipeline()
        sentiment_ind = pd.read_csv(sentiment_path)
        # Ensure date is parsed properly for merging
        sentiment_ind["date"] = pd.to_datetime(sentiment_ind["date"]).dt.normalize()
        homophily_ind = None
    else:
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
    
    print("== Step 5: control / factor features ==")
    from src.features.financial_features import build_control_variables
    control_feat = build_control_variables(ar_df)
    control_feat.to_csv(out_path / "control_features.csv", index=False)

    print("== Step 6: build fusion matrix + validate ==")
    price_feat = build_price_features(prices)
    labels = build_labels(prices)
    
    # SHIFT SENTIMENT DATES TO AVOID LEAKAGE
    from src.models.spcm import shift_sentiment_dates
    sentiment_ind = shift_sentiment_dates(sentiment_ind, price_feat)
    
    fusion = build_fusion_matrix(price_feat, sentiment_ind, homophily_ind, labels, control_feat=control_feat)

    print("\nWALK-FORWARD SMOKE TEST")
    
    # Sort and ensure time is chronological
    fusion = fusion.sort_values("date").reset_index(drop=True)
    
    # Smoke test on a small subset (e.g., 2018-2019 train, 2020 test, 2021 test)
    # Define folds manually for the smoke test
    folds = [
        {"train_start": "2018-01-01", "train_end": "2018-12-14", "test_start": "2019-01-01", "test_end": "2019-12-31"},
        {"train_start": "2018-01-01", "train_end": "2019-12-16", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    ]
    
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    any_overlap = False
    
    for i, f in enumerate(folds):
        print(f"\nFold {i+1}:")
        train_mask = (fusion["date"] >= f["train_start"]) & (fusion["date"] <= f["train_end"])
        test_mask = (fusion["date"] >= f["test_start"]) & (fusion["date"] <= f["test_end"])
        
        train_df = fusion[train_mask].copy()
        test_df = fusion[test_mask].copy()
        
        if train_df.empty or test_df.empty:
            print(f"Skipping Fold {i+1} due to empty train or test set.")
            continue
            
        max_train_date = train_df["date"].max()
        min_test_date = test_df["date"].min()
        
        overlap = max_train_date >= min_test_date
        if overlap:
            any_overlap = True
            
        print(f"Train: {train_df['date'].min().date()} to {max_train_date.date()}")
        print(f"Test:  {min_test_date.date()} to {test_df['date'].max().date()}")
        print(f"Train rows: {len(train_df)}")
        print(f"Test rows:  {len(test_df)}")
        print(f"Train stocks: {train_df['stock'].nunique()}")
        print(f"Test stocks:  {test_df['stock'].nunique()}")
        
        # Train
        model, scaler, imputer, feature_cols = train_spcm(fusion, classifier="nn", train_mask=train_mask)
        
        # Test (using same imputer and scaler via predict_proba)
        test_proba = predict_proba(model, scaler, imputer, feature_cols, test_df)
        
        # Evaluate on TEST only
        # We need true labels. Drop rows in test where label is NaN
        test_eval = test_df.merge(test_proba, on=["stock", "date"], how="inner").dropna(subset=["label", "proba"])
        
        pos_train = train_df["label"].sum()
        pos_test = test_eval["label"].sum()
        print(f"Positive-label count in train: {pos_train}")
        print(f"Positive-label count in test:  {pos_test}")
        
        if len(test_eval) == 0:
            print("AUC: NaN (no valid test labels)")
            continue
            
        y_true = test_eval["label"].values
        y_pred = (test_eval["proba"] >= 0.5).astype(int)
        
        if len(set(y_true)) < 2:
            print("AUC: NaN (test set contains only one class)")
        else:
            auc = roc_auc_score(y_true, test_eval["proba"])
            print(f"AUC: {auc:.4f}")
            
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Number of test observations: {len(test_eval)}")
        
    print(f"\nAny train/test date overlap: {any_overlap}")
    
    if india and not synthetic_india:
        import sys
        sys.exit(0)
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
