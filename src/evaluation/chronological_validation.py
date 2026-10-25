"""
Chronological Out-of-Sample Walk-Forward Validation Engine for Indian SPCM (Phase 6).

Implements strictly chronological walk-forward cross-validation with:
1. Exact 10-trading-day temporal purge (embargo) to eliminate forward-label overlap.
2. Preprocessing isolation (SimpleImputer and MinMaxScaler fitted strictly on training data).
3. Comprehensive metric suite: ROC-AUC, PR-AUC, Balanced Accuracy, Accuracy, Precision,
   Recall, F1, Confusion Matrix, Positive-Class Prevalence, Precision@3, and Top-3 Forward Returns.
4. Rigorous baseline comparisons (Majority-Class and Random/Prior Baselines).
5. Reproducible CSV export of fold-level and summary metrics.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

from src.models.baseline_models import MODEL_REGISTRY
from src.models.spcm import (
    FORWARD_HORIZON,
    PRICE_LAGS,
    TOP_PCT_LABEL,
    build_fusion_matrix,
    build_labels,
    build_price_features,
    get_feature_columns,
)
from src.utils.fetch_indian_prices import STOCK_UNIVERSE
from src.utils.indian_market_data import extract_stock_code

logger = logging.getLogger("chronological_validation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_PRICES_PATH = "data/processed/india/prices.csv"
DEFAULT_REPORTS_DIR = "reports"


@dataclass
class ChronologicalFold:
    """Represents a single walk-forward split with temporal purge boundaries."""
    fold_id: int
    train_start: pd.Timestamp
    train_end_raw: pd.Timestamp
    train_end_purged: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_trading_days: int
    last_train_label_realization: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start.date()),
            "train_end_raw": str(self.train_end_raw.date()),
            "train_end_purged": str(self.train_end_purged.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "purge_trading_days": self.purge_trading_days,
            "last_train_label_realization": str(self.last_train_label_realization.date()),
        }


class ChronologicalSplitter:
    """
    Constructs expanding walk-forward folds with mathematical temporal purging.

    Because labels are forward 10-trading-day returns (close_{t+10} / close_t - 1),
    observations in the final 10 trading sessions of the training window derive their
    outcomes from the test window. To prevent forward-looking overlap leakage,
    the training window is purged such that:
        t_train_end + 10 trading sessions < t_test_start
    """

    def __init__(self, trading_dates: Sequence[pd.Timestamp], forward_horizon: int = 10):
        self.trading_dates = sorted(pd.to_datetime(d).normalize() for d in set(trading_dates))
        self.date_to_idx = {d: i for i, d in enumerate(self.trading_dates)}
        self.forward_horizon = forward_horizon

    def create_annual_folds(self) -> list[ChronologicalFold]:
        """
        Creates annual expanding walk-forward folds.
        With 2021-2024 data:
            Fold 1: Train 2021 (purged), Test 2022
            Fold 2: Train 2021-2022 (purged), Test 2023
            Fold 3: Train 2021-2023 (purged), Test 2024
        """
        years = sorted(list(set(d.year for d in self.trading_dates)))
        if len(years) < 2:
            raise ValueError(f"At least 2 calendar years required for annual walk-forward folds. Found: {years}")

        start_year = years[0]
        folds: list[ChronologicalFold] = []

        for fold_idx, test_year in enumerate(years[1:], start=1):
            train_dates = [d for d in self.trading_dates if start_year <= d.year < test_year]
            test_dates = [d for d in self.trading_dates if d.year == test_year]

            if not train_dates or not test_dates:
                continue

            test_start = test_dates[0]
            test_end = test_dates[-1]
            train_start = train_dates[0]
            train_end_raw = train_dates[-1]

            # Enforce 10-trading-day purge boundary
            test_start_idx = self.date_to_idx[test_start]
            last_safe_train_idx = test_start_idx - self.forward_horizon - 1

            if last_safe_train_idx < 0:
                raise ValueError(f"Insufficient history to purge {self.forward_horizon} days before {test_start}")

            train_end_purged = self.trading_dates[last_safe_train_idx]
            last_realization = self.trading_dates[last_safe_train_idx + self.forward_horizon]

            # Verify zero-leakage guarantee
            if last_realization >= test_start:
                raise AssertionError(
                    f"Purge violation in fold {fold_idx}: "
                    f"Last realization {last_realization.date()} >= Test start {test_start.date()}"
                )

            folds.append(
                ChronologicalFold(
                    fold_id=fold_idx,
                    train_start=train_start,
                    train_end_raw=train_end_raw,
                    train_end_purged=train_end_purged,
                    test_start=test_start,
                    test_end=test_end,
                    purge_trading_days=self.forward_horizon,
                    last_train_label_realization=last_realization,
                )
            )

        return folds


@dataclass
class FoldEvaluationResult:
    """Detailed evaluation metrics for a single model on a single out-of-sample fold."""
    fold_id: int
    model: str
    feature_set: str
    train_start: str
    train_end_purged: str
    test_start: str
    test_end: str
    train_rows: int
    train_pos_count: int
    train_pos_prevalence: float
    test_rows: int
    test_pos_count: int
    test_pos_prevalence: float
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int
    mean_predicted_prob: float
    precision_at_3: float
    top3_mean_fwd_return: float
    benchmark_mean_fwd_return: float


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_eval: pd.DataFrame,
    fold_id: int,
    model_name: str,
    feature_set: str,
    train_info: dict[str, Any],
    test_info: dict[str, Any],
) -> Tuple[FoldEvaluationResult, pd.DataFrame]:
    """
    Computes all classification, discrimination, ranking, and baseline metrics.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Discrimination metrics
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    # 2. Confusion Matrix & Thresholded Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # 3. Cross-Sectional Ranking Metrics (Precision@3 and Top-3 Forward Return)
    eval_work = df_eval.copy()
    eval_work["prob"] = y_prob
    eval_work["pred"] = y_pred

    prec_at_3_list: list[float] = []
    top3_ret_list: list[float] = []
    bench_ret_list: list[float] = []
    oos_rows: list[dict[str, Any]] = []

    for d, group in eval_work.groupby("date"):
        g_sorted = group.sort_values("prob", ascending=False).reset_index(drop=True)
        top3 = g_sorted.head(3)
        p3 = top3["label"].sum() / min(3.0, float(len(top3))) if len(top3) > 0 else 0.0
        prec_at_3_list.append(p3)
        top3_ret_list.append(top3["fwd_ret"].mean())
        bench_ret_list.append(group["fwd_ret"].mean())

        for rank_idx, row in g_sorted.iterrows():
            oos_rows.append({
                "fold": fold_id,
                "model": model_name,
                "feature_set": feature_set,
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10],
                "stock": row["stock"],
                "fwd_ret": row["fwd_ret"],
                "y_true": int(row["label"]),
                "y_prob": round(float(row["prob"]), 4),
                "y_pred": int(row["pred"]),
                "rank": int(rank_idx + 1),
                "is_recommended": bool(rank_idx < 3),
            })

    mean_p3 = float(np.mean(prec_at_3_list)) if prec_at_3_list else float("nan")
    mean_top3_ret = float(np.mean(top3_ret_list)) if top3_ret_list else float("nan")
    mean_bench_ret = float(np.mean(bench_ret_list)) if bench_ret_list else float("nan")

    result = FoldEvaluationResult(
        fold_id=fold_id,
        model=model_name,
        feature_set=feature_set,
        train_start=str(train_info["start"].date()),
        train_end_purged=str(train_info["end"].date()),
        test_start=str(test_info["start"].date()),
        test_end=str(test_info["end"].date()),
        train_rows=train_info["rows"],
        train_pos_count=train_info["pos_count"],
        train_pos_prevalence=round(train_info["pos_count"] / train_info["rows"], 4) if train_info["rows"] > 0 else 0.0,
        test_rows=test_info["rows"],
        test_pos_count=test_info["pos_count"],
        test_pos_prevalence=round(test_info["pos_count"] / test_info["rows"], 4) if test_info["rows"] > 0 else 0.0,
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        balanced_accuracy=round(bal_acc, 4),
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        mean_predicted_prob=round(float(np.mean(y_prob)), 4),
        precision_at_3=round(mean_p3, 4),
        top3_mean_fwd_return=round(mean_top3_ret, 4),
        benchmark_mean_fwd_return=round(mean_bench_ret, 4),
    )

    return result, pd.DataFrame(oos_rows)


def run_chronological_validation(
    prices_path: str = DEFAULT_PRICES_PATH,
    reports_dir: str = DEFAULT_REPORTS_DIR,
    feature_modes: Sequence[str] = ("Price-Only", "Price + Sentiment"),
    models: Sequence[str] = ("nn", "rf", "knn"),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes full walk-forward evaluation across models and feature configurations.

    Returns:
        (fold_metrics_df, model_comparison_df, oos_predictions_df)
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Prices
    logger.info(f"Loading historical prices from '{prices_path}'...")
    prices = pd.read_csv(prices_path)
    prices["stock"] = prices["stock"].apply(extract_stock_code)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.sort_values(["stock", "date"]).reset_index(drop=True)

    universe = sorted(prices["stock"].unique())
    logger.info(f"Loaded {len(prices)} rows across {len(universe)} stocks: {universe}")

    # 2. Build Technical Features & Forward-Return Labels
    logger.info(f"Building price features with lags {PRICE_LAGS}...")
    price_feat = build_price_features(prices, lags=PRICE_LAGS)

    logger.info(f"Building labels (horizon={FORWARD_HORIZON}d, top_pct={TOP_PCT_LABEL})...")
    labels = build_labels(prices, horizon=FORWARD_HORIZON, top_pct=TOP_PCT_LABEL)

    # 3. Create Neutral Sentiment Grid for Price+Sentiment Mode
    sentiment_grid = prices[["stock", "date"]].drop_duplicates().copy()
    sentiment_grid["Sent"] = np.nan
    sentiment_grid["PV"] = np.nan
    sentiment_grid["AI"] = np.nan
    sentiment_grid["news_available"] = 0
    sentiment_grid["headline_count"] = 0

    # Populate 10% neutral sample so median imputer fits cleanly
    rng = np.random.default_rng(random_state)
    active_idx = rng.choice(sentiment_grid.index, size=max(1, len(sentiment_grid) // 10), replace=False)
    sentiment_grid.loc[active_idx, "Sent"] = 0.0
    sentiment_grid.loc[active_idx, "PV"] = np.log(2.0)
    sentiment_grid.loc[active_idx, "AI"] = 0.0
    sentiment_grid.loc[active_idx, "news_available"] = 1
    sentiment_grid.loc[active_idx, "headline_count"] = 1

    # 4. Build Fusion Matrices
    logger.info("Assembling fusion matrices...")
    fusion_sentiment = build_fusion_matrix(
        price_feat=price_feat,
        sentiment_ind=sentiment_grid,
        homophily_ind=None,
        labels=labels,
        control_feat=None,
    )
    fusion_price_only = build_fusion_matrix(
        price_feat=price_feat,
        sentiment_ind=None,
        homophily_ind=None,
        labels=labels,
        control_feat=None,
    )

    feature_cols_map = {
        "Price-Only": get_feature_columns(fusion_price_only, include_sentiment=False),
        "Price + Sentiment": get_feature_columns(fusion_sentiment, include_sentiment=True),
    }

    # 5. Build Chronological Purged Folds
    trading_dates = sorted(prices["date"].unique())
    splitter = ChronologicalSplitter(trading_dates, forward_horizon=FORWARD_HORIZON)
    folds = splitter.create_annual_folds()

    logger.info(f"Constructed {len(folds)} annual expanding walk-forward folds:")
    for f in folds:
        logger.info(
            f"  Fold {f.fold_id}: Train {f.train_start.date()} to {f.train_end_purged.date()} (Purged) | "
            f"Test {f.test_start.date()} to {f.test_end.date()} | "
            f"Last Label Realization: {f.last_train_label_realization.date()}"
        )

    all_fold_results: list[FoldEvaluationResult] = []
    all_oos_dfs: list[pd.DataFrame] = []

    # 6. Walk-Forward Loop
    for mode in feature_modes:
        feat_cols = feature_cols_map[mode]
        fusion_df = fusion_sentiment if mode == "Price + Sentiment" else fusion_price_only
        logger.info(f"\n=======================================================")
        logger.info(f" EVALUATING CONFIGURATION: {mode} ({len(feat_cols)} features)")
        logger.info(f"=======================================================")

        for f in folds:
            train_mask = (fusion_df["date"] >= f.train_start) & (fusion_df["date"] <= f.train_end_purged)
            test_mask = (fusion_df["date"] >= f.test_start) & (fusion_df["date"] <= f.test_end)

            train_df = fusion_df[train_mask].dropna(subset=["label"]).copy()
            test_df = fusion_df[test_mask].dropna(subset=["label"]).copy()

            if len(train_df) == 0 or len(test_df) == 0:
                logger.warning(f"Fold {f.fold_id} has empty train or test split. Skipping.")
                continue

            # Verify temporal boundary
            max_train_date = train_df["date"].max()
            min_test_date = test_df["date"].min()
            assert max_train_date < min_test_date, f"Temporal overlap in fold {f.fold_id}: {max_train_date} >= {min_test_date}"

            X_train = train_df[feat_cols].values.astype(float)
            y_train = train_df["label"].values.astype(int)
            X_test = test_df[feat_cols].values.astype(float)
            y_test = test_df["label"].values.astype(int)

            train_info = {
                "start": train_df["date"].min(),
                "end": max_train_date,
                "rows": len(train_df),
                "pos_count": int(y_train.sum()),
            }
            test_info = {
                "start": min_test_date,
                "end": test_df["date"].max(),
                "rows": len(test_df),
                "pos_count": int(y_test.sum()),
            }

            # Preprocessing: Fit on TRAIN ONLY
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_train_imp)

            # Transform TEST using frozen train imputer & scaler
            X_test_imp = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imp)

            eval_test_df = test_df[["date", "stock", "fwd_ret", "label"]].copy()

            # --- A. Evaluate Trained Models ---
            for m_key in models:
                model_fn = MODEL_REGISTRY[m_key]
                model = model_fn()
                model.fit(X_train_scaled, y_train)

                y_prob = model.predict_proba(X_test_scaled)[:, 1]

                res, oos_df = compute_comprehensive_metrics(
                    y_true=y_test,
                    y_prob=y_prob,
                    df_eval=eval_test_df,
                    fold_id=f.fold_id,
                    model_name=m_key.upper(),
                    feature_set=mode,
                    train_info=train_info,
                    test_info=test_info,
                )
                all_fold_results.append(res)
                all_oos_dfs.append(oos_df)

                logger.info(
                    f"  Fold {f.fold_id} [{m_key.upper():3s}] AUC={res.roc_auc:.4f} | "
                    f"PR-AUC={res.pr_auc:.4f} | BalAcc={res.balanced_accuracy:.4f} | "
                    f"Acc={res.accuracy:.4f} | P@3={res.precision_at_3:.3f} | "
                    f"Top3Ret={res.top3_mean_fwd_return*100:.2f}% (Bench={res.benchmark_mean_fwd_return*100:.2f}%)"
                )

            # --- B. Evaluate Baselines ---
            # 1. Majority-Class Baseline (Predict constant 0)
            y_prob_maj = np.zeros(len(y_test), dtype=float)
            res_maj, oos_maj = compute_comprehensive_metrics(
                y_true=y_test,
                y_prob=y_prob_maj,
                df_eval=eval_test_df,
                fold_id=f.fold_id,
                model_name="MAJORITY_BASELINE",
                feature_set=mode,
                train_info=train_info,
                test_info=test_info,
            )
            all_fold_results.append(res_maj)
            all_oos_dfs.append(oos_maj)

            # 2. Prior-Probability Random Baseline
            train_prior = train_info["pos_count"] / train_info["rows"]
            rng_base = np.random.default_rng(random_state + f.fold_id)
            y_prob_rand = rng_base.uniform(0.0, 1.0, size=len(y_test))
            res_rand, oos_rand = compute_comprehensive_metrics(
                y_true=y_test,
                y_prob=y_prob_rand,
                df_eval=eval_test_df,
                fold_id=f.fold_id,
                model_name="RANDOM_BASELINE",
                feature_set=mode,
                train_info=train_info,
                test_info=test_info,
            )
            all_fold_results.append(res_rand)
            all_oos_dfs.append(oos_rand)

    # 7. Aggregate and Export Results
    fold_df = pd.DataFrame([asdict(r) for r in all_fold_results])
    oos_predictions_df = pd.concat(all_oos_dfs, ignore_index=True)

    # Compute model comparison summary
    summary_rows = []
    for (m, feat), g in fold_df.groupby(["model", "feature_set"]):
        summary_rows.append({
            "model": m,
            "feature_set": feat,
            "folds_evaluated": len(g),
            "mean_roc_auc": round(g["roc_auc"].mean(), 4),
            "std_roc_auc": round(g["roc_auc"].std(), 4),
            "mean_pr_auc": round(g["pr_auc"].mean(), 4),
            "mean_balanced_accuracy": round(g["balanced_accuracy"].mean(), 4),
            "mean_accuracy": round(g["accuracy"].mean(), 4),
            "mean_precision": round(g["precision"].mean(), 4),
            "mean_recall": round(g["recall"].mean(), 4),
            "mean_f1": round(g["f1"].mean(), 4),
            "mean_precision_at_3": round(g["precision_at_3"].mean(), 4),
            "mean_top3_fwd_return": round(g["top3_mean_fwd_return"].mean(), 4),
            "mean_bench_fwd_return": round(g["benchmark_mean_fwd_return"].mean(), 4),
            "test_prevalence_mean": round(g["test_pos_prevalence"].mean(), 4),
        })

    comparison_df = pd.DataFrame(summary_rows).sort_values(
        ["feature_set", "mean_roc_auc"], ascending=[True, False]
    ).reset_index(drop=True)

    # Write CSVs
    fold_csv_path = out_dir / "phase6_walkforward_fold_metrics.csv"
    comp_csv_path = out_dir / "phase6_model_comparison.csv"
    oos_csv_path = out_dir / "phase6_oos_predictions.csv"

    fold_df.to_csv(fold_csv_path, index=False)
    comparison_df.to_csv(comp_csv_path, index=False)
    oos_predictions_df.to_csv(oos_csv_path, index=False)

    logger.info(f"\nSaved walk-forward fold metrics to: {fold_csv_path.resolve()}")
    logger.info(f"Saved model comparison summary to:    {comp_csv_path.resolve()}")
    logger.info(f"Saved out-of-sample predictions to:   {oos_csv_path.resolve()}")

    return fold_df, comparison_df, oos_predictions_df


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Chronological Walk-Forward Validation Engine.")
    parser.add_argument("--prices", default=DEFAULT_PRICES_PATH, help="Path to cleaned prices CSV")
    parser.add_argument("--out-dir", default=DEFAULT_REPORTS_DIR, help="Output directory for CSV reports")
    args = parser.parse_args()

    print("=" * 85)
    print(" PHASE 6: RIGOROUS CHRONOLOGICAL OUT-OF-SAMPLE SPCM VALIDATION")
    print("=" * 85)

    fold_df, comp_df, _ = run_chronological_validation(
        prices_path=args.prices,
        reports_dir=args.out_dir,
    )

    print("\n" + "=" * 85)
    print(" SPCM MODEL COMPARISON SUMMARY ACROSS WALK-FORWARD FOLDS")
    print("=" * 85)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
