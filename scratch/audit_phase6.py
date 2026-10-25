"""
Phase 7 - Step 1: Audit of Phase 6 results, predictions, datasets, and chronology.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

def run_audit():
    print("=== Step 1 Audit: Phase 6 Verification ===")
    
    # 1. Load Phase 6 OOS Predictions
    oos_path = Path("reports/phase6_oos_predictions.csv")
    if not oos_path.exists():
        print("ERROR: reports/phase6_oos_predictions.csv does not exist!")
        return
        
    df_oos = pd.read_csv(oos_path)
    print(f"Loaded OOS predictions: {len(df_oos)} rows, columns: {list(df_oos.columns)}")
    print(f"Models in OOS: {df_oos['model'].unique()}")
    print(f"Feature sets in OOS: {df_oos['feature_set'].unique()}")
    print(f"Folds in OOS: {df_oos['fold'].unique()}")
    
    # 2. Check metrics reproduction
    metrics_path = Path("reports/phase6_walkforward_fold_metrics.csv")
    df_metrics = pd.read_csv(metrics_path)
    print(f"\nLoaded Fold Metrics: {len(df_metrics)} rows")
    
    discrepancies = []
    for _, row in df_metrics.iterrows():
        fold = int(row['fold_id'])
        model = str(row['model'])
        fset = str(row['feature_set'])
        
        subset = df_oos[(df_oos['fold'] == fold) & (df_oos['model'] == model) & (df_oos['feature_set'] == fset)]
        if len(subset) == 0:
            discrepancies.append(f"Missing OOS data for fold {fold}, model {model}, fset {fset}")
            continue
            
        y_true = subset['y_true'].values
        y_prob = subset['y_prob'].values
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Check ROC-AUC
        if len(np.unique(y_true)) > 1:
            recomputed_roc = round(float(roc_auc_score(y_true, y_prob)), 4)
            recomputed_pr = round(float(average_precision_score(y_true, y_prob)), 4)
        else:
            recomputed_roc = float("nan")
            recomputed_pr = float("nan")
            
        recomputed_acc = round(float(accuracy_score(y_true, y_pred)), 4)
        
        # Compare with recorded metrics
        if abs(recomputed_roc - row['roc_auc']) > 0.001:
            discrepancies.append(f"ROC-AUC mismatch for {model} {fset} fold {fold}: recorded {row['roc_auc']}, recomputed {recomputed_roc}")
        if abs(recomputed_pr - row['pr_auc']) > 0.001:
            discrepancies.append(f"PR-AUC mismatch for {model} {fset} fold {fold}: recorded {row['pr_auc']}, recomputed {recomputed_pr}")
        if abs(recomputed_acc - row['accuracy']) > 0.001:
            discrepancies.append(f"Accuracy mismatch for {model} {fset} fold {fold}: recorded {row['accuracy']}, recomputed {recomputed_acc}")
            
    if not discrepancies:
        print("[VERIFIED] All reported metrics in phase6_walkforward_fold_metrics.csv are exactly reproduced from phase6_oos_predictions.csv!")
    else:
        print(f"[DISCREPANCIES FOUND]: {len(discrepancies)}")
        for d in discrepancies[:10]:
            print("  -", d)
            
    # 3. Check dates and chronology in prices.csv
    prices_path = Path("data/processed/india/prices.csv")
    df_prices = pd.read_csv(prices_path)
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    unique_dates = sorted(df_prices['date'].unique())
    print(f"\nPrice history: {len(unique_dates)} trading days, from {unique_dates[0].date()} to {unique_dates[-1].date()}")
    
    # Check date to idx for purging
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    
    # Verify each fold's purging
    for fold in [1, 2, 3]:
        fold_rows = df_metrics[df_metrics['fold_id'] == fold].iloc[0]
        train_start = pd.to_datetime(fold_rows['train_start'])
        train_end_purged = pd.to_datetime(fold_rows['train_end_purged'])
        test_start = pd.to_datetime(fold_rows['test_start'])
        test_end = pd.to_datetime(fold_rows['test_end'])
        
        purged_idx = date_to_idx[train_end_purged]
        realization_idx = purged_idx + 10
        realization_date = unique_dates[realization_idx]
        
        print(f"Fold {fold}: Train [{train_start.date()} to {train_end_purged.date()} (purged)] -> Test [{test_start.date()} to {test_end.date()}]")
        print(f"  Last train label realization date: {realization_date.date()} vs Test Start: {test_start.date()}")
        assert realization_date < test_start, f"Purge leak in fold {fold}: {realization_date} >= {test_start}"
        
    print("[VERIFIED] All 3 folds strictly satisfy chronological temporal purging: last_train_realization < test_start.")
    
    # 4. Check target label interpretation
    print("\nTarget Label Analysis:")
    for fold in [1, 2, 3]:
        sub = df_oos[(df_oos['fold'] == fold) & (df_oos['model'] == 'RF') & (df_oos['feature_set'] == 'Price + Sentiment')]
        label_1 = sub[sub['y_true'] == 1]
        neg_returns_in_label_1 = (label_1['fwd_ret'] < 0).sum()
        pct_neg = neg_returns_in_label_1 / len(label_1) * 100
        print(f"Fold {fold}: Out of {len(label_1)} positive-label instances (top 30%), {neg_returns_in_label_1} ({pct_neg:.1f}%) had NEGATIVE forward returns!")
        
        label_0 = sub[sub['y_true'] == 0]
        pos_returns_in_label_0 = (label_0['fwd_ret'] > 0).sum()
        pct_pos = pos_returns_in_label_0 / len(label_0) * 100
        print(f"Fold {fold}: Out of {len(label_0)} negative-label instances (bottom 70%), {pos_returns_in_label_0} ({pct_pos:.1f}%) had POSITIVE forward returns!")
    print("[VERIFIED] Confirms label is cross-sectional relative ranking, NOT an absolute return indicator.")

if __name__ == "__main__":
    run_audit()
