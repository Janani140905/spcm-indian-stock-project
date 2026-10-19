import pandas as pd
import numpy as np
from src.utils.data_loader import load_prices, load_ff5_factors
from src.features.financial_features import compute_abnormal_returns, build_control_variables
from src.models.spcm import build_price_features, build_labels, shift_sentiment_dates, build_fusion_matrix, train_spcm, predict_proba
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("Loading data for diagnostics...")
prices = load_prices("data/processed/india/prices.csv")
ff5 = load_ff5_factors("data/processed/india/ff4_factors.csv")
sentiment_ind = pd.read_csv("data/processed/india/daily_sentiment_features.csv")
sentiment_ind["date"] = pd.to_datetime(sentiment_ind["date"]).dt.normalize()

prices = prices[(prices["date"] >= "2018-01-01") & (prices["date"] <= "2020-12-31")].copy()

ar_df = compute_abnormal_returns(prices, ff5, window=60)
control_feat = build_control_variables(ar_df)
price_feat = build_price_features(prices)
labels = build_labels(prices)
sentiment_ind = shift_sentiment_dates(sentiment_ind, price_feat)
fusion = build_fusion_matrix(price_feat, sentiment_ind, None, labels, control_feat=control_feat)
fusion = fusion.sort_values("date").reset_index(drop=True)

folds = [
    {"train_start": "2018-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
    {"train_start": "2018-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
]

models = ['knn', 'rf', 'nn']
focus_features = ["Sent", "PV", "AI", "news_available", "headline_count", "AR", "CAR_t1_t5", "CAR_t6_t20", "Volatility"]

for i, f in enumerate(folds):
    print(f"\n{'='*50}\nFOLD {i+1}\n{'='*50}")
    train_mask = (fusion["date"] >= f["train_start"]) & (fusion["date"] <= f["train_end"])
    test_mask = (fusion["date"] >= f["test_start"]) & (fusion["date"] <= f["test_end"])
    
    train_df = fusion[train_mask].dropna(subset=["label"]).copy()
    test_df = fusion[test_mask].dropna(subset=["label"]).copy()
    
    print("\n--- TASK 1: Class Distribution ---")
    pos_train = train_df['label'].sum()
    neg_train = len(train_df) - pos_train
    pct_train = pos_train / len(train_df) * 100
    
    pos_test = test_df['label'].sum()
    neg_test = len(test_df) - pos_test
    pct_test = pos_test / len(test_df) * 100
    
    print(f"TRAIN: rows={len(train_df)}, pos={pos_train:.0f}, neg={neg_train:.0f}, pos_pct={pct_train:.1f}%")
    print(f"TEST:  rows={len(test_df)}, pos={pos_test:.0f}, neg={neg_test:.0f}, pos_pct={pct_test:.1f}%")

    print("\n--- TASK 4: Feature Matrix Diagnostics (Train set after Imputation) ---")
    dummy_model, scaler, imputer, feature_cols = train_spcm(fusion, classifier="nn", train_mask=train_mask)
    X_raw = train_df[feature_cols].values.astype(float)
    X_imputed = imputer.transform(X_raw)
    df_imputed = pd.DataFrame(X_imputed, columns=feature_cols)
    
    constant_feats = [c for c in feature_cols if df_imputed[c].nunique() <= 1]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Constant features: {len(constant_feats)} -> {constant_feats}")
    print(f"Non-constant features: {len(feature_cols) - len(constant_feats)}")
    
    print("Focus Features Stats:")
    for c in focus_features:
        if c in df_imputed.columns:
            val = df_imputed[c]
            print(f"  {c:15s} | Var: {val.var():.6f} | Min: {val.min():.4f} | Max: {val.max():.4f} | UniqueVals: {val.nunique()}")
        else:
            print(f"  {c} NOT FOUND")
            
    print("\n--- TASK 6: Simple Baselines ---")
    y_true = test_df['label'].values
    y_maj = np.zeros_like(y_true)
    maj_acc = accuracy_score(y_true, y_maj)
    maj_prec = precision_score(y_true, y_maj, zero_division=0)
    maj_rec = recall_score(y_true, y_maj, zero_division=0)
    maj_f1 = f1_score(y_true, y_maj, zero_division=0)
    print(f"Majority Baseline (All 0s) -> Acc: {maj_acc:.4f} | Prec: {maj_prec:.4f} | Rec: {maj_rec:.4f} | F1: {maj_f1:.4f}")

    print("\n--- TASK 2, 3, 5: Model Predictions ---")
    for m in models:
        print(f"\n> MODEL: {m.upper()}")
        model, scaler, imputer, feature_cols = train_spcm(fusion, classifier=m, train_mask=train_mask)
        test_proba = predict_proba(model, scaler, imputer, feature_cols, test_df)
        
        y_prob = test_proba["proba"].values
        y_pred = (y_prob >= 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        n_pos_pred = y_pred.sum()
        n_neg_pred = len(y_pred) - n_pos_pred
        pct_pos_pred = n_pos_pred / len(y_pred) * 100
        
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Predictions: {n_pos_pred} pos, {n_neg_pred} neg ({pct_pos_pred:.1f}% pos)")
        print(f"Probabilities: Mean={y_prob.mean():.4f}, Min={y_prob.min():.4f}, Max={y_prob.max():.4f}")
