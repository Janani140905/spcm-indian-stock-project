import pandas as pd
import numpy as np
from src.utils.data_loader import load_prices, load_ff5_factors
from src.features.financial_features import compute_abnormal_returns, build_control_variables
from src.models.spcm import build_price_features, build_labels, build_fusion_matrix

print("Loading data...")
prices = load_prices("data/processed/india/prices.csv")
ff5 = load_ff5_factors("data/processed/india/ff4_factors.csv")
sentiment_ind = pd.read_csv("data/processed/india/daily_sentiment_features.csv")
sentiment_ind["date"] = pd.to_datetime(sentiment_ind["date"]).dt.normalize()

# Filter to a small date range to keep it fast
prices = prices[(prices["date"] >= "2018-01-01") & (prices["date"] <= "2019-12-31")].copy()

print("Building AR and control features...")
ar_df = compute_abnormal_returns(prices, ff5, window=60)
control_feat = build_control_variables(ar_df)

print("Building fusion matrix...")
price_feat = build_price_features(prices)
labels = build_labels(prices)
fusion = build_fusion_matrix(price_feat, sentiment_ind, None, labels, control_feat=control_feat)

print("\n--- FUSION VALIDATION OUTPUT ---")
print("Total fusion rows:", len(fusion))
print("Unique stocks:", fusion['stock'].nunique())
print("Date range:", fusion['date'].min(), "to", fusion['date'].max())
print("Duplicate stock/date count:", fusion.duplicated(['stock', 'date']).sum())

cols = fusion.columns.tolist()
print("\nFeature Columns Details:")
price_cols = [c for c in cols if c in price_feat.columns and c not in ['stock', 'date']]
print("Price feature columns:", len(price_cols), "columns ->", price_cols[:5], "...")

sent_cols = [c for c in cols if c in ["Sent", "PV", "AI", "news_available", "headline_count"]]
print("Sentiment columns:", sent_cols)

ar_cols = [c for c in cols if c == "AR"]
print("AR columns:", ar_cols)

car_cols = [c for c in cols if "CAR" in c or c == "Volatility"]
print("CAR columns:", car_cols)

print("Label column present:", "label" in cols)
print("fwd_ret column present:", "fwd_ret" in cols)

print("\nMissing Counts:")
print("Missing Sent/PV/AI:", fusion[["Sent", "PV", "AI"]].isnull().sum().to_dict())
if ar_cols:
    print("Missing AR:", fusion[ar_cols].isnull().sum().to_dict())
if car_cols:
    print("Missing CAR:", fusion[car_cols].isnull().sum().to_dict())

print("\nVerifications:")
print("Is AR actually present?", "AR" in cols)
print("Is at least one CAR/rolling feature present?", len(car_cols) > 0)
print("Is fwd_ret present?", "fwd_ret" in cols)
print("Is label present?", "label" in cols)
