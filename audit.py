import pandas as pd
import numpy as np
import sys
import os

print("--- TASK 1: Prices ---")
p_path = "data/processed/india/prices.csv"
if os.path.exists(p_path):
    df_p = pd.read_csv(p_path)
    print("File path:", p_path)
    print("Row count:", len(df_p))
    print("Date range:", df_p['date'].min(), "to", df_p['date'].max())
    print("10-stock coverage:", df_p['stock'].nunique(), "stocks")
    print("Duplicate stock/date count:", df_p.duplicated(['stock', 'date']).sum())
    print("Missing OHLCV values:", df_p[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum())
    print("Stock names:", df_p['stock'].unique()[:5])
else:
    print("FAIL: prices.csv not found")

print("\n--- TASK 2: Factors ---")
f_path = "data/processed/india/ff4_factors.csv"
if os.path.exists(f_path):
    df_f = pd.read_csv(f_path)
    print("File path:", f_path)
    print("Row count:", len(df_f))
    print("Date range:", df_f['date'].min(), "to", df_f['date'].max())
    print("Columns:", df_f.columns.tolist())
    
    expected = ['date', 'mkt_rf', 'smb', 'hml', 'wml', 'rf']
    print("Columns match expected:", all(c in df_f.columns for c in expected))
    print("Max absolute factor value (check conversion):", df_f[['mkt_rf', 'smb', 'hml', 'wml', 'rf']].abs().max().max())
else:
    print("FAIL: ff4_factors.csv not found")

print("\n--- TASK 5: Small Real-Data Fusion Test ---")
from src.models.spcm import build_price_features, build_labels, build_fusion_matrix
import warnings
warnings.filterwarnings('ignore')

df_p_small = df_p[df_p['stock'] == 'RELIANCE'].copy()
df_p_small['date'] = pd.to_datetime(df_p_small['date']).dt.normalize()
df_f_small = df_f.copy()

price_feat = build_price_features(df_p_small)
labels = build_labels(df_p_small)

s_path = "data/processed/india/daily_sentiment_features.csv"
if os.path.exists(s_path):
    df_s = pd.read_csv(s_path)
    df_s_small = df_s[df_s['stock'] == 'RELIANCE'].copy()
    df_s_small['date'] = pd.to_datetime(df_s_small['date']).dt.normalize()
else:
    print("No daily_sentiment_features.csv")
    df_s_small = pd.DataFrame()

fusion = build_fusion_matrix(price_feat, df_s_small, None, labels)
print("1. Rows:", len(fusion))
print("2. Unique stocks:", fusion['stock'].nunique())
print("3. Date range:", fusion['date'].min(), "to", fusion['date'].max())
print("4. All feature columns:", fusion.columns.tolist())
print("5. Factor-derived columns:", [c for c in fusion.columns if 'AR' in c or 'CAR' in c or 'mkt' in c])
print("6. Sentiment columns:", [c for c in fusion.columns if c in ['Sent', 'PV', 'AI', 'Homo']])
print("7. Label column:", 'label' in fusion.columns)
print("8. Duplicate stock/date count:", fusion.duplicated(['stock', 'date']).sum())
print("9. Missing values:\n", fusion.isnull().sum())
print("10. AR/CAR present:", any('AR' in c or 'CAR' in c for c in fusion.columns))
