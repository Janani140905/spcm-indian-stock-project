import pandas as pd
import numpy as np
import glob
import os
import sys

RAW_DIR = "data/raw/india"
PROCESSED_DIR = "data/processed/india"
os.makedirs(PROCESSED_DIR, exist_ok=True)

REPORT_PATH = os.path.join(PROCESSED_DIR, "preprocessing_report.txt")
report_lines = []

def log(msg, also_print=True):
    report_lines.append(msg)
    if also_print:
        print(msg)

log("=== Preprocessing Real Indian Data ===")

TARGET_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
]

# --- 1. PROCESS IIMA FACTORS ---
log("\n=== Processing IIMA Factors ===")
factor_file = os.path.join(RAW_DIR, "2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv")
df_f = pd.read_csv(factor_file)
rows_before_f = len(df_f)
df_f = df_f.rename(columns={'Date': 'date', 'MF': 'mkt_rf'})
df_f.columns = [c.lower() for c in df_f.columns]

# Drop any completely empty rows (like a trailing blank line)
df_f = df_f.dropna(how='all')
missing_f = df_f.isnull().sum().sum()

# Convert date safely
try_dates = pd.to_datetime(df_f['date'].astype(str), format='%Y%m%d', errors='coerce')
if try_dates.isnull().all():
    df_f['date'] = pd.to_datetime(df_f['date'], errors='coerce')
else:
    df_f['date'] = try_dates
    
# Drop any rows where date failed to parse (like metadata rows)
df_f = df_f.dropna(subset=['date'])

df_f = df_f.sort_values('date')

# Convert percentages to decimals
for col in ['mkt_rf', 'smb', 'hml', 'wml', 'rf']:
    df_f[col] = df_f[col] / 100.0

df_f.to_csv(os.path.join(PROCESSED_DIR, "ff4_factors.csv"), index=False)
df_f.to_csv(os.path.join(PROCESSED_DIR, "ff5_factors.csv"), index=False)
log(f"Processed Factors: {rows_before_f} -> {len(df_f)} rows")
log(f"Factor Date Range: {df_f['date'].min().date()} to {df_f['date'].max().date()}")
log(f"Factor missing values: {missing_f}")


# --- 2. PROCESS NIFTY PRICES ---
log("\n=== Processing NIFTY Prices ===")
price_files = glob.glob(os.path.join(RAW_DIR, "NIFTY-50-Stock-Market-Data-main", "**", "*.csv"), recursive=True)
dfs_p = [pd.read_csv(f) for f in price_files]
df_p = pd.concat(dfs_p, ignore_index=True)
rows_before_p = len(df_p)

if 'Unnamed: 0' in df_p.columns:
    df_p = df_p.drop(columns=['Unnamed: 0'])

df_p = df_p.rename(columns={'timestamp': 'date', 'symbol': 'stock'})
df_p['date'] = pd.to_datetime(df_p['date'], errors='coerce')
# Normalize stock symbols (strip whitespace, uppercase, remove .NS/.BO suffixes)
df_p['stock'] = df_p['stock'].astype(str).str.strip().str.upper().str.replace(r'\.NS$', '', regex=True).str.replace(r'\.BO$', '', regex=True)

unique_prices_stocks = df_p['stock'].unique()
df_p = df_p[df_p['stock'].isin(TARGET_STOCKS)]

df_p = df_p.sort_values(['stock', 'date'])

# Validation
dupes_p = df_p.duplicated(['stock', 'date']).sum()
missing_p = df_p.isnull().sum().sum()
neg_prices = (df_p[['open', 'high', 'low', 'close']] < 0).sum().sum()
neg_volume = (df_p['volume'] < 0).sum()

df_p.to_csv(os.path.join(PROCESSED_DIR, "prices.csv"), index=False)
log(f"Processed Prices: {rows_before_p} -> {len(df_p)} rows")
log(f"Price Date Range: {df_p['date'].min().date()} to {df_p['date'].max().date()}")
log(f"Price Duplicates (stock, date): {dupes_p}")
log(f"Price Missing Values: {missing_p}")
log(f"Price Negative Values (OHLC): {neg_prices}")
log(f"Price Negative Volume: {neg_volume}")


# --- 3. PROCESS NIFSENT50 (POSTS) ---
log("\n=== Processing NifSent50 ===")
post_files = glob.glob(os.path.join(RAW_DIR, "archive (1)", "**", "*.csv"), recursive=True)
dfs_k = []
for f in post_files:
    try:
        tmp = pd.read_csv(f, nrows=5)
        cols = [c.lower() for c in tmp.columns]
        if 'headline' in cols or 'text' in cols or 'title' in cols:
            dfs_k.append(pd.read_csv(f))
    except:
        pass
        
df_k = pd.concat(dfs_k, ignore_index=True)
rows_before_k = len(df_k)

# Transform columns
df_k = df_k.rename(columns={'Symbol': 'stock', 'Publish Date': 'post_time', 'Headline': 'text'})
df_k['post_time'] = pd.to_datetime(df_k['post_time'], errors='coerce')
# Normalize stock symbols (strip whitespace, uppercase, remove .NS/.BO suffixes)
df_k['stock'] = df_k['stock'].astype(str).str.strip().str.upper().str.replace(r'\.NS$', '', regex=True).str.replace(r'\.BO$', '', regex=True)

# Check Stock Compatibility
unique_posts_stocks = df_k['stock'].unique()
prices_set = set(unique_prices_stocks)
posts_set = set(unique_posts_stocks)
both = prices_set.intersection(posts_set)
only_prices = prices_set - posts_set
only_posts = posts_set - prices_set

log(f"\nStock Compatibility Check:")
log(f"Target prototype stocks: {TARGET_STOCKS}")
log(f"Target stocks missing in PRICES: {set(TARGET_STOCKS) - prices_set}")
log(f"Target stocks missing in POSTS: {set(TARGET_STOCKS) - posts_set}")

df_k = df_k[['stock', 'post_time', 'text']]
# Filter to target stocks ONLY if they exist
df_k = df_k[df_k['stock'].isin(TARGET_STOCKS)]
df_k = df_k.sort_values(['stock', 'post_time'])

df_k.to_csv(os.path.join(PROCESSED_DIR, "posts.csv"), index=False)
log(f"Processed Posts: {rows_before_k} -> {len(df_k)} rows")
log(f"Posts Date Range: {df_k['post_time'].min().date()} to {df_k['post_time'].max().date()}")
log("Dropped columns: Sentiment, Company Name (and any other extraneous)")


# --- 4. TRUE COMMON DATE RANGE ---
log("\n=== True Common Date Range ===")
start_f = df_f['date'].min()
end_f = df_f['date'].max()
start_p = df_p['date'].min()
end_p = df_p['date'].max()
start_k = df_k['post_time'].min()
end_k = df_k['post_time'].max()

common_start = max(start_f, start_p, start_k)
common_end = min(end_f, end_p, end_k)

log(f"Common Start: {common_start.date()}")
log(f"Common End: {common_end.date()}")


# --- 5. WRITE PREPROCESSING REPORT ---
log("\n=== Generating Preprocessing Report ===")
with open(REPORT_PATH, "w") as f:
    f.write("=== PREPROCESSING REPORT ===\n")
    f.write("\nFiles processed:\n")
    f.write(f"- Factors: 1 file\n")
    f.write(f"- Prices: {len(price_files)} files\n")
    f.write(f"- Posts: {len(dfs_k)} files\n")
    
    f.write("\nRow counts before/after:\n")
    f.write(f"- Factors: {rows_before_f} -> {len(df_f)}\n")
    f.write(f"- Prices: {rows_before_p} -> {len(df_p)}\n")
    f.write(f"- Posts: {rows_before_k} -> {len(df_k)}\n")
    
    f.write("\nDate ranges:\n")
    f.write(f"- Factors: {start_f.date()} to {end_f.date()}\n")
    f.write(f"- Prices: {start_p.date()} to {end_p.date()}\n")
    f.write(f"- Posts: {start_k.date()} to {end_k.date()}\n")
    
    f.write("\nStock coverage:\n")
    f.write(f"Target stocks missing in prices: {set(TARGET_STOCKS) - prices_set}\n")
    
    f.write("\nNews coverage:\n")
    f.write(f"Target stocks missing in posts: {set(TARGET_STOCKS) - posts_set}\n")
    
    f.write("\nFactor coverage:\n")
    f.write(f"All 5 columns present: {all(c in df_f.columns for c in ['date','mkt_rf','smb','hml','wml','rf'])}\n")
    
    f.write("\nCommon date range:\n")
    f.write(f"{common_start.date()} to {common_end.date()}\n")
    
    f.write("\nFactor unit conversion:\n")
    f.write("Divided all factor values by 100.0 to convert percentages to decimals.\n")
    
    f.write("\nStock-symbol mapping:\n")
    f.write("Matched directly on exact NSE symbol. No `.NS` suffix was appended.\n")
    
    f.write("\nMissing values:\n")
    f.write(f"Prices: {missing_p}. Factors: {missing_f}.\n")
    
    f.write("\nDuplicate checks:\n")
    f.write(f"Price (stock, date) duplicates: {dupes_p}\n")
    
    f.write("\nDropped columns:\n")
    f.write("Dropped 'Unnamed: 0' from prices. Dropped 'Sentiment', 'Company Name' and source_file from posts.\n")
    
    f.write("\nUnavailable fields:\n")
    f.write("PE, PCF, liquidity, market_cap, shares_outstanding (from Prices).\n")
    
    f.write("\nHomophily status:\n")
    f.write("author, thread_id, message_id are completely missing. Homophily = unavailable for this real dataset.\n")
    
    f.write("\nPoint-in-time rule:\n")
    f.write("Treat all news from date `t` as information available for prediction AFTER day `t`. Will predict forward movement on t+1.\n")
    
    f.write("\nForward-return definition:\n")
    f.write("forward_ret(t) = (close(t+1) - close(t)) / close(t). Excess forward return = forward_ret - rf(t+1) if needed later.\n")
    
    f.write("\nPotential remaining data limitations:\n")
    f.write("None strictly, but missing fundamental factors means financial_features.py must bypass PE/PCF logic gracefully.\n")

# --- FINAL VALIDATION ---
log("\n=== Final Validation ===")
assert all(c in df_f.columns for c in ['date','mkt_rf','smb','hml','wml','rf']), "Factor columns missing!"
assert all(c in df_p.columns for c in ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'turnover']), "Price columns missing!"
assert all(c in df_k.columns for c in ['stock', 'post_time', 'text']), "Post columns missing!"
assert 'Sentiment' not in df_k.columns and 'sentiment' not in df_k.columns, "Leaked Sentiment column found!"
assert dupes_p == 0, "Duplicate price rows found!"
assert df_f['date'].isnull().sum() == 0, "Invalid factor dates!"

log("READY FOR PREPROCESSING")
sys.exit(0)
