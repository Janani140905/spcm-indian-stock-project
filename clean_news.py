import pandas as pd
import numpy as np
import os

PROCESSED_DIR = r"C:\Users\Admin\Documents\Stock market project\spcm_project\data\processed\india"
RAW_DIR = r"C:\Users\Admin\Documents\Stock market project\spcm_project\data\raw\india"
REPORT_PATH = os.path.join(PROCESSED_DIR, "news_cleaning_report.txt")
CLEAN_POSTS_PATH = os.path.join(PROCESSED_DIR, "posts_clean.csv")

TARGET_STOCKS = [
    'RELIANCE', 'TCS', 'INFY', 'ICICIBANK', 
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
]

def main():
    report_lines = ["=== NEWS CLEANING REPORT ===\n"]
    
    # 1. Investigate duplicates
    report_lines.append("### Source-file overlap explanation")
    report_lines.append("The 3,834 exact duplicates are caused by overlapping source files in the Kaggle dataset archive. The root contains `final_news_sentiment_analysis.csv` (spanning 2008-2025 with ~13k rows). The subdirectories contain 5 smaller `news_sentiment_analysis.csv` files that are exact subsets of the final file. When combined, this resulted in massive duplication of the same observations.\n")
    
    # Load processed data
    df_posts = pd.read_csv(os.path.join(PROCESSED_DIR, "posts.csv"))
    df_prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices.csv"))
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    orig_count = len(df_posts)
    
    # 2. Remove true duplicate observations
    df_posts_clean = df_posts.drop_duplicates(subset=['stock', 'post_time', 'text'])
    dupes_removed = orig_count - len(df_posts_clean)
    
    # 3. Handle missing dates
    missing_dates_mask = df_posts_clean['post_time'].isnull()
    missing_dates_count = missing_dates_mask.sum()
    df_posts_clean = df_posts_clean[~missing_dates_mask]
    
    # 4. Handle very short headlines
    df_posts_clean['text_len'] = df_posts_clean['text'].astype(str).str.len()
    short_mask = df_posts_clean['text'] == '[Removed]'
    short_removed = short_mask.sum()
    df_posts_clean = df_posts_clean[~short_mask]
    
    df_posts_clean = df_posts_clean.drop(columns=['text_len'])
    df_posts_clean['post_time'] = pd.to_datetime(df_posts_clean['post_time'])
    
    # 5. Create final clean news dataset
    # Must only contain stock, post_time, text
    df_posts_clean = df_posts_clean[['stock', 'post_time', 'text']]
    df_posts_clean = df_posts_clean.sort_values(['stock', 'post_time'])
    
    df_posts_clean.to_csv(CLEAN_POSTS_PATH, index=False)
    
    final_count = len(df_posts_clean)
    
    report_lines.append(f"### Original row count\n{orig_count}\n")
    report_lines.append(f"### Duplicate rows removed\n{dupes_removed}\n")
    report_lines.append(f"### Missing-date rows removed\n{missing_dates_count} (Source dates were entirely blank and could not be recovered)\n")
    report_lines.append(f"### Short/malformed rows removed\n{short_removed} (Headline text was literally '[Removed]', signifying a deleted post/article)\n")
    report_lines.append(f"### Final row count\n{final_count}\n")
    
    # 6. Final stock counts & date ranges
    report_lines.append("### Final stock counts")
    counts = df_posts_clean['stock'].value_counts()
    for s, c in counts.items():
        report_lines.append(f"{s}: {c}")
    report_lines.append("")
    
    report_lines.append("### Final stock date ranges")
    for s in TARGET_STOCKS:
        s_df = df_posts_clean[df_posts_clean['stock'] == s]
        if len(s_df) > 0:
            report_lines.append(f"{s}: {s_df['post_time'].min().date()} to {s_df['post_time'].max().date()}")
        else:
            report_lines.append(f"{s}: NO DATA")
    report_lines.append("")
    
    # Cross-stock duplicates remaining
    cross_dupes = df_posts_clean.groupby(['post_time', 'text']).filter(lambda x: x['stock'].nunique() > 1)
    num_cross = cross_dupes['text'].nunique()
    report_lines.append(f"### Cross-stock repeated headlines\nKept {num_cross} unique headlines that were broadcasted across multiple stocks on the same day (e.g. general market index news). These are legitimate stock-specific observations.\n")
    
    report_lines.append("### Remaining data limitations\nNews density is sparse for many stocks. No imputation of missing days or '0' sentiment was performed at this stage. Homophily remains disabled.\n")
    
    # 7. Recalculate coverage after cleaning
    report_lines.append("\n=== FINAL COVERAGE METRICS ===")
    
    for s in TARGET_STOCKS:
        s_news = df_posts_clean[df_posts_clean['stock'] == s]
        s_prices = df_prices[df_prices['stock'] == s]
        
        n_rows = len(s_news)
        p_dates = set(s_prices['date'].dt.date)
        
        if n_rows > 0:
            n_dates = set(s_news['post_time'].dt.date)
            unique_n = len(n_dates)
            f_date = s_news['post_time'].min().date()
            l_date = s_news['post_time'].max().date()
            
            trading_days_with_news = len(p_dates.intersection(n_dates))
            total_trading_days = len(p_dates)
            pct = round((trading_days_with_news / total_trading_days) * 100, 2) if total_trading_days > 0 else 0
            
            report_lines.append(f"\n{s}:")
            report_lines.append(f"  - Total clean news rows: {n_rows}")
            report_lines.append(f"  - Unique news dates: {unique_n}")
            report_lines.append(f"  - First date: {f_date}")
            report_lines.append(f"  - Last date: {l_date}")
            report_lines.append(f"  - News trading days: {trading_days_with_news}")
            report_lines.append(f"  - % of price trading days with news: {pct}%")
        else:
            report_lines.append(f"\n{s}: NO NEWS DATA")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print("Report generated.")
    
    # Final assertions
    assert df_posts_clean.duplicated(subset=['stock', 'post_time', 'text']).sum() == 0, "Exact duplicates remain!"
    assert df_posts_clean['stock'].isnull().sum() == 0, "Missing stock!"
    assert df_posts_clean['post_time'].isnull().sum() == 0, "Missing post_time!"
    assert df_posts_clean['text'].isnull().sum() == 0, "Missing text!"
    assert 'Sentiment' not in df_posts_clean.columns, "Leaked Sentiment found!"
    assert 'HDFCBANK' not in df_posts_clean['stock'].unique(), "HDFCBANK was not excluded!"
    print("All validation checks passed.")

if __name__ == '__main__':
    main()
