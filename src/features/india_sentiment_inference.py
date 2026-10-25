import pandas as pd
import numpy as np
import os
import time
try:
    import torch
    from transformers import pipeline
except ImportError:
    torch = None
    pipeline = None
import sys

try:
    from src.utils.data_loader import TARGET_STOCKS, normalize_indian_symbol
except ImportError:
    TARGET_STOCKS = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
    ]
    def normalize_indian_symbol(symbol):
        if pd.isna(symbol):
            return symbol
        s = str(symbol).strip().upper()
        if s.endswith(".NS") or s.endswith(".BO"):
            s = s.rsplit(".", 1)[0]
        return s

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'india'))
POSTS_CLEAN = os.path.join(PROCESSED_DIR, 'posts_clean.csv')
NEWS_SENTIMENT = os.path.join(PROCESSED_DIR, 'news_sentiment.csv')
LOG_FILE = os.path.join(PROCESSED_DIR, 'sentiment_inference_log.txt')

def setup_pipeline():
    if torch is None or pipeline is None:
        raise ImportError("torch and transformers are required to run the FinBERT sentiment pipeline.")
    device_id = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA (GPU)" if device_id == 0 else "CPU"
    try:
        # Use top_k=None instead of return_all_scores=True to get all scores properly
        finbert = pipeline("text-classification", model="ProsusAI/finbert", device=device_id, top_k=None)
    except Exception as e:
        print(f"Failed to load ProsusAI/finbert. Error: {e}")
        sys.exit(1)
    return finbert, device_name

def test_run(df_posts):
    print("=== SMALL TEST RUN (10 ROWS) ===")
    test_df = df_posts.head(10).copy()
    texts = test_df['text'].tolist()
    
    finbert, device_name = setup_pipeline()
    
    try:
        with torch.inference_mode():
            results = finbert(texts, batch_size=2, truncation=True, max_length=512)
    except Exception as e:
        print(f"Test run failed during inference: {e}")
        return False
        
    for res in results:
        # Ensure res is a list of dicts
        if not isinstance(res, list) or not all(isinstance(d, dict) for d in res):
            print(f"Test run failed: Unexpected output format from pipeline: {res}")
            return False
            
        scores = {d['label']: d['score'] for d in res}
        pos = scores.get('positive', 0.0)
        neg = scores.get('negative', 0.0)
        neu = scores.get('neutral', 0.0)
        
        prob_sum = pos + neg + neu
        if not np.isclose(prob_sum, 1.0, atol=1e-3):
            print(f"Test run failed: Probabilities do not sum to 1 ({prob_sum})")
            return False
            
        if pd.isna(pos) or pd.isna(neg) or pd.isna(neu):
            print("Test run failed: NaN probabilities detected.")
            return False
            
    print("Test run successful!")
    return True

def run_resumable_inference():
    print("=== FULL INFERENCE ===")
    df_posts = pd.read_csv(POSTS_CLEAN)
    df_posts['stock'] = df_posts['stock'].apply(normalize_indian_symbol)
    df_posts = df_posts[df_posts['stock'].isin(TARGET_STOCKS)]
    total_rows = len(df_posts)
    
    if os.path.exists(NEWS_SENTIMENT):
        print(f"Found existing output at {NEWS_SENTIMENT}. Resuming...")
        df_out = pd.read_csv(NEWS_SENTIMENT)
        processed_texts = set(df_out['text'].tolist())
        df_todo = df_posts[~df_posts['text'].isin(processed_texts)].copy()
    else:
        print("Starting from scratch...")
        df_out = pd.DataFrame(columns=['stock', 'post_time', 'text', 'sentiment_label', 'positive_prob', 'negative_prob', 'neutral_prob'])
        df_todo = df_posts.copy()
        
    remaining_rows = len(df_todo)
    completed_rows = total_rows - remaining_rows
    print(f"Total rows: {total_rows}")
    print(f"Already processed: {completed_rows}")
    print(f"Remaining to process: {remaining_rows}")
    
    if remaining_rows == 0:
        print("All inference already completed.")
        return
        
    finbert, device_name = setup_pipeline()
    batch_size = 16
    chunk_size = 128
    
    with open(LOG_FILE, 'w' if completed_rows == 0 else 'a') as f:
        if completed_rows == 0:
            f.write("=== SENTIMENT INFERENCE LOG ===\n")
            f.write(f"Model: ProsusAI/finbert\n")
            f.write(f"Device: {device_name}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Total rows: {total_rows}\n")
            f.write("--------------------------------\n")
            
    start_time = time.time()
    
    # Process in chunks to save progress
    for i in range(0, remaining_rows, chunk_size):
        chunk_df = df_todo.iloc[i:i+chunk_size].copy()
        texts = chunk_df['text'].tolist()
        
        try:
            with torch.inference_mode():
                results = finbert(texts, batch_size=batch_size, truncation=True, max_length=512)
                
            pos_probs = []
            neg_probs = []
            neu_probs = []
            labels = []
            
            for res in results:
                scores = {d['label']: d['score'] for d in res}
                pos = scores.get('positive', 0.0)
                neg = scores.get('negative', 0.0)
                neu = scores.get('neutral', 0.0)
                
                pos_probs.append(pos)
                neg_probs.append(neg)
                neu_probs.append(neu)
                
                max_label = max(scores, key=scores.get)
                labels.append(max_label)
                
            chunk_df['sentiment_label'] = labels
            chunk_df['positive_prob'] = pos_probs
            chunk_df['negative_prob'] = neg_probs
            chunk_df['neutral_prob'] = neu_probs
            
            df_out = pd.concat([df_out, chunk_df], ignore_index=True)
            df_out.to_csv(NEWS_SENTIMENT, index=False)
            
            completed_rows += len(chunk_df)
            elapsed = time.time() - start_time
            rate = len(chunk_df) / elapsed if elapsed > 0 else 0
            
            msg = f"Processed {completed_rows}/{total_rows} ({(completed_rows/total_rows)*100:.1f}%) | Chunk saved."
            print(msg)
            
            with open(LOG_FILE, 'a') as f:
                f.write(f"{msg} | Elapsed: {elapsed:.1f}s\n")
                
            start_time = time.time() # Reset for next chunk
            
        except Exception as e:
            err_msg = f"ERROR at row index {i} to {i+chunk_size}: {e}"
            print(err_msg)
            with open(LOG_FILE, 'a') as f:
                f.write(err_msg + "\n")
            sys.exit(1)

def run_validation():
    print("=== RUNNING VALIDATION ===")
    df = pd.read_csv(NEWS_SENTIMENT)
    total = len(df)
    
    pos = (df['sentiment_label'] == 'positive').sum()
    neg = (df['sentiment_label'] == 'negative').sum()
    neu = (df['sentiment_label'] == 'neutral').sum()
    
    prob_sums = df['positive_prob'] + df['negative_prob'] + df['neutral_prob']
    violations = (~np.isclose(prob_sums, 1.0, atol=1e-3)).sum()
    min_prob = df[['positive_prob', 'negative_prob', 'neutral_prob']].min().min()
    max_prob = df[['positive_prob', 'negative_prob', 'neutral_prob']].max().max()
    missing_prob = df[['positive_prob', 'negative_prob', 'neutral_prob']].isnull().sum().sum()
    
    print(f"Total Rows: {total}")
    print(f"Positive: {pos} ({pos/total*100:.1f}%)")
    print(f"Negative: {neg} ({neg/total*100:.1f}%)")
    print(f"Neutral: {neu} ({neu/total*100:.1f}%)")
    print(f"Prob Sum Violations: {violations}")
    print(f"Min Prob: {min_prob:.4f}, Max Prob: {max_prob:.4f}")
    print(f"Missing Probs: {missing_prob}")
    
    print("\nStock Coverage:")
    df['date'] = pd.to_datetime(df['post_time']).dt.date
    cov = df.groupby('stock').agg(
        headline_count=('text', 'count'),
        unique_news_dates=('date', 'nunique'),
        first_news_date=('date', 'min'),
        last_news_date=('date', 'max'),
        positive_count=('sentiment_label', lambda x: (x == 'positive').sum()),
        negative_count=('sentiment_label', lambda x: (x == 'negative').sum()),
        neutral_count=('sentiment_label', lambda x: (x == 'neutral').sum())
    )
    print(cov)
    
if __name__ == '__main__':
    df_posts = pd.read_csv(POSTS_CLEAN)
    df_posts['stock'] = df_posts['stock'].apply(normalize_indian_symbol)
    df_posts = df_posts[df_posts['stock'].isin(TARGET_STOCKS)]
    if test_run(df_posts):
        run_resumable_inference()
        run_validation()
