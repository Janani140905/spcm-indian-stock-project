# SPCM: Sentiment and Price Combined Model
Implementation scaffold for *"SPCM: A Machine Learning Approach for Sentiment-Based Stock Recommendation System"* (Wang & Chen, IEEE Access 2024).

This repo reproduces the paper's pipeline end-to-end. Because the original Guba/Wind/CSMAR
datasets are proprietary, you'll plug your own data sources into the loader functions
(clearly marked `TODO` in `src/utils/data_loader.py`). Everything else — sentiment model,
feature engineering, FF5 abnormal returns, SPCM fusion model, and backtest — is fully
implemented and runnable on your own data or the included synthetic-data demo.

## 1. Project structure

```
spcm_project/
├── data/                          # raw & processed data (you supply raw)
├── src/
│   ├── sentiment/
│   │   └── bert_sentiment.py      # BERT fine-tuning + inference for 3-class sentiment
│   ├── features/
│   │   ├── sentiment_features.py  # Sent, PV, AI  (paper eq. 1-3)
│   │   ├── homophily.py           # Homo          (paper eq. 4)
│   │   └── financial_features.py  # FF5 abnormal returns, CAR (paper eq. 5-7)
│   ├── models/
│   │   ├── baseline_models.py     # KNN / RF / NN on price-only or price+sentiment
│   │   └── spcm.py                # SPCM fusion pipeline + labeling + recommendation
│   ├── backtest/
│   │   └── backtest.py            # rolling-window backtest, AUC, cumulative return
│   └── utils/
│       └── data_loader.py         # data schemas + loaders (fill in your sources)
├── main.py                        # orchestrates the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## 2. Implementation steps (maps 1:1 onto the paper's sections)

### Step 0 — Environment
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Step 1 — Collect data (paper §III.A)
You need three sources:
1. **Social media posts** (paper: Guba) — thread_id, message_id, content, post_time,
   author, stock_code. → `data/raw/posts.csv`
2. **Stock price/trading data** (paper: Wind) — open/high/low/close, volume, turnover,
   for each stock/day. → `data/raw/prices.csv`
3. **Financial factor data** (paper: CSMAR) — FF5 factors (Mkt-RF, SMB, HML, RMW, CMA),
   risk-free rate, and firm fundamentals (Size, BM, ROA). → `data/raw/ff5_factors.csv`,
   `data/raw/fundamentals.csv`

Fill in `src/utils/data_loader.py` to read these into the standard DataFrame schemas
documented at the top of that file. (StockTwits/Reddit/Weibo APIs, Tushare/AkShare/yfinance
for prices, and Ken French's data library or a local computation for FF5 factors are common
substitutes if you don't have Guba/Wind/CSMAR access.)

### Step 2 — Label & fine-tune BERT for sentiment (paper §III.B)
```bash
python -m src.sentiment.bert_sentiment train \
    --labeled_csv data/labeled/annotated_posts.csv \
    --output_dir models/bert_sentiment
```
- `annotated_posts.csv` needs columns `text,label` with label ∈ {-1 (bearish), 0 (neutral), 1 (bullish)}.
- Uses `bert-base-chinese` (swap for `bert-base-uncased` if your posts are in English).
- Trains with a 7:3 train/test split, reports accuracy (paper reports 84.12%).

Then run inference on all posts:
```bash
python -m src.sentiment.bert_sentiment predict \
    --posts_csv data/raw/posts.csv \
    --model_dir models/bert_sentiment \
    --output_csv data/processed/posts_with_sentiment.csv
```

### Step 3 — Build sentiment & homophily indicators (paper §III.C, eq. 1-4)
```bash
python -m src.features.sentiment_features
python -m src.features.homophily
```
Produces daily `Sent_{s,t}`, `PV_{s,t}`, `AI_{s,t}` per stock, and per-user/per-stock
`Homo_{i,s,t}`, aggregated to stock level.

### Step 4 — Compute FF5 abnormal returns & CAR (paper §III.C.3, eq. 5-7)
```bash
python -m src.features.financial_features
```
Rolling 250-trading-day OLS of the FF5 model per stock → abnormal return `AR_{s,t}`,
and `CAR[t1,t2]`.

### Step 5 — (Optional) Reproduce Table 3 regression — homophily's moderating effect
See `src/features/financial_features.py::run_homophily_regression` (statsmodels panel OLS
with stock & date fixed effects, clustered SE by firm and year-month) — this replicates
equation (8) and Table 3.

### Step 6 — Train SPCM (paper §III.D / §V, Figure 3)
```bash
python -m src.models.spcm train \
    --start 2018-01-01 --end 2020-12-31 \
    --classifier nn
```
- Labels: top 30% of forward returns → 1, bottom 70% → 0.
- Fuses price features (with 1/5/10/30-day lags) + sentiment features (Sent, PV, AI, Homo)
  into one matrix, standardized to [-1, 1].
- Trains KNN / RF / NN (paper found NN best, RF second).

### Step 7 — Backtest & recommend (paper §V.B)
```bash
python -m src.backtest.backtest run \
    --backtest_start 2018-07-01 --backtest_end 2020-12-31 \
    --top_n 50
```
- Rolling 6-month training window, daily re-fit, top-50-probability stocks recommended.
- Reports AUC per half-year (reproduces Figure 5) and cumulative return vs CSI300
  (reproduces Figure 6).

### Step 8 — Sensitivity analysis (paper §IV.C, Figure 7)
```bash
python -m src.models.spcm sensitivity --feature PV
python -m src.models.spcm sensitivity --feature AI
python -m src.models.spcm sensitivity --feature Homo
```

## 3. Quick end-to-end demo (synthetic data)
`main.py` runs the whole pipeline on synthetic data so you can sanity-check the code
before wiring in real data:
```bash
python main.py --demo
```

## 4. Notes / deviations you may want to make
- The paper's universe is CSI300 (286 stocks, 2018-2020); swap in any equity universe.
- If you don't have a Guba-equivalent, StockTwits + yfinance is a workable English-language
  substitute; just relabel your annotation set accordingly.
- `bert-base-chinese` requires GPU for reasonable fine-tuning speed on 20k posts; CPU works
  but is slow.
- All rolling windows (250-day FF5, 6-month backtest training, 1/5/10/30-day price lags)
  are configurable via constants at the top of each module.
