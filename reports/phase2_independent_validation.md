# Phase 2: Independent Validation Report of the Indian SPCM Experiment

## 1. Executive Summary & Final Verdict
This report provides an independent validation of the real Indian SPCM predictive pipeline, historical walk-forward experiment, and portfolio backtest. Every claim, dataset range, observation count, and metric reported in Phase 1 was audited against the source code, raw data files, and saved artifacts.

**Overall Verdict**: **VALIDATED & REPRODUCIBLE (WITH CRITICAL DOCUMENTED BOUNDARIES)**
- **Methodology & Leakage Isolation**: **PASS** (10-day boundary purge, point-in-time sentiment shifting, trailing factor regressions, train-only preprocessing).
- **Target Label Formulation**: **PASS** (strictly cross-sectional top 30% per trading date; missing forward returns explicitly preserved as NaN).
- **Metric Reproducibility**: **PASS** (100% of reported fold AUCs, aggregate AUCs, Prec@3, and backtest returns were independently recalculated and match to the exact decimal).
- **Key Caveat**: News data in the NifSent50 dataset effectively terminates on **2025-01-10** (with only 5 stock-days of news in early 2025). The remaining 2025 period operates on price/factor features with imputed median sentiment.

---

## 2. Detailed Resolution of Specific Questions

### Question 1: Dataset Date Consistency
* **Finding**: The raw NifSent50 news dataset and `posts_clean.csv` span **2008-08-24 to 2025-01-10**. The latest headline in the cleaned dataset is on `2025-01-10`.
* **Explanation of Discrepancy**: The statement in Phase 1 that "news extends to 2026-01-14" arose because `daily_sentiment_features.csv` was generated via an outer join with `prices.csv` (which extends to 2026-01-14).
* **2025 Handling**: In 2025, exactly 5 stock-days have headlines (all between Jan 2 and Jan 10). From 2025-01-11 through 2025-12-31, `headline_count = 0`, `news_available = 0`, and `Sent = NaN`. These missing values are filled via the training fold median by `SimpleImputer(strategy='median')`. Fold 8 (2025) effectively tests the model in a sparse/no-news regime.

### Question 2: Stock Universe Consistency & HDFCBANK News
* **Finding**: Exactly the same 10 stocks (`BHARTIARTL`, `HDFCBANK`, `HINDUNILVR`, `ICICIBANK`, `INFY`, `ITC`, `KOTAKBANK`, `RELIANCE`, `SBIN`, `TCS`) are used across prices, feature engineering, labels, model training, predictions, and backtesting.
* **HDFCBANK News**: In `posts_clean.csv`, `HDFCBANK` has 0 headlines. In `daily_sentiment_features.csv`, `HDFCBANK` has `headline_count = 0`, `news_available = 0`, and NaN for `Sent`, `PV`, `AI` across all dates. It is retained in the cross-sectional ranking with sentiment safely imputed by the training median.
* **Neutral vs. Missing Sentiment**: Neutral-only days (488 stock-days) have `news_available = 1` and `Sent = NaN` (not 0.0). Missing sentiment days have `news_available = 0` and `Sent = NaN`. Neither is confused with genuine neutral polarity (`Sent = 0.0`, which occurs only when `bullish == bearish > 0`).

### Question 3: Reconciliation of Sample Counts
* **Total Prices Rows**: 24,840 rows (10 stocks $\times$ 2,484 trading days, 2016-01-01 to 2026-01-14).
* **Burn-in Training Period (2016–2017)**: 494 trading days $\times 10 = 4,940$ rows.
* **Out-of-Sample Test Period (2018–2025)**: Exactly 1,980 trading days $\times 10 = 19,800$ rows.
  - 2018: 246 days (2,460 rows)
  - 2019: 244 days (2,440 rows)
  - 2020: 250 days (2,500 rows)
  - 2021: 248 days (2,480 rows)
  - 2022: 248 days (2,480 rows)
  - 2023: 246 days (2,460 rows)
  - 2024: 249 days (2,490 rows)
  - 2025: 249 days (2,490 rows)
  - Total: 1,980 trading days = 19,800 out-of-sample evaluated test rows.
* **January 2026**: 10 trading days $\times 10 = 100$ rows. These provide the forward 10-day returns for December 2025.
* **Sum**: $4,940 \text{ (burn-in)} + 19,800 \text{ (test)} + 100 \text{ (Jan 2026)} = 24,840$ rows. Complete reconciliation achieved.

### Question 4: Target Labels Formulation
* **Calculation**: `df.groupby("stock")["close"].shift(-10) / df["close"] - 1`. Shifting occurs strictly within each stock across 10 trading days.
* **Cross-Sectional Top 30%**: `df.groupby("date")` computes `cutoff = quantile(0.70)`. Exactly 3 of 10 stocks (30.0%) receive class 1 per trading date across all 2,474 historical dates.
* **Missing Returns**: For the final 10 trading days in Jan 2026, forward returns are NaN, and labels are preserved as NaN (verified across all 100 observations).

### Question 5: Temporal Boundary Leakage & Purging
* **Purge Verification**: In every fold, an exact **10-trading-day purge** is enforced:
  - Fold 1: Train ends 2017-12-14 ($t+10$ realization: 2017-12-29) $\to$ Test starts 2018-01-01 (1 trading day gap).
  - Fold 2: Train ends 2018-12-14 ($t+10$ realization: 2018-12-31) $\to$ Test starts 2019-01-01 (1 trading day gap).
  - Fold 3: Train ends 2019-12-16 ($t+10$ realization: 2019-12-31) $\to$ Test starts 2020-01-01 (1 trading day gap).
  - Fold 4: Train ends 2020-12-16 ($t+10$ realization: 2020-12-31) $\to$ Test starts 2021-01-01 (1 trading day gap).
  - Fold 5: Train ends 2021-12-17 ($t+10$ realization: 2021-12-31) $\to$ Test starts 2022-01-03 (1 trading day gap).
  - Fold 6: Train ends 2022-12-16 ($t+10$ realization: 2022-12-30) $\to$ Test starts 2023-01-02 (1 trading day gap).
  - Fold 7: Train ends 2023-12-14 ($t+10$ realization: 2023-12-29) $\to$ Test starts 2024-01-01 (1 trading day gap).
  - Fold 8: Train ends 2024-12-16 ($t+10$ realization: 2024-12-31) $\to$ Test starts 2025-01-01 (1 trading day gap).
* Every training label is realized strictly prior to the first test date. Zero forward returns cross the temporal split.
* `SimpleImputer` and `MinMaxScaler` are fitted strictly inside `train_spcm()` on the training slice and frozen for test transformation.

### Question 6: Sentiment Timing
* **Date Shifting**: `s_stock['join_date'] = s_stock['date'] + pd.Timedelta(days=1)`. News on day $k$ is mapped forward via `merge_asof(direction='forward')` to the first trading day $t \ge k + 1$.
* Weekend news maps to Monday. Friday news maps to Monday. On trading day $t$, all news is from calendar date $t-1$ or earlier, available before market close ($15{:}30$ IST). No intraday or same-day post-close leakage exists.

### Question 7: Backtest Correctness
* **Holding Periods**: Rebalancing occurs on `rebal_dates = oos_dates[::10]`, creating 198 non-overlapping 10-trading-day intervals.
* **Weights**: Equal-weighted Top-3 stocks (1/3 each).
* **Turnover & Costs**: 15 bps per transaction. For turnover $T$, cost $= 2 \times T \times 15\text{ bps} = 30\text{ bps} \times T$. On initial buy, 15 bps (one-way). No double-counting.
* **Benchmark**: Equal-weighted universe across all 10 stocks rebalanced on the exact same 10-day cycle.

### Question 8: Metric Recalculation & Reproducibility
All metrics were independently recalculated from `reports/india_oos_predictions.csv` and matched `reports/india_model_comparison.csv` and `reports/india_walkforward_fold_results.csv`:
- **KNN**: Recalculated Mean AUC = 0.4996, Pooled AUC = 0.4991, Net Return = +104.40%, Sharpe = 0.52.
- **RF**: Recalculated Mean AUC = 0.4952, Pooled AUC = 0.4951, Net Return = +106.14%, Sharpe = 0.51.
- **NN**: Recalculated Mean AUC = 0.5137, Pooled AUC = 0.5116, Net Return = +121.38%, Sharpe = 0.50.
- **Benchmark Return**: +138.86%.

---

## 3. Reconciled Metrics Table

| Model | Recalc Mean AUC | Reported Mean AUC | Recalc Pooled AUC | Reported Pooled AUC | Recalc Prec@3 | Reported Prec@3 | Recalc Cum Net | Reported Cum Net | Recalc Sharpe | Reported Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 0.4996 | 0.4996 | 0.4991 | 0.4991 | 0.2980 | 0.2980 | +104.40% | +104.40% | 0.52 | 0.52 |
| **RF** | 0.4952 | 0.4952 | 0.4951 | 0.4951 | 0.2932 | 0.2932 | +106.14% | +106.14% | 0.51 | 0.51 |
| **NN** | **0.5137** | **0.5137** | **0.5116** | **0.5116** | **0.3138** | **0.3138** | **+121.38%** | **+121.38%** | 0.50 | 0.50 |
| **Benchmark** | — | — | — | — | — | — | +138.86% | +138.86% | 0.65 | 0.65 |

---

## 4. Final Recommendation
* **Readiness**: The codebase and historical experimental baseline are **METHODOLOGICALLY SOUND, REPRODUCIBLE, AND LEAKAGE-FREE**.
* **Model Improvement Recommendation**: The project is **READY TO PROCEED TO MODEL IMPROVEMENT** (e.g., expanding the stock universe beyond 10 stocks, testing longer rebalancing horizons such as 20–60 days to reduce turnover friction, integrating alternative news feeds to address 2025 news sparsity, and exploring regularization/feature selection).
* **Constraints**: Do NOT proceed to live data integration or live capital deployment until higher cross-sectional rank separation ($\text{AUC} > 0.55$) is demonstrated in expanded universes.
