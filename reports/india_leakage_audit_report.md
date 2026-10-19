# Research Methodology & Data Leakage Audit Report

## 1. Executive Summary
A rigorous line-by-line audit of the Indian SPCM model pipeline was conducted to inspect feature availability, factor regression timing, label formulation, and cross-validation boundaries.

## 2. Key Audit Findings

### A. Prediction Target & Label Formulation
- **Target Definition**: Top 30% cross-sectional forward 10-trading-day return ($close_{t+10} / close_t - 1$).
- **Ranking Scope**: Ranking occurs cross-sectionally within each trading date (`df.groupby('date')`), requiring at least 5 active stocks per date. It does NOT rank across the historical time series.
- **Missing Value Preservation**: A code bug was identified where `(g['fwd_ret'] >= cutoff).astype(float)` converted `NaN` to `0.0`. This has been corrected in `src/models/spcm.py` to strictly preserve `NaN` for missing/unbalanced observations.

### B. Point-in-Time Alignment & Sentiment Shift
- **Timing Rule**: Financial news published on calendar date $k$ is shifted forward to the next available trading day $t \ge k + 1$ via `shift_sentiment_dates()`.
- **Intraday Safety**: Weekend and holiday news maps forward to Monday. Friday news maps forward to Monday. No same-day post-close announcements leak into date $t$ features.

### C. Rolling Factor Regressions & Abnormal Returns
- **OLS Window**: Rolling 60-day OLS is estimated strictly over trailing observations $[t-60, t-1]$.
- **Out-of-Sample AR**: Day $t$ factor returns evaluate against $\hat{\beta}_{t-1}$. Trailing CARs cover $[t-5, t-1]$ and $[t-20, t-6]$. Zero future data enters AR or CAR features.

### D. Temporal Purging at Cross-Validation Boundaries (CRITICAL FIX)
- **The Overlap Leakage Discovery**: Because the target label is a forward 10-day return, training observations during the final 10 trading days of year $Y$ derive their labels from price movements occurring in January of year $Y+1$ (the out-of-sample test window).
- **The Fix**: An exact **10-trading-day temporal purge (embargo)** is enforced before each test start date. Training observations whose forward 10-day realization crosses into the test window are excluded from the training mask.
- **Mathematical Guarantee**: For every fold, the last training observation $t_{train}$ satisfies $t_{train} + 10\text{ days} < t_{test, start}$. Zero test-period returns enter model training.

| Fold | Test Start | Test End | Raw Train End | Purged Train End | Realization of Last Train Label | Purged Days (Stocks) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2018-01-01 | 2018-12-31 | 2017-12-31 | 2017-12-14 | 2017-12-29 | 11 days (110 obs) |
| 2 | 2019-01-01 | 2019-12-31 | 2018-12-31 | 2018-12-14 | 2018-12-31 | 11 days (110 obs) |
| 3 | 2020-01-01 | 2020-12-31 | 2019-12-31 | 2019-12-16 | 2019-12-31 | 11 days (110 obs) |
| 4 | 2021-01-01 | 2021-12-31 | 2020-12-31 | 2020-12-16 | 2020-12-31 | 11 days (110 obs) |
| 5 | 2022-01-01 | 2022-12-31 | 2021-12-31 | 2021-12-17 | 2021-12-31 | 11 days (110 obs) |
| 6 | 2023-01-01 | 2023-12-31 | 2022-12-31 | 2022-12-16 | 2022-12-30 | 11 days (110 obs) |
| 7 | 2024-01-01 | 2024-12-31 | 2023-12-31 | 2023-12-14 | 2023-12-29 | 11 days (110 obs) |
| 8 | 2025-01-01 | 2025-12-31 | 2024-12-31 | 2024-12-16 | 2024-12-31 | 11 days (110 obs) |

### E. Preprocessing Isolation
- `SimpleImputer` and `MinMaxScaler` are instantiated and fitted (`fit_transform`) exclusively inside `train_spcm` on the training fold slice.
- Out-of-sample test data is transformed (`transform`) using the frozen training imputer and scaler. No global fit occurs.

## 3. Verdict
**PASS**: The pipeline enforces point-in-time sentiment shifting, strictly historical rolling factor regressions, train-only preprocessing, and a 10-trading-day boundary purge.