# Historical Walk-Forward Evaluation Report: Indian SPCM Pipeline

## Executive Summary
This report documents the rigorous historical evaluation of the Sentiment and Price Combined Model (SPCM) adapted to the Indian equity market across 10 years of real daily data (2016–2025).
Following an in-depth audit, an essential **10-trading-day temporal purge (embargo)** was introduced at every training boundary to eliminate label-overlap leakage across temporal boundaries.

### Key Highlights:
- **Best Model by Mean Out-of-Sample ROC-AUC**: **NN** (0.5137)
- **Best Model by Net Strategy Return**: **NN** (+121.38% net vs 138.86% benchmark)
- **Leakage Status**: Strictly leakage-free (10-day boundary purge, next-trading-day sentiment shift, trailing factor regressions, train-only preprocessing).

---

## 1. Out-of-Sample Model Comparison

| Model | Mean AUC | Median AUC | Std AUC | Min AUC | Max AUC | Folds > 0.50 | Folds > 0.55 | Pooled OOS AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 0.4996 | 0.4995 | 0.0087 | 0.4890 | 0.5127 | 4/8 (50%) | 0/8 (0%) | 0.4991 |
| **RF** | 0.4952 | 0.4968 | 0.0269 | 0.4632 | 0.5369 | 4/8 (50%) | 0/8 (0%) | 0.4951 |
| **NN** | 0.5137 | 0.5099 | 0.0319 | 0.4722 | 0.5521 | 5/8 (62%) | 1/8 (12%) | 0.5116 |

> [!IMPORTANT]
> **Why 70% Accuracy is Misleading**: By construction, the positive class represents exactly the top 30% of forward returns. A naive dummy model that always predicts class 0 achieves an accuracy of **70.00%** with zero skill ($F_1 = 0.0$). Models must be evaluated using continuous discrimination (ROC-AUC) and cross-sectional ranking quality (Precision@K), not naive accuracy.

---

## 2. Secondary Classification Metrics (Threshold = 0.5)

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean Prec@3 | Baseline Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.7000 |
| **KNN** | 0.6606 | 0.2924 | 0.0903 | 0.1352 | 0.2980 | 0.7000 |
| **RF** | 0.6870 | 0.1524 | 0.0209 | 0.0324 | 0.2932 | 0.7000 |
| **NN** | 0.6998 | 0.0625 | 0.0002 | 0.0003 | 0.3138 | 0.7000 |

---

## 3. Trading Strategy Backtest Evaluation

### Strategy Parameters:
- **Rebalance Frequency**: Every 10 trading days (non-overlapping, exactly matching the 10-day forward return target).
- **Universe & Sizing**: Top $K=3$ stocks selected by out-of-sample predicted probability (equal-weighted 33.3% per stock).
- **Transaction Costs**: 15 bps (0.15%) delivery brokerage/STT per trade (30 bps roundtrip).
- **Benchmark**: Equal-weighted buy-and-hold across all 10 stocks rebalanced on the exact same 10-day cycle.
- **Test Period**: 2018-01-01 to 2025-12-31 (8 full years, ~200 non-overlapping 10-day cycles).

| Model Strategy | Cumulative Return (Gross) | Cumulative Return (Net) | Annualized Net Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Benchmark Cum Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | +188.23% | +104.40% | +9.53% | 18.31% | 0.52 | -29.07% | +138.86% |
| **RF** | +152.01% | +106.14% | +9.64% | 18.95% | 0.51 | -29.97% | +138.86% |
| **NN** | +177.81% | +121.38% | +10.64% | 21.35% | 0.50 | -31.39% | +138.86% |

---

## 4. Fold-Level Breakdown

| Fold | Model | Train Window (Purged) | Test Window (OOS) | Test Rows | ROC-AUC | Accuracy | F1 | Prec@3 | Top-3 Fwd Ret | Bench Fwd Ret |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | KNN | 2016-01-01 to 2017-12-14 | 2018-01-01 to 2018-12-31 | 2460 | 0.5127 | 0.6768 | 0.1330 | 0.318 | 0.03% | 0.18% |
| 1 | RF | 2016-01-01 to 2017-12-14 | 2018-01-01 to 2018-12-31 | 2460 | 0.5099 | 0.6996 | 0.0134 | 0.305 | 0.34% | 0.18% |
| 1 | NN | 2016-01-01 to 2017-12-14 | 2018-01-01 to 2018-12-31 | 2460 | 0.5127 | 0.7000 | 0.0000 | 0.323 | 0.22% | 0.18% |
| 2 | KNN | 2016-01-01 to 2018-12-14 | 2019-01-01 to 2019-12-31 | 2440 | 0.5048 | 0.6762 | 0.1143 | 0.300 | 1.06% | 0.62% |
| 2 | RF | 2016-01-01 to 2018-12-14 | 2019-01-01 to 2019-12-31 | 2440 | 0.5075 | 0.6902 | 0.0156 | 0.314 | 0.90% | 0.62% |
| 2 | NN | 2016-01-01 to 2018-12-14 | 2019-01-01 to 2019-12-31 | 2440 | 0.5490 | 0.7000 | 0.0000 | 0.335 | 1.07% | 0.62% |
| 3 | KNN | 2016-01-01 to 2019-12-16 | 2020-01-01 to 2020-12-31 | 2500 | 0.4914 | 0.6260 | 0.2056 | 0.299 | 0.63% | 0.98% |
| 3 | RF | 2016-01-01 to 2019-12-16 | 2020-01-01 to 2020-12-31 | 2500 | 0.4744 | 0.6404 | 0.1543 | 0.269 | 0.34% | 0.98% |
| 3 | NN | 2016-01-01 to 2019-12-16 | 2020-01-01 to 2020-12-31 | 2500 | 0.4827 | 0.6996 | 0.0000 | 0.281 | 0.89% | 0.98% |
| 4 | KNN | 2016-01-01 to 2020-12-16 | 2021-01-01 to 2021-12-31 | 2480 | 0.4986 | 0.6734 | 0.1785 | 0.294 | 1.20% | 1.00% |
| 4 | RF | 2016-01-01 to 2020-12-16 | 2021-01-01 to 2021-12-31 | 2480 | 0.4860 | 0.6823 | 0.0051 | 0.265 | 0.67% | 1.00% |
| 4 | NN | 2016-01-01 to 2020-12-16 | 2021-01-01 to 2021-12-31 | 2480 | 0.5071 | 0.6984 | 0.0000 | 0.296 | 1.05% | 1.00% |
| 5 | KNN | 2016-01-01 to 2021-12-17 | 2022-01-03 to 2022-12-30 | 2480 | 0.5005 | 0.6472 | 0.1007 | 0.285 | 0.27% | 0.31% |
| 5 | RF | 2016-01-01 to 2021-12-17 | 2022-01-03 to 2022-12-30 | 2480 | 0.5369 | 0.6831 | 0.0709 | 0.352 | 0.74% | 0.31% |
| 5 | NN | 2016-01-01 to 2021-12-17 | 2022-01-03 to 2022-12-30 | 2480 | 0.5521 | 0.7000 | 0.0000 | 0.357 | 0.92% | 0.31% |
| 6 | KNN | 2016-01-01 to 2022-12-16 | 2023-01-02 to 2023-12-29 | 2460 | 0.4910 | 0.6760 | 0.0974 | 0.301 | 0.40% | 0.49% |
| 6 | RF | 2016-01-01 to 2022-12-16 | 2023-01-02 to 2023-12-29 | 2460 | 0.5186 | 0.7000 | 0.0000 | 0.318 | 0.46% | 0.49% |
| 6 | NN | 2016-01-01 to 2022-12-16 | 2023-01-02 to 2023-12-29 | 2460 | 0.5455 | 0.7000 | 0.0000 | 0.344 | 0.83% | 0.49% |
| 7 | KNN | 2016-01-01 to 2023-12-14 | 2024-01-01 to 2024-12-31 | 2490 | 0.5086 | 0.6518 | 0.1321 | 0.294 | 0.40% | 0.30% |
| 7 | RF | 2016-01-01 to 2023-12-14 | 2024-01-01 to 2024-12-31 | 2490 | 0.4654 | 0.7000 | 0.0000 | 0.245 | 0.23% | 0.30% |
| 7 | NN | 2016-01-01 to 2023-12-14 | 2024-01-01 to 2024-12-31 | 2490 | 0.4881 | 0.7000 | 0.0000 | 0.297 | 0.30% | 0.30% |
| 8 | KNN | 2016-01-01 to 2024-12-16 | 2025-01-01 to 2025-12-31 | 2490 | 0.4890 | 0.6570 | 0.1196 | 0.292 | -0.05% | 0.01% |
| 8 | RF | 2016-01-01 to 2024-12-16 | 2025-01-01 to 2025-12-31 | 2490 | 0.4632 | 0.7000 | 0.0000 | 0.277 | 0.16% | 0.01% |
| 8 | NN | 2016-01-01 to 2024-12-16 | 2025-01-01 to 2025-12-31 | 2490 | 0.4722 | 0.7000 | 0.0027 | 0.277 | 0.08% | 0.01% |

---

## 5. Methodological & Leakage Checklist
- [x] **No Future Dates in Features**: Observable at market close $15{:}30$ IST.
- [x] **Point-in-Time News Alignment**: News on calendar day $k$ lagged to trading day $t \ge k+1$.
- [x] **Trailing Factor Regression**: Rolling 60-day OLS uses only $[t-60, t-1]$.
- [x] **Train-Only Preprocessing**: `SimpleImputer` and `MinMaxScaler` fit solely on training fold.
- [x] **Temporal Purging**: 10-day boundary embargo removes overlapping forward return labels from training.
- [x] **Zero Temporal Shuffling**: Strictly chronological expanding window.
- [x] **Verified Units**: Factor returns in decimal; zero double-division.