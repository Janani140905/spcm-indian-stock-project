# SPCM Phase 6: Rigorous Chronological Validation & Live Prediction Outcome Tracking Report

**Project:** Indian Stock Price Classification Model (SPCM)  
**Evaluation Scope:** National Stock Exchange of India (NSE) 10-Stock Large-Cap Universe  
**Target Stocks:** `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `ICICIBANK`, `HINDUNILVR`, `ITC`, `SBIN`, `BHARTIARTL`, `KOTAKBANK`  
**Report Date:** 2026-10-22  

---

## 1. Executive Summary & Audit of Trained Artifacts

Prior to out-of-sample evaluation, an audit of the active model artifact, preprocessing pipeline, and training data was conducted:

1. **Active Production Bundle:**
   - Filepath: `models/spcm_model_bundle.joblib` (133.5 KB)
   - Classifier: `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)` (`spcm_nn_price_and_sentiment`)
   - Preprocessors: `SimpleImputer(strategy="median")` and `MinMaxScaler(feature_range=(-1, 1))`
   - Feature Dimension: Exactly 45 features (40 Price Lag Features + 5 Point-in-Time Sentiment Indicators)
   - In-Sample Training AUC: `0.5183` | Training Observations: `9,760`
2. **Label Definition Verification:**
   - Formula: $R_{i, t+10} = \frac{Close_{i, t+10} - Close_{i, t}}{Close_{i, t}}$
   - Binary Target: $y_{i, t} = 1$ if $R_{i, t+10} \ge Q_{70}(R_{\cdot, t+10})$ across the universe on date $t$, else $0$.
   - Positive Class Prevalence: Strictly fixed at **30.0%** cross-sectionally per trading session.
3. **Data Integrity Audit:**
   - Historical Dataset: `data/processed/india/prices.csv` containing 9,860 clean trading rows across all 10 target stocks from `2021-01-01` to `2024-12-30` (986 unique trading dates).
   - Validation Guarantee: No synthetic data, no random train/test splits, and zero cross-validation lookahead.

---

## 2. Chronological Walk-Forward Methodology & Temporal Purging

Because the SPCM target label is a forward 10-trading-day return, observations during the final 10 trading sessions of year $Y$ evaluate against price movements occurring in January of year $Y+1$ (the out-of-sample test window).

### Temporal Purge (Embargo) Formulation
To prevent this forward-looking overlap leakage, the training window is mathematically purged before every test fold:
$$t_{\text{train, purged}} = t_{\text{test, start}} - 10 \text{ trading days} - 1$$
$$\text{Realization}(y_{t_{\text{train, purged}}}) = t_{\text{train, purged}} + 10 \text{ trading days} < t_{\text{test, start}}$$

### Out-of-Sample Folds
| Fold | Training Window (Purged) | Test Window (Out-of-Sample) | Purged Gap | Train Obs | Test Obs | Positive Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2021-01-01 to 2021-12-17 | 2022-01-03 to 2022-12-30 | 10 days | 2,380 | 2,480 | 30.0% (744) |
| **Fold 2** | 2021-01-01 to 2022-12-16 | 2023-01-02 to 2023-12-29 | 10 days | 4,860 | 2,450 | 30.0% (735) |
| **Fold 3** | 2021-01-01 to 2023-12-14 | 2024-01-01 to 2024-12-13 | 10 days | 7,310 | 2,350 | 30.0% (705) |

### Preprocessing Isolation Guarantee
- `SimpleImputer` and `MinMaxScaler` are instantiated and fitted (`fit_transform`) exclusively on $X_{\text{train}}$ inside each fold.
- Out-of-sample test data is transformed (`transform`) using the frozen training parameters.
- Test statistics never alter training scales.

---

## 3. Comprehensive Out-of-Sample Model Comparison

Metrics averaged across all 3 out-of-sample folds (7,280 total out-of-sample evaluations):

| Model | Feature Configuration | Mean ROC-AUC | Std AUC | Mean PR-AUC | Balanced Acc | Raw Acc (0.5) | Precision | Recall | F1 Score | Precision@3 | Top-3 Fwd Return | Benchmark Return | Alpha (vs Bench) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (RF)** | **Price + Sentiment** | **0.5836** | 0.0580 | **0.3643** | **0.5036** | 0.6901 | 0.3312 | 0.0373 | 0.0592 | **0.3913** | **+1.05%** | +0.54% | **+0.51%** |
| **Random Forest (RF)** | Price-Only | 0.5771 | 0.0586 | 0.3606 | 0.5046 | 0.6901 | 0.3354 | 0.0409 | 0.0638 | 0.3923 | +1.02% | +0.54% | **+0.48%** |
| **KNN (k=15)** | Price + Sentiment | 0.5164 | 0.0031 | 0.3168 | 0.5055 | 0.6697 | 0.3343 | 0.0950 | 0.1431 | 0.3112 | +0.59% | +0.54% | +0.05% |
| **KNN (k=15)** | Price-Only | 0.5134 | 0.0098 | 0.3170 | 0.5076 | 0.6715 | 0.3428 | 0.0979 | 0.1452 | 0.3079 | +0.59% | +0.54% | +0.05% |
| **Neural Net (NN / MLP)** | Price + Sentiment | 0.5090 | 0.0095 | 0.3106 | 0.5000 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.3032 | +0.43% | +0.54% | -0.11% |
| **Neural Net (NN / MLP)** | Price-Only | 0.5079 | 0.0179 | 0.3107 | 0.5000 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.3076 | +0.49% | +0.54% | -0.05% |
| **Majority Baseline** | Constant 0 | 0.5000 | 0.0000 | 0.3000 | 0.5000 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.3087 | +0.57% | +0.54% | +0.03% |
| **Random Baseline** | Uniform Prior | 0.5036 | 0.0136 | 0.3039 | 0.5053 | 0.5058 | 0.3046 | 0.5039 | 0.3796 | 0.3115 | +0.63% | +0.54% | +0.09% |

### Key Analytical Takeaways:
1. **The 70% Accuracy Illusion:**
   - Both the Neural Network and the Majority-Class baseline achieve exactly **70.0% accuracy** at threshold 0.5 because the positive-class prevalence is 30% and the model predicts 0 for almost all instances.
   - Fixed-threshold accuracy is uninformative for cross-sectional financial ranking.
2. **Discrimination & Ranking Superiority:**
   - Random Forest achieves the highest mean ROC-AUC (**0.5836**) and PR-AUC (**0.3643**), beating both the Random Baseline (0.5036) and Majority Baseline (0.5000).
   - In cross-sectional ranking, RF achieves a **Precision@3 of 39.13%** (vs random 30.0%), generating an average 10-day forward return of **+1.05%** per period compared to the **+0.54%** benchmark return.
3. **Sentiment Contribution:**
   - Adding sentiment features (`Sent`, `PV`, `AI`, `news_available`, `headline_count`) moderately improves ROC-AUC across models (+0.0065 for RF, +0.0030 for KNN, +0.0011 for NN) without destabilizing volatility.

---

## 4. Fold-Level Breakdown (Confusion Matrix & Classification Behavior)

### Fold 1: Test Year 2022 (Bearish Market Cycle)
- Test Observations: 2,480 | Benchmark Return: **+0.37%**
- **RF (Price + Sentiment):** ROC-AUC: `0.5611` | PR-AUC: `0.3394` | P@3: `37.9%` | Top-3 Ret: **+0.55%**  
  - Confusion Matrix: $TN=1596, FP=140, FN=668, TP=76$
- **KNN (Price + Sentiment):** ROC-AUC: `0.5181` | PR-AUC: `0.3119` | P@3: `31.0%` | Top-3 Ret: **+0.26%**  
  - Confusion Matrix: $TN=1495, FP=241, FN=636, TP=108$
- **NN (Price + Sentiment):** ROC-AUC: `0.5018` | PR-AUC: `0.3213` | P@3: `30.8%` | Top-3 Ret: **+0.08%**  
  - Confusion Matrix: $TN=1736, FP=0, FN=744, TP=0$

### Fold 2: Test Year 2023 (Strong Momentum Cycle)
- Test Observations: 2,450 | Benchmark Return: **+0.59%**
- **RF (Price + Sentiment):** ROC-AUC: `0.6495` | PR-AUC: `0.4347` | P@3: `45.3%` | Top-3 Ret: **+1.39%** (Alpha: +0.80%)  
  - Confusion Matrix: $TN=1710, FP=5, FN=732, TP=3$
- **KNN (Price + Sentiment):** ROC-AUC: `0.5183` | PR-AUC: `0.3212` | P@3: `32.0%` | Top-3 Ret: **+0.57%**  
  - Confusion Matrix: $TN=1600, FP=115, FN=675, TP=60$
- **NN (Price + Sentiment):** ROC-AUC: `0.5055` | PR-AUC: `0.2954` | P@3: `29.1%` | Top-3 Ret: **+0.38%**  
  - Confusion Matrix: $TN=1715, FP=0, FN=735, TP=0$

### Fold 3: Test Year 2024 (Late-Stage Expansion)
- Test Observations: 2,350 | Benchmark Return: **+0.66%**
- **RF (Price + Sentiment):** ROC-AUC: `0.5403` | PR-AUC: `0.3189` | P@3: `34.2%` | Top-3 Ret: **+1.20%** (Alpha: +0.54%)  
  - Confusion Matrix: $TN=1634, FP=11, FN=701, TP=4$
- **KNN (Price + Sentiment):** ROC-AUC: `0.5128` | PR-AUC: `0.3173` | P@3: `30.3%` | Top-3 Ret: **+0.93%**  
  - Confusion Matrix: $TN=1569, FP=76, FN=664, TP=41$
- **NN (Price + Sentiment):** ROC-AUC: `0.5198` | PR-AUC: `0.3152` | P@3: `31.1%` | Top-3 Ret: **+0.83%**  
  - Confusion Matrix: $TN=1645, FP=0, FN=705, TP=0$

---

## 5. Indian Market Data & News Sentiment Limitations Audit

1. **Market Data Timing & Delays:**
   - Free Yahoo Finance quotes for NSE equities carry an exchange-mandated **15-minute delay** during trading hours (09:15 to 15:30 IST).
   - End-of-day daily closes are official historical prints; intraday polling must not be represented as an un-delayed direct market access (DMA) feed.
2. **News Availability & Sparsity:**
   - News headline volume across Indian stocks is highly concentrated in index bellwethers (`RELIANCE`, `TCS`, `INFY`, `HDFCBANK`).
   - Stocks like `SBIN`, `ITC`, and `KOTAKBANK` frequently have zero news on given trading sessions.
   - The pipeline handles this through `news_available=0`, leaving `Sent`, `PV`, and `AI` as `NaN`, which are transformed via the frozen training median (`0.0`).
3. **Sentiment Model Differences:**
   - Training datasets utilized FinBERT probability representations.
   - Inference-time news fallbacks utilize Loughran-McDonald / financial-lexicon scoring when local FinBERT weights are not loaded. Both yield values bounded in $[-1, 1]$.

---

## 6. Live Prediction Outcome Tracking Engine

Implemented in [`src/evaluation/outcome_tracker.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/src/evaluation/outcome_tracker.py) with storage in [`reports/prediction_outcomes.csv`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/reports/prediction_outcomes.csv) and [`reports/prediction_outcomes.json`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/reports/prediction_outcomes.json).

### Operational Guarantees:
1. **Pending Status Lifecycle:**
   - When inference runs, predictions are recorded as `status = "PENDING"`.
   - `realized_price_t10`, `realized_return_10d`, `realized_label`, and `prediction_correct` remain `None`.
   - `days_elapsed` tracks forward trading sessions closed since prediction.
2. **Zero Premature Resolution:**
   - If fewer than 10 trading sessions have elapsed, the outcome remains `PENDING`. No interim returns are treated as final labels.
3. **Matured Label Calculation:**
   - Once exactly 10 forward trading days have closed ($t+10$), the 10-day forward return is calculated:
     $$R_{t+10} = \frac{Close_{t+10} - Close_t}{Close_t}$$
   - The cross-sectional 70th percentile cutoff of the eligible 10-stock universe on date $t$ is calculated.
   - `realized_label = 1` if $R_{t+10} \ge Cutoff$, else `0`.
   - `prediction_correct = (predicted_class == realized_label)`.
   - Status transitions to `COMPLETED`.
4. **Sparse Universe Handling:**
   - If fewer than 5 universe stocks have valid quotes on that date, the record transitions to `INSUFFICIENT_DATA` rather than assigning an arbitrary label.

---

## 7. Windows PowerShell Reproduction Commands

### Run Full Chronological Walk-Forward Validation
```powershell
cd "E:\Mini Project - Stock Prediction\Model Training"
py -3.11 -m src.evaluation.chronological_validation --prices data/processed/india/prices.csv --out-dir reports
```

### Check Prediction Outcome Tracker Status
```powershell
py -3.11 -m src.evaluation.outcome_tracker --status
```

### Resolve Matured Prediction Outcomes Against Historical / Current Data
```powershell
py -3.11 -m src.evaluation.outcome_tracker --update --prices data/processed/india/prices.csv
```

### Run Live Inference with Automatic Outcome Tracking Registration
```powershell
py -3.11 -m src.inference.spcm_inference_pipeline --symbols RELIANCE TCS INFY SBIN HDFCBANK --top-k 3 --track-outcomes
```

### Run Full Python Test Suite (69 Tests)
```powershell
py -3.11 -m unittest discover tests
```

---

## 8. Test Suite Verification Summary

All 69 unit and integration tests run and pass in **6.336 seconds**:
- **Phase 1 (10 tests):** Market data validation, bounds, duplicate timestamp handling ([`test_indian_market_data.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_indian_market_data.py))
- **Phase 2 (11 tests):** 40 price lag features, zero future leakage ([`test_price_features_inference.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_price_features_inference.py))
- **Phase 2 & Baselines (10 tests):** Chronological integrity, walk-forward isolation ([`test_phase3.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_phase3.py))
- **Phase 3 (20 tests):** News fetching, RSS parsing, deduplication, sentiment scoring ([`test_indian_news_pipeline.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_indian_news_pipeline.py))
- **Phase 4 (7 tests):** Inference bundle persistence, zero-refit guarantee ([`test_spcm_inference_pipeline.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_spcm_inference_pipeline.py))
- **Phase 5 (4 tests):** FastAPI adapter endpoints and schema validation ([`test_fastapi_adapter.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_fastapi_adapter.py))
- **Phase 6 (7 tests):** Walk-forward purge gap, preprocessing isolation, metrics, pending outcome lifecycle, 10-day resolution ([`test_chronological_validation_and_tracking.py`](file:///e:/Mini%20Project%20-%20Stock%20Prediction/Model%20Training/tests/test_chronological_validation_and_tracking.py))
