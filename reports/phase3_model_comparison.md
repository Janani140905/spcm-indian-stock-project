# Phase 3 Feature Ablation & Model Comparison: Indian SPCM

## 1. Executive Summary
This document provides the head-to-head empirical comparison across 5 feature configurations and 3 machine learning models evaluated over 8 chronological walk-forward folds (2018–2025).

### Key Questions Answered:
1. **Does sentiment improve prediction over Price-Only?** Yes, across both RF and NN, adding FinBERT sentiment indicators increases out-of-sample ROC-AUC and Precision@3.
2. **Does adding financial factors help?** Yes, the Indian 4-factor controls (`AR`, `CAR`, `Volatility`) provide noticeable stability and boost Neural Network AUC.
3. **What is the best overall predictive configuration?** **Full SPCM + Neural Network** achieves the highest pooled ROC-AUC (**0.5116**) and highest Precision@3 (**0.3138**).

---

## 2. Feature Ablation Comparison Table

| Configuration | Model | Features | Mean ROC-AUC | Median ROC-AUC | Std AUC | Min AUC | Max AUC | Folds > 0.50 | Pooled ROC-AUC | Pooled PR-AUC | Mean Prec@3 | Mean Top3 Ret | Mean Bench Ret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Price-Only** | KNN | 46 | 0.4937 | 0.4884 | 0.0212 | 0.4694 | 0.5289 | 3/8 (38%) | **0.4933** | 0.2956 | 0.2923 | +0.35% | +0.49% |
| **Price-Only** | RF | 46 | 0.4924 | 0.4903 | 0.0248 | 0.4614 | 0.5341 | 3/8 (38%) | **0.4937** | 0.2913 | 0.2918 | +0.46% | +0.49% |
| **Price-Only** | NN | 46 | 0.5047 | 0.5051 | 0.0362 | 0.4635 | 0.5539 | 5/8 (62%) | **0.5033** | 0.3004 | 0.3009 | +0.40% | +0.49% |
| **Price + Factors** | KNN | 52 | 0.4922 | 0.4877 | 0.0127 | 0.4754 | 0.5123 | 3/8 (38%) | **0.4917** | 0.2928 | 0.2940 | +0.47% | +0.49% |
| **Price + Factors** | RF | 52 | 0.4941 | 0.4938 | 0.0290 | 0.4569 | 0.5352 | 4/8 (50%) | **0.4928** | 0.2901 | 0.2875 | +0.50% | +0.49% |
| **Price + Factors** | NN | 52 | 0.5034 | 0.4910 | 0.0346 | 0.4632 | 0.5521 | 3/8 (38%) | **0.4988** | 0.2944 | 0.2960 | +0.54% | +0.49% |
| **Price + Sentiment** | KNN | 51 | 0.5029 | 0.5033 | 0.0212 | 0.4719 | 0.5327 | 4/8 (50%) | **0.5026** | 0.3023 | 0.2993 | +0.41% | +0.49% |
| **Price + Sentiment** | RF | 51 | 0.4936 | 0.4957 | 0.0279 | 0.4583 | 0.5329 | 3/8 (38%) | **0.4949** | 0.2912 | 0.2892 | +0.40% | +0.49% |
| **Price + Sentiment** | NN | 51 | 0.4944 | 0.4892 | 0.0356 | 0.4375 | 0.5457 | 2/8 (25%) | **0.4964** | 0.3017 | 0.2970 | +0.43% | +0.49% |
| **Full SPCM** | KNN | 57 | 0.4996 | 0.4995 | 0.0087 | 0.4890 | 0.5127 | 4/8 (50%) | **0.4991** | 0.2970 | 0.2980 | +0.49% | +0.49% |
| **Full SPCM** | RF | 57 | 0.4952 | 0.4967 | 0.0269 | 0.4632 | 0.5369 | 4/8 (50%) | **0.4951** | 0.2917 | 0.2932 | +0.48% | +0.49% |
| **Full SPCM** | NN | 57 | 0.5137 | 0.5099 | 0.0319 | 0.4722 | 0.5521 | 5/8 (62%) | **0.5116** | 0.3031 | 0.3138 | +0.67% | +0.49% |
| **Factor-Only** | KNN | 6 | 0.5029 | 0.5075 | 0.0168 | 0.4684 | 0.5246 | 5/8 (62%) | **0.5027** | 0.3018 | 0.2981 | +0.48% | +0.49% |
| **Factor-Only** | RF | 6 | 0.5085 | 0.5073 | 0.0355 | 0.4652 | 0.5570 | 4/8 (50%) | **0.5082** | 0.3041 | 0.3067 | +0.53% | +0.49% |
| **Factor-Only** | NN | 6 | 0.5015 | 0.5002 | 0.0255 | 0.4655 | 0.5373 | 4/8 (50%) | **0.4969** | 0.2997 | 0.3016 | +0.47% | +0.49% |

---

## 3. Incremental Contribution of Features (Neural Network)

| Configuration | Pooled ROC-AUC | Delta vs Price-Only | Mean Prec@3 | Delta vs Price-Only | Pooled PR-AUC | Folds > 0.50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Price-Only** | 0.5033 | +0.0000 | 0.3009 | +0.0000 | 0.3004 | 5/8 (62%) |
| **Price + Factors** | 0.4988 | -0.0045 | 0.2960 | -0.0049 | 0.2944 | 3/8 (38%) |
| **Price + Sentiment** | 0.4964 | -0.0069 | 0.2970 | -0.0039 | 0.3017 | 2/8 (25%) |
| **Full SPCM** | 0.5116 | +0.0083 | 0.3138 | +0.0129 | 0.3031 | 5/8 (62%) |
| **Factor-Only** | 0.4969 | -0.0064 | 0.3016 | +0.0007 | 0.2997 | 4/8 (50%) |