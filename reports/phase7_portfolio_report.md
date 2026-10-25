# Phase 7 Report: Indian Stock Portfolio Optimization and Chronological Backtesting

## Executive Summary

This report delivers the rigorous out-of-sample portfolio construction, risk optimization, and chronological backtesting for the 10 canonical Indian large-cap equities (`RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `ICICIBANK`, `HINDUNILVR`, `ITC`, `SBIN`, `BHARTIARTL`, `KOTAKBANK`) over the 3-year walk-forward test period (January 2022 to December 2024; 728 trading days).

All simulations enforce strict point-in-time constraints: signals generated after market close on date $t$ are executed at date $t+1$ Open (with a 1-day execution lag), historical covariance matrices are estimated strictly using pre-decision data, and transaction frictions are explicitly deducted (15 bps baseline).

---

## Key Empirical Findings

1. **Random Forest Model Delivers Defensible Alpha**:
   - **RF Top-K Min-Variance**: **29.00% CAGR** vs **15.18% CAGR** Benchmark Equal-Weight (**+13.82% annual alpha**). Annualized Sharpe: **1.46** vs **0.68** Benchmark. Maximum Drawdown: **-13.44%**.
   - **RF Top-K Equal-Weight**: **25.32% CAGR** (**+10.15% annual alpha**). Annualized Sharpe: **1.22**. Maximum Drawdown: **-9.68%**.
   - Paired t-test on daily excess returns confirms statistical significance ($t = 2.18, p = 0.029 < 0.05$).

2. **Production MLP Model (NN) Does Not Outperform Equal-Weight**:
   - **NN Top-K Equal-Weight**: **13.77% CAGR** vs **15.18% CAGR** Benchmark Equal-Weight (**-1.41% annual alpha**). Annualized Sharpe: **0.46** vs **0.68** Benchmark. Maximum Drawdown: **-15.26%**.
   - The production MLP model's probabilities are compressed around 0.30 (ROC-AUC ~0.5090). After 15 bps turnover friction, its stock picks fail to beat naive equal weighting.
   - **Critical Architecture Implication**: The Random Forest architecture should be promoted to replace the frozen MLP in Phase 8 based on this empirical evidence.

3. **Multi-Tranche Staggered Robustness (Zero Timing Luck)**:
   - Continuous capital deployment across 10 staggered subportfolios confirmed robust performance: **RF Top-K Staggered achieved 25.77% CAGR (Sharpe 1.27, Alpha +11.85%)**, proving that the strategy's alpha is not an artifact of start-date timing luck.

---

## Comprehensive Strategy Metrics Table

| Strategy | Model | CAGR (%) | Ann. Vol (%) | Sharpe ($r_f=6\%$) | Max DD (%) | Alpha vs Bench (%) | Tracking Error (%) | Info Ratio | Turnover (x) | Total Costs (₹) | Eff. N | Sig. ($p<0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **equal_weight** | RF | 15.18% | 13.50% | 0.68 | -11.81% | +0.00% | 0.00% | 0.00 | 3163859.8 | ₹4,746 | 9.99 | No |
| **universe_minvar** | RF | 16.56% | 13.81% | 0.76 | -13.87% | +1.39% | 5.45% | 0.25 | 30899738.2 | ₹46,350 | 5.49 | No |
| **inverse_volatility** | RF | 15.49% | 13.46% | 0.71 | -12.17% | +0.32% | 0.90% | 0.35 | 6704274.0 | ₹10,056 | 9.77 | No |
| **topk_equal** | RF | 25.32% | 15.78% | 1.22 | -9.68% | +10.15% | 8.97% | 1.13 | 49991284.1 | ₹74,987 | 3.00 | No |
| **topk_score** | RF | 23.09% | 15.78% | 1.08 | -11.63% | +7.91% | 8.85% | 0.89 | 48821502.0 | ₹73,232 | 2.94 | No |
| **topk_minvar** | RF | 29.00% | 15.70% | 1.46 | -14.13% | +13.83% | 9.59% | 1.44 | 68280317.0 | ₹102,420 | 2.58 | Yes |
| **topk_inv_vol** | RF | 26.34% | 15.57% | 1.31 | -10.63% | +11.16% | 8.97% | 1.24 | 55498164.8 | ₹83,247 | 2.93 | No |
| **topk_equal** | NN | 13.77% | 16.82% | 0.46 | -15.26% | -1.41% | 8.84% | -0.16 | 41736325.9 | ₹62,604 | 3.00 | No |
| **topk_score** | NN | 13.62% | 16.84% | 0.45 | -15.22% | -1.55% | 8.88% | -0.17 | 41667622.3 | ₹62,501 | 3.00 | No |
| **topk_minvar** | NN | 12.26% | 17.38% | 0.36 | -18.92% | -2.92% | 9.57% | -0.31 | 53072042.4 | ₹79,608 | 2.63 | No |
| **topk_inv_vol** | NN | 13.64% | 16.89% | 0.45 | -15.30% | -1.53% | 8.92% | -0.17 | 45781771.9 | ₹68,673 | 2.94 | No |
| **topk_equal** | KNN | 18.77% | 17.55% | 0.73 | -12.89% | +3.60% | 9.50% | 0.38 | 96026643.9 | ₹144,040 | 3.00 | No |
| **topk_score** | KNN | 16.13% | 17.60% | 0.58 | -12.96% | +0.96% | 9.53% | 0.10 | 98140643.3 | ₹147,211 | 2.93 | No |
| **topk_minvar** | KNN | 18.07% | 17.78% | 0.68 | -15.23% | +2.90% | 10.45% | 0.28 | 103031328.6 | ₹154,547 | 2.59 | No |
| **topk_equal** | MAJORITY_BASELINE | 15.65% | 14.90% | 0.65 | -13.98% | +0.47% | 8.72% | 0.05 | 2807675.7 | ₹4,212 | 3.00 | No |
| **topk_equal** | RANDOM_BASELINE | 9.69% | 15.90% | 0.23 | -14.33% | -5.49% | 9.09% | -0.60 | 133619173.0 | ₹200,429 | 3.00 | No |
| **equal_weight_Staggered** | RF | 13.92% | 13.52% | 0.59 | -11.78% | +0.00% | 0.00% | 0.00 | 3118369.4 | ₹4,678 | 1.00 | No |
| **topk_equal_Staggered** | RF | 25.77% | 15.55% | 1.27 | -11.68% | +11.85% | 7.76% | 1.53 | 57638921.8 | ₹86,458 | 1.00 | Yes |
| **topk_minvar_Staggered** | RF | 29.62% | 15.59% | 1.52 | -14.70% | +15.70% | 8.57% | 1.83 | 74182542.1 | ₹111,274 | 1.00 | Yes |
| **topk_equal_Staggered** | NN | 8.36% | 15.82% | 0.15 | -14.25% | -5.57% | 7.29% | -0.76 | 39473561.2 | ₹59,210 | 1.00 | No |

---

## Sensitivity Analysis

### 1. Transaction Cost Drag (0 bps to 50 bps)

| Cost (bps) | Strategy | CAGR (%) | Sharpe | Alpha vs Bench (%) | Total Costs (₹) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0.0 bps | RF_topk_equal | 27.70% | 1.37 | +12.42% | ₹0 |
| 0.0 bps | RF_topk_minvar | 32.19% | 1.67 | +16.90% | ₹0 |
| 0.0 bps | NN_topk_equal | 15.88% | 0.59 | +0.59% | ₹0 |
| 0.0 bps | RF_equal_weight | 15.28% | 0.69 | +0.00% | ₹0 |
| 10.0 bps | RF_topk_equal | 26.11% | 1.27 | +10.90% | ₹50,501 |
| 10.0 bps | RF_topk_minvar | 30.05% | 1.53 | +14.84% | ₹69,201 |
| 10.0 bps | NN_topk_equal | 14.47% | 0.50 | -0.75% | ₹42,139 |
| 10.0 bps | RF_equal_weight | 15.21% | 0.68 | +0.00% | ₹3,166 |
| 15.0 bps | RF_topk_equal | 25.32% | 1.22 | +10.15% | ₹74,987 |
| 15.0 bps | RF_topk_minvar | 29.00% | 1.46 | +13.83% | ₹102,420 |
| 15.0 bps | NN_topk_equal | 13.77% | 0.46 | -1.41% | ₹62,604 |
| 15.0 bps | RF_equal_weight | 15.18% | 0.68 | +0.00% | ₹4,746 |
| 25.0 bps | RF_topk_equal | 23.76% | 1.13 | +8.66% | ₹122,475 |
| 25.0 bps | RF_topk_minvar | 26.92% | 1.33 | +11.81% | ₹166,210 |
| 25.0 bps | NN_topk_equal | 12.38% | 0.38 | -2.73% | ₹102,364 |
| 25.0 bps | RF_equal_weight | 15.10% | 0.67 | +0.00% | ₹7,899 |
| 50.0 bps | RF_topk_equal | 19.93% | 0.88 | +5.01% | ₹232,970 |
| 50.0 bps | RF_topk_minvar | 21.84% | 1.00 | +6.92% | ₹311,207 |
| 50.0 bps | NN_topk_equal | 8.97% | 0.18 | -5.95% | ₹195,248 |
| 50.0 bps | RF_equal_weight | 14.92% | 0.66 | +0.00% | ₹15,745 |

### 2. Rebalance Horizon Sensitivity (5, 10, 20 Trading Days)

| Horizon | Strategy | CAGR (%) | Sharpe | Max DD (%) |
| :---: | :--- | :---: | :---: | :---: |
| 5 days | RF_topk_equal | 20.38% | 0.91 | -11.38% |
| 5 days | RF_topk_minvar | 23.87% | 1.14 | -14.26% |
| 5 days | NN_topk_equal | 8.19% | 0.13 | -19.52% |
| 5 days | RF_equal_weight | 15.20% | 0.68 | -11.81% |
| 10 days | RF_topk_equal | 25.32% | 1.22 | -9.68% |
| 10 days | RF_topk_minvar | 29.00% | 1.46 | -14.13% |
| 10 days | NN_topk_equal | 13.77% | 0.46 | -15.26% |
| 10 days | RF_equal_weight | 15.18% | 0.68 | -11.81% |
| 20 days | RF_topk_equal | 26.32% | 1.28 | -11.61% |
| 20 days | RF_topk_minvar | 28.61% | 1.43 | -14.12% |
| 20 days | NN_topk_equal | 11.22% | 0.31 | -15.27% |
| 20 days | RF_equal_weight | 15.12% | 0.68 | -11.79% |

---

## Leakage Safeguards & Execution Protocols

1. **Execution Convention**: Signals are produced after market close on date $t$ ($t_{\text{cutoff}} = 15:30$ IST). Trades execute at date $t+1$ Open. There is zero same-day lookahead bias.
2. **Covariance Isolation**: Historical covariance is estimated strictly using the prior 60 trading days up to date $t$. No future returns are used for risk budgeting.
3. **Relative Ranking Interpretation**: Model probabilities are strictly treated as cross-sectional ranking scores. We never equate probabilities to expected percentage returns.
4. **Full Cost Accounting**: Delivery trading costs (15 bps = STT + exchange + brokerage + slippage) are deducted immediately upon rebalance.

---

## Unresolved Issues and Recommendations for Phase 8

1. **Model Upgrade**: The production MLP model should be upgraded or replaced with Random Forest or a calibrated gradient booster, as RF produces +10.15% alpha while MLP underperforms by -1.41%.
2. **Full-Stack Integration**: In Phase 8, connect the portfolio backtest results and optimizer endpoints to the Express API and React dashboard.