"""
Scratch script to test covariance estimation, optimization, and fallback mechanisms.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def test_optimizer():
    prices = pd.read_csv("data/processed/india/prices.csv")
    prices['date'] = pd.to_datetime(prices['date'])
    piv = prices.pivot(index='date', columns='stock', values='close')
    daily_returns = piv.pct_change().dropna()
    
    # Take a 60-day window
    window = daily_returns.iloc[:60]
    cov = window.cov().values * 252 # annualized cov
    stocks = list(window.columns)
    N = len(stocks)
    
    print(f"Testing on {N} stocks with 60-day window. Covariance condition number: {np.linalg.cond(cov):.2f}")
    
    # 1. Equal weight
    w_eq = np.ones(N) / N
    vol_eq = np.sqrt(w_eq @ cov @ w_eq)
    print(f"Equal Weight annualized volatility: {vol_eq:.4f}")
    
    # 2. Inverse Volatility
    stds = np.sqrt(np.diag(cov))
    inv_vols = 1.0 / stds
    w_inv = inv_vols / np.sum(inv_vols)
    vol_inv = np.sqrt(w_inv @ cov @ w_inv)
    print(f"Inverse Volatility annualized volatility: {vol_inv:.4f}")
    
    # 3. Minimum Variance with constraints: sum(w)=1, 0 <= w <= 0.40
    def objective(w):
        return w @ cov @ w
        
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 0.40) for _ in range(N)]
    res = minimize(objective, w_eq, method='SLSQP', bounds=bounds, constraints=cons)
    
    if res.success:
        w_minvar = res.x
        vol_minvar = np.sqrt(w_minvar @ cov @ w_minvar)
        print(f"Minimum Variance annualized volatility: {vol_minvar:.4f}")
        print("MinVar weights:", {s: round(w, 3) for s, w in zip(stocks, w_minvar) if w > 0.01})
    else:
        print("MinVar optimization failed!")

    # 4. Top-3 subset optimization
    top3_idx = [0, 2, 4]
    cov_top3 = cov[np.ix_(top3_idx, top3_idx)]
    cons_top3 = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds_top3 = [(0.10, 0.60) for _ in range(3)]
    res_top3 = minimize(lambda w: w @ cov_top3 @ w, np.ones(3)/3, method='SLSQP', bounds=bounds_top3, constraints=cons_top3)
    if res_top3.success:
        print("Top-3 MinVar weights:", res_top3.x)
        vol_top3 = np.sqrt(res_top3.x @ cov_top3 @ res_top3.x)
        print(f"Top-3 MinVar volatility: {vol_top3:.4f}")

if __name__ == "__main__":
    test_optimizer()
