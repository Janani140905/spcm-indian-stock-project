"""
Quick sanity check for the newly created src/portfolio modules.
"""
import pandas as pd
from src.portfolio.backtest import BacktestConfig, BacktestEngine

def main():
    prices = pd.read_csv("data/processed/india/prices.csv")
    oos = pd.read_csv("reports/phase6_oos_predictions.csv")
    
    cfg = BacktestConfig(cost_bps=15.0, rebalance_freq=10)
    engine = BacktestEngine(prices, oos, cfg)
    
    # Run EqualWeight
    print("Testing EqualWeight...")
    ew_res = engine.run_strategy("equal_weight", "RF")
    print(f"EqualWeight CAGR: {ew_res.report.cagr:.4f}, Sharpe: {ew_res.report.sharpe_ratio:.4f}")
    
    # Run RF TopK Equal
    print("Testing RF TopK Equal...")
    rf_res = engine.run_strategy("topk_equal", "RF")
    print(f"RF TopK Equal CAGR: {rf_res.report.cagr:.4f}, Sharpe: {rf_res.report.sharpe_ratio:.4f}, Alpha: {rf_res.report.alpha_cagr:.4f}")
    
    # Run NN TopK Equal (Production MLP)
    print("Testing NN TopK Equal...")
    nn_res = engine.run_strategy("topk_equal", "NN")
    print(f"NN TopK Equal CAGR: {nn_res.report.cagr:.4f}, Sharpe: {nn_res.report.sharpe_ratio:.4f}, Alpha: {nn_res.report.alpha_cagr:.4f}")
    
    # Run RF TopK MinVar
    print("Testing RF TopK MinVar...")
    rf_mv_res = engine.run_strategy("topk_minvar", "RF")
    print(f"RF TopK MinVar CAGR: {rf_mv_res.report.cagr:.4f}, Sharpe: {rf_mv_res.report.sharpe_ratio:.4f}")
    
    print("Sanity check PASSED successfully!")

if __name__ == "__main__":
    main()
