"""
Scratch script to investigate the exact arithmetic of the first rebalance cycle (2022-01-04).
"""
import pandas as pd
import numpy as np

def investigate_first_cycle():
    df_curves = pd.read_csv("reports/phase7_equity_curves.csv")
    df_prices = pd.read_csv("data/processed/india/prices.csv")
    df_trades = pd.read_csv("reports/phase7_trades_turnover.csv")
    
    print("Curves row 0 (2022-01-04):")
    row0 = df_curves.iloc[0]
    for col in ["date", "Benchmark_EqualWeight", "RF_topk_equal", "NN_topk_equal"]:
        print(f"  {col}: {row0[col]}")
        
    # Check trades on 2022-01-04
    trades_rf = df_trades[df_trades['strategy'] == 'RF_topk_equal']
    print(f"\nRF trades count: {len(trades_rf)}")
    t0 = trades_rf.iloc[0]
    print("Trade 0 details:")
    print(t0.to_dict())
    
    # Check prices on 2022-01-04 for BHARTIARTL, HINDUNILVR, INFY
    p_0104 = df_prices[df_prices['date'] == '2022-01-04'].set_index('stock')
    print("\nPrices on 2022-01-04:")
    for s in ["BHARTIARTL", "HINDUNILVR", "INFY"]:
        row = p_0104.loc[s]
        print(f"  {s}: Open={row['open']:.2f}, Close={row['close']:.2f}")

    # Let's inspect what happens in backtest.py on the very first trade (exec_start_idx):
    # What was pre_rebal_val?
    # Initial capital = 1,000,000
    # Before rebalance on 2022-01-04:
    # positions = {s: 0.0}
    # cash = 1,000,000
    # pre_rebal_val = cash + 0 = 1,000,000
    # turnover = 1,000,000
    # cost = 1,500 (15 bps)
    # net_val = 998,500
    # Target weights = 1/3 each (0.3333333333333333)
    # Dollars per stock = 998,500 / 3 = 332,833.333333
    # Shares:
    # BHARTIARTL: 332,833.333333 / 486.9761962890625 ???
    # WAIT! What was the Open price of BHARTIARTL on 2022-01-04 in data/processed/india/prices.csv???
    print("\nExact prices in prices.csv on 2022-01-04:")
    for s in ["BHARTIARTL", "HINDUNILVR", "INFY"]:
        row = p_0104.loc[s]
        print(f"  {s}: Open={row['open']}, Close={row['close']}")
        sh = (998500.0 / 3.0) / row['open']
        val_close = sh * row['close']
        print(f"    shares = {sh:.6f}, close value = {val_close:.2f}")

if __name__ == "__main__":
    investigate_first_cycle()
