"""
Scratch script to prototype PortfolioBacktester and evaluate all models and strategies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

def run_prototype():
    # 1. Load prices and pivot
    prices = pd.read_csv("data/processed/india/prices.csv")
    prices['date'] = pd.to_datetime(prices['date'])
    piv_close = prices.pivot(index='date', columns='stock', values='close')
    piv_open = prices.pivot(index='date', columns='stock', values='open')
    
    # 2. Load OOS predictions
    oos = pd.read_csv("reports/phase6_oos_predictions.csv")
    oos['date'] = pd.to_datetime(oos['date'])
    
    # Filter for RF and NN (Price + Sentiment)
    trading_dates = sorted(piv_close.index)
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    oos_dates = sorted(oos['date'].unique())
    print(f"Total OOS dates: {len(oos_dates)} (from {oos_dates[0].date()} to {oos_dates[-1].date()})")
    
    # Select rebalance dates every 10 trading days
    rebalance_freq = 10
    rebalance_dates = oos_dates[::rebalance_freq]
    print(f"Rebalance cycles ({rebalance_freq} days): {len(rebalance_dates)} cycles")
    
    # Define strategy functions
    def get_weights_equal(symbols):
        return {s: 1.0 / len(symbols) for s in symbols}
        
    def get_weights_topk(scores_df, k=3):
        topk = scores_df.sort_values("y_prob", ascending=False).head(k)
        return {s: 1.0 / k for s in topk['stock']}
        
    def get_weights_score_weighted(scores_df, k=3):
        topk = scores_df.sort_values("y_prob", ascending=False).head(k)
        tot = topk['y_prob'].sum()
        if tot <= 0:
            return {s: 1.0 / k for s in topk['stock']}
        return {s: prob / tot for s, prob in zip(topk['stock'], topk['y_prob'])}

    def get_weights_topk_minvar(scores_df, cov_matrix, k=3, max_weight=0.50):
        topk = scores_df.sort_values("y_prob", ascending=False).head(k)
        topk_stocks = list(topk['stock'])
        cov_sub = cov_matrix.loc[topk_stocks, topk_stocks].values
        
        # Min variance
        init_w = np.ones(k) / k
        bounds = [(0.10, max_weight) for _ in range(k)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res = minimize(lambda w: w @ cov_sub @ w, init_w, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            return {s: w for s, w in zip(topk_stocks, res.x)}
        return {s: 1.0 / k for s in topk_stocks}

    def get_weights_universe_minvar(symbols, cov_matrix, max_weight=0.35):
        cov_sub = cov_matrix.loc[symbols, symbols].values
        N = len(symbols)
        init_w = np.ones(N) / N
        bounds = [(0.0, max_weight) for _ in range(N)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res = minimize(lambda w: w @ cov_sub @ w, init_w, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            return {s: w for s, w in zip(symbols, res.x) if w > 0.001}
        return {s: 1.0 / N for s in symbols}

    # Run backtest for a strategy
    def simulate_strategy(model_name, strat_type, cost_bps=15.0):
        initial_capital = 1_000_000.0
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {s: 0.0 for s in piv_close.columns}
        
        daily_records = []
        trade_records = []
        
        # Start backtest from first rebalance date
        start_idx = date_to_idx[rebalance_dates[0]]
        # We need next day to execute
        exec_start_idx = start_idx + 1
        last_date = oos_dates[-1]
        last_idx = date_to_idx[last_date]
        
        # Rebalance calendar
        rebal_set = set(rebalance_dates)
        
        for idx in range(exec_start_idx, last_idx + 1):
            curr_date = trading_dates[idx]
            prev_date = trading_dates[idx - 1]
            
            # Check if prev_date was a decision date
            is_rebal_day = (prev_date in rebal_set)
            
            if is_rebal_day:
                # Mark portfolio value at open of curr_date
                open_val = cash + sum(positions[s] * piv_open.loc[curr_date, s] for s in positions)
                
                # Compute Covariance using historical returns up to prev_date
                cov_start_idx = max(0, idx - 1 - 60)
                hist_returns = piv_close.iloc[cov_start_idx : idx].pct_change().dropna()
                cov_mat = hist_returns.cov() * 252
                
                # Get model scores from prev_date
                model_oos = oos[(oos['date'] == prev_date) & (oos['model'] == model_name) & (oos['feature_set'] == 'Price + Sentiment')]
                
                if strat_type == "equal_weight":
                    target_w = get_weights_equal(list(piv_close.columns))
                elif strat_type == "topk_equal":
                    target_w = get_weights_topk(model_oos, k=3)
                elif strat_type == "topk_score":
                    target_w = get_weights_score_weighted(model_oos, k=3)
                elif strat_type == "topk_minvar":
                    target_w = get_weights_topk_minvar(model_oos, cov_mat, k=3)
                elif strat_type == "universe_minvar":
                    target_w = get_weights_universe_minvar(list(piv_close.columns), cov_mat)
                else:
                    raise ValueError(strat_type)
                    
                # Execute rebalance at curr_date OPEN
                # Compute target dollar amounts
                target_dollars = {s: target_w.get(s, 0.0) * open_val for s in piv_close.columns}
                current_dollars = {s: positions[s] * piv_open.loc[curr_date, s] for s in piv_close.columns}
                
                # Turnover & costs
                trade_dollars = {s: target_dollars[s] - current_dollars[s] for s in piv_close.columns}
                turnover = sum(abs(trade_dollars[s]) for s in piv_close.columns)
                t_cost = turnover * (cost_bps / 10000.0)
                
                # Net available capital after cost
                net_val = open_val - t_cost
                new_positions = {}
                for s in piv_close.columns:
                    target_val_s = target_w.get(s, 0.0) * net_val
                    new_positions[s] = target_val_s / piv_open.loc[curr_date, s]
                    
                positions = new_positions
                cash = 0.0 # fully invested
                
                trade_records.append({
                    "date": curr_date,
                    "open_val": open_val,
                    "turnover": turnover,
                    "cost": t_cost,
                    "holdings": target_w
                })
                
            # End of day mark-to-market at CLOSE
            close_val = cash + sum(positions[s] * piv_close.loc[curr_date, s] for s in positions)
            daily_records.append({
                "date": curr_date,
                "portfolio_value": close_val,
            })
            
        df_daily = pd.DataFrame(daily_records)
        df_daily['daily_return'] = df_daily['portfolio_value'].pct_change().fillna(0.0)
        
        # Summary metrics
        total_ret = (df_daily['portfolio_value'].iloc[-1] / initial_capital) - 1.0
        n_days = len(df_daily)
        cagr = (1.0 + total_ret) ** (252.0 / n_days) - 1.0
        ann_vol = df_daily['daily_return'].std() * np.sqrt(252)
        rf = 0.06
        sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else 0.0
        
        # Max Drawdown
        cummax = df_daily['portfolio_value'].cummax()
        drawdown = (df_daily['portfolio_value'] - cummax) / cummax
        mdd = drawdown.min()
        
        total_costs = sum(t['cost'] for t in trade_records)
        
        return {
            "model": model_name,
            "strategy": strat_type,
            "total_return": total_ret,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "mdd": mdd,
            "total_costs": total_costs,
            "final_value": df_daily['portfolio_value'].iloc[-1]
        }

    print("\n=== Running Prototype Backtest (Cycle = 10 days, Cost = 15 bps, Exec = Next Open) ===")
    results = []
    # Benchmark
    results.append(simulate_strategy("RF", "equal_weight"))
    results.append(simulate_strategy("RF", "universe_minvar"))
    
    # Models
    for m in ["RF", "NN", "KNN"]:
        results.append(simulate_strategy(m, "topk_equal"))
        results.append(simulate_strategy(m, "topk_score"))
        results.append(simulate_strategy(m, "topk_minvar"))
        
    df_res = pd.DataFrame(results)
    for col in ["total_return", "cagr", "ann_vol", "sharpe", "mdd"]:
        df_res[col] = df_res[col].apply(lambda x: f"{x:.4f}")
    print(df_res[["model", "strategy", "total_return", "cagr", "ann_vol", "sharpe", "mdd", "total_costs"]].to_string())

if __name__ == "__main__":
    run_prototype()
