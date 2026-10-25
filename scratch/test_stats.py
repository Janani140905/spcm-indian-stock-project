"""
Scratch script to test statistical significance, annual breakdowns, and cost sensitivity.
"""
import pandas as pd
import numpy as np
from scipy import stats

def test_stats():
    # Let's verify paired t-test on daily excess returns
    # Create sample returns
    np.random.seed(42)
    strat_ret = np.random.normal(0.001, 0.01, 720)
    bench_ret = np.random.normal(0.0006, 0.009, 720)
    excess_ret = strat_ret - bench_ret
    
    t_stat, p_val = stats.ttest_1samp(excess_ret, 0.0)
    print(f"Sample test: t_stat={t_stat:.4f}, p_val={p_val:.4f}")
    
    # Check HHI and Effective N
    w = np.array([0.333, 0.333, 0.334])
    hhi = np.sum(w**2)
    eff_n = 1.0 / hhi
    print(f"Top 3 equal: HHI={hhi:.4f}, Effective N={eff_n:.2f}")

if __name__ == "__main__":
    test_stats()
