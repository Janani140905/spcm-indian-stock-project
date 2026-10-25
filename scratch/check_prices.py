"""
Scratch script to inspect price alignment, trading calendars, and return series.
"""
import pandas as pd
import numpy as np

def check_prices():
    prices = pd.read_csv("data/processed/india/prices.csv")
    prices['date'] = pd.to_datetime(prices['date'])
    oos = pd.read_csv("reports/phase6_oos_predictions.csv")
    oos['date'] = pd.to_datetime(oos['date'])
    
    unique_oos_dates = sorted(oos['date'].unique())
    print(f"OOS dates: {len(unique_oos_dates)} dates, from {unique_oos_dates[0].date()} to {unique_oos_dates[-1].date()}")
    
    # Pivot prices to close matrix
    piv_close = prices.pivot(index='date', columns='stock', values='close')
    piv_open = prices.pivot(index='date', columns='stock', values='open')
    
    print("Close matrix shape:", piv_close.shape)
    print("Missing close values in OOS period:", piv_close.loc[unique_oos_dates[0]:unique_oos_dates[-1]].isna().sum().to_dict())
    print("Missing open values in OOS period:", piv_open.loc[unique_oos_dates[0]:unique_oos_dates[-1]].isna().sum().to_dict())
    
    # Check 10-day forward returns alignment
    # In OOS, fwd_ret is given for each date. Let's check how it compares with piv_close
    trading_dates = sorted(piv_close.index)
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    sample_row = oos[(oos['model'] == 'RF') & (oos['feature_set'] == 'Price + Sentiment')].iloc[0]
    s_date = sample_row['date']
    s_stock = sample_row['stock']
    s_ret = sample_row['fwd_ret']
    
    idx_t = date_to_idx[s_date]
    idx_t10 = idx_t + 10
    d_t10 = trading_dates[idx_t10]
    c_t = piv_close.loc[s_date, s_stock]
    c_t10 = piv_close.loc[d_t10, s_stock]
    computed_ret = (c_t10 - c_t) / c_t
    print(f"\nSample date {s_date.date()} stock {s_stock}:")
    print(f"  OOS fwd_ret: {s_ret:.6f}")
    print(f"  Computed ret (close_{d_t10.date()} / close_{s_date.date()} - 1): {computed_ret:.6f}")
    assert abs(s_ret - computed_ret) < 1e-5, "Mismatch in fwd_ret!"
    print("  [MATCH] fwd_ret aligns perfectly with price close at t and t+10.")

if __name__ == "__main__":
    check_prices()
