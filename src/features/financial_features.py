"""
Financial variable construction (paper Sec III.C.3, eq. 5-7) and the homophily
moderation regression (Sec IV.B, eq. 8 / Table 3).

    R_{s,t} - R_{f,t} = a_s + b1*(R_m,t - R_f,t) + b2*SMB_t + b3*HML_t + b4*RMW_t + b5*CMA_t + e   (5)
    AR_{s,t} = (R_{s,t}-R_{f,t}) - fitted(5)                                                        (6)
    CAR[t1,t2]_s = prod(1+R_s,t) - prod(1+R_p,t)  over [t1,t2]                                     (7)

Rolling 250-trading-day OLS per stock (as stated in the paper) produces out-of-sample-ish
FF5 coefficients used to compute the abnormal return on the *next* day (rolling window
precedes the target day).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults

ROLL_WINDOW = 250  # trading days, per paper


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["stock", "date"]).copy()
    prices["ret"] = prices.groupby("stock")["close"].pct_change()
    return prices


def _rolling_ff5_ar_for_stock(stock_df: pd.DataFrame, ff5: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Compute AR_{s,t} for a single stock using a trailing `window`-day rolling FF5 regression."""
    df = stock_df.merge(ff5, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["excess_ret"] = df["ret"] - df["rf"]

    ar = np.full(len(df), np.nan)
    # Dynamically detect available factors
    available_factors = [col for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "wml"] if col in df.columns]
    
    for i in range(window, len(df)):
        train = df.iloc[i - window:i]
        if train["excess_ret"].isna().any():
            continue
        X = sm.add_constant(train[available_factors].values)
        y = train["excess_ret"].values
        try:
            model: RegressionResults = sm.OLS(y, X).fit()
        except Exception:
            continue
        x_t = np.concatenate([[1.0], df.iloc[i][available_factors].values.astype(float)])
        expected = x_t @ model.params
        ar[i] = df.iloc[i]["excess_ret"] - expected

    df["AR"] = ar
    return df[["stock", "date", "ret", "excess_ret", "AR"]]


def compute_abnormal_returns(prices: pd.DataFrame, ff5: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Compute AR_{s,t} for every stock. Returns long DataFrame (stock, date, ret, excess_ret, AR)."""
    prices = compute_daily_returns(prices)
    out = []
    for stock, g in prices.groupby("stock"):
        out.append(_rolling_ff5_ar_for_stock(g, ff5, window=window))
    return pd.concat(out, ignore_index=True)


def compute_car(ar_df: pd.DataFrame, t1: int, t2: int) -> pd.DataFrame:
    """
    Cumulative abnormal return CAR[t1,t2]_s at each date, using AR as a proxy for the
    (R_s,t - R_p,t) spread already embedded in the abnormal return (paper eq. 7).
    t1, t2 are offsets in trading days relative to the "current" date (e.g. t1=1, t2=10
    for CAR[t+1, t+10]; t1=-5, t2=-1 for CAR[t-1, t-5]).
    """
    ar_df = ar_df.sort_values(["stock", "date"]).reset_index(drop=True)
    results = []
    for stock, g in ar_df.groupby("stock"):
        g = g.reset_index(drop=True)
        ar_vals = (1 + g["AR"].fillna(0)).values
        n = len(g)
        car = np.full(n, np.nan)
        for i in range(n):
            lo, hi = i + t1, i + t2
            lo, hi = min(lo, hi), max(lo, hi)
            if lo < 0 or hi >= n:
                continue
            car[i] = np.prod(ar_vals[lo:hi + 1]) - 1
        g[f"CAR_{t1}_{t2}"] = car
        results.append(g[["stock", "date", f"CAR_{t1}_{t2}"]])
    return pd.concat(results, ignore_index=True)


def build_control_variables(ar_df: pd.DataFrame) -> pd.DataFrame:
    """Builds CAR[t-1,t-5], CAR[t-6,t-20], Volatility, AbLogVolume-ready base table."""
    car_recent = compute_car(ar_df, -5, -1).rename(columns={"CAR_-5_-1": "CAR_t1_t5"})
    car_mid = compute_car(ar_df, -20, -6).rename(columns={"CAR_-20_-6": "CAR_t6_t20"})
    merged = ar_df.merge(car_recent, on=["stock", "date"]).merge(car_mid, on=["stock", "date"])
    merged["Volatility"] = merged.groupby("stock")["AR"].transform(lambda s: s.rolling(20, min_periods=5).std())
    return merged


def run_homophily_regression(panel: pd.DataFrame):
    """
    Reproduce paper eq. (8) / Table 3:

        CAR[t+1,t+10]_s = b1*Sent_s,t + b2*Homo_s,t + b3*Sent_s,t*Homo_s,t + X*delta
                           + stock_FE + date_FE + eps

    `panel` must contain columns: CAR_fwd (dependent var), Sent, Homo, plus any control
    columns you want in `controls`, and `stock`, `date` for fixed effects.
    Uses statsmodels OLS with stock & date dummies; for clustered SEs by (stock, year-month)
    pass cov_type='cluster' with the appropriate groups.
    """
    import statsmodels.formula.api as smf

    df = panel.copy()
    df["Sent_x_Homo"] = df["Sent"] * df["Homo"]
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    control_cols = [c for c in df.columns if c.startswith("CAR_t") or c in
                    ("Volatility", "AbLogVolume", "PostVolume", "Agreement",
                     "InterNewsSent", "InterNewsReport", "PaperNewsSent", "PaperNewsReport",
                     "BM", "ROA", "lnSize")]
    rhs = ["Sent", "Homo", "Sent_x_Homo"] + control_cols
    formula = "CAR_fwd ~ " + " + ".join(rhs) + " + C(stock) + C(date)"

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[["stock", "year_month"]].apply(tuple, axis=1)}
    )
    return model
