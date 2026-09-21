"""
Real A-share data fetcher using AKShare, to replace the synthetic data in
`data/raw/` with real CSI300 constituents, prices, fundamentals, and FF5-style factors.

IMPORTANT: AKShare's function names have changed across versions before (this script
was written/verified against akshare==1.18.81 -- run `pip show akshare` to check yours).
If a call below throws AttributeError, run:
    python -c "import akshare as ak; print([f for f in dir(ak) if 'KEYWORD' in f])"
to find the current name and swap it in. The functions used here and confirmed to
exist in 1.18.81:
    ak.index_stock_cons(symbol="000300")            # CSI300 constituents
    ak.stock_a_code_to_symbol(symbol="000001")       # code -> exchange-prefixed symbol
    ak.stock_zh_a_hist(symbol, period, start_date, end_date, adjust)  # OHLCV + turnover
    ak.stock_value_em(symbol)                        # daily PE/PB/market cap
    ak.stock_financial_analysis_indicator_em(symbol) # ROE/ROA/profitability ratios
    ak.bond_zh_us_rate(start_date)                   # China/US bond yields (risk-free proxy)

USAGE
-----
    python -m src.utils.akshare_fetcher fetch --n_stocks 25 --start 2018-01-01 --end 2020-12-31

This populates data/raw/prices.csv, data/raw/fundamentals.csv, and data/raw/ff5_factors.csv
with the schemas expected by the rest of the pipeline (see src/utils/data_loader.py).
It does NOT fetch Guba posts -- see the README section on scraping those separately;
this script only covers the "Wind" and "CSMAR" substitutes.

NOTE ON RUNNING THIS: it needs real internet access to Eastmoney/Sina's servers. If you're
running inside a sandboxed/offline environment, run this on your own machine instead.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError as e:
    raise ImportError(
        "akshare not installed. Run: pip install akshare"
    ) from e

from src.utils.data_loader import DATA_DIR


# ---------------------------------------------------------------------------
# 1. Universe selection
# ---------------------------------------------------------------------------
def get_csi300_constituents(n_stocks: int | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: code (6-digit), name.
    Set n_stocks to sample a smaller pilot universe (recommended: 20-30 for a first pass,
    rather than the full 300 -- much faster and kinder to Eastmoney's servers).
    """
    df = ak.index_stock_cons(symbol="000300")
    # Column names in this endpoint are Chinese: '品种代码' (code), '品种名称' (name)
    code_col = [c for c in df.columns if "代码" in c][0]
    name_col = [c for c in df.columns if "名称" in c][0]
    out = df[[code_col, name_col]].rename(columns={code_col: "code", name_col: "name"})
    out["code"] = out["code"].astype(str).str.zfill(6)
    if n_stocks:
        out = out.sample(n=min(n_stocks, len(out)), random_state=42).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. Prices
# ---------------------------------------------------------------------------
def fetch_price_history(code: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch one stock's daily OHLCV + turnover via stock_zh_a_hist (qfq-adjusted)."""
    start_fmt = start.replace("-", "")
    end_fmt = end.replace("-", "")
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                 start_date=start_fmt, end_date=end_fmt, adjust="qfq")
    except Exception as e:
        print(f"  [warn] price fetch failed for {code}: {e}")
        return None
    if df is None or df.empty:
        return None

    # Chinese column names -> standard schema. AKShare's stock_zh_a_hist typically returns:
    # 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    rename_map = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount", "换手率": "turnover_pct",
    }
    df = df.rename(columns=rename_map)
    df["stock"] = code
    df["date"] = pd.to_datetime(df["date"])
    df["turnover"] = df["turnover_pct"].astype(float) / 100.0  # % -> fraction
    return df[["stock", "date", "open", "high", "low", "close", "volume", "turnover"]]


def fetch_valuation(code: str) -> pd.DataFrame | None:
    """Fetch daily PE/PB/market-cap history via stock_value_em (whole history, filter later)."""
    try:
        df = ak.stock_value_em(symbol=code)
    except Exception as e:
        print(f"  [warn] valuation fetch failed for {code}: {e}")
        return None
    if df is None or df.empty:
        return None
    # Typical columns: 数据日期, PE(TTM)/PE(静), PB, 总市值, 总股本 -- names vary by akshare version.
    date_col = next((c for c in df.columns if "日期" in c), None)
    pe_col = next((c for c in df.columns if "PE" in c.upper() and "TTM" in c.upper()), None) \
        or next((c for c in df.columns if "PE" in c.upper()), None)
    pb_col = next((c for c in df.columns if "PB" in c.upper()), None)
    mv_col = next((c for c in df.columns if "总市值" in c), None)
    shares_col = next((c for c in df.columns if "总股本" in c), None)
    if date_col is None:
        return None

    out = pd.DataFrame({"date": pd.to_datetime(df[date_col])})
    out["stock"] = code
    out["pe"] = pd.to_numeric(df[pe_col], errors="coerce") if pe_col else np.nan
    out["pb"] = pd.to_numeric(df[pb_col], errors="coerce") if pb_col else np.nan
    out["market_cap"] = pd.to_numeric(df[mv_col], errors="coerce") if mv_col else np.nan
    out["shares_outstanding"] = pd.to_numeric(df[shares_col], errors="coerce") if shares_col else np.nan
    return out


def build_prices_table(codes: list[str], start: str, end: str, pause: float = 0.3) -> pd.DataFrame:
    """Assemble the full prices_df matching src/utils/data_loader.py's schema."""
    price_frames, val_frames = [], []
    for i, code in enumerate(codes):
        print(f"[{i+1}/{len(codes)}] fetching prices for {code} ...")
        p = fetch_price_history(code, start, end)
        if p is not None:
            price_frames.append(p)
        time.sleep(pause)  # be polite to the API -- don't hammer it

        v = fetch_valuation(code)
        if v is not None:
            val_frames.append(v)
        time.sleep(pause)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    valuation = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()

    if not valuation.empty:
        merged = prices.merge(valuation, on=["stock", "date"], how="left")
    else:
        merged = prices.copy()
        for c in ["pe", "pb", "market_cap", "shares_outstanding"]:
            merged[c] = np.nan

    date_range = (start, end)
    merged = merged[(merged["date"] >= date_range[0]) & (merged["date"] <= date_range[1])]

    # Fill remaining schema columns not directly available from AKShare with
    # best-effort derived values (documented approximations, not exact CSMAR data):
    merged["pcf"] = np.nan          # price-to-cash-flow: not directly in stock_value_em
    merged["nav"] = np.where(merged["pb"] > 0, merged["close"] / merged["pb"], np.nan)  # NAV per share ~= price / PB
    merged["income_per_share"] = np.nan  # requires income statement join -- left NaN, fill from fundamentals if needed
    merged["liquidity"] = np.nan     # current assets / current liabilities -- requires balance sheet, see fundamentals fetcher

    merged = merged.sort_values(["stock", "date"]).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# 3. Fundamentals (Size, BM, ROA)
# ---------------------------------------------------------------------------
def fetch_fundamentals_for_stock(code: str) -> pd.DataFrame | None:
    """
    Pull ROA/ROE from stock_financial_analysis_indicator_em, quarterly.
    Note: symbol for this endpoint typically needs an exchange suffix, e.g. '000001.SZ'.
    We try both bare code and a couple of common suffix conventions.
    """
    suffix_guess = ".SZ" if code.startswith(("0", "3")) else ".SH"
    for sym in (code + suffix_guess, code):
        try:
            df = ak.stock_financial_analysis_indicator_em(symbol=sym)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        print(f"  [warn] fundamentals fetch failed for {code}")
        return None

    # Expect columns like: 指标名称, 报告期, 指标值 (long format) -- pivot to find ROA/ROE rows.
    name_col = next((c for c in df.columns if "指标名称" in c or "指标" == c), None)
    period_col = next((c for c in df.columns if "报告期" in c), None)
    value_col = next((c for c in df.columns if "指标值" in c or "数值" in c), None)
    if name_col is None or period_col is None:
        return None

    roa_rows = df[df[name_col].astype(str).str.contains("ROA|总资产收益率|总资产报酬率", regex=True, na=False)]
    if roa_rows.empty:
        return None

    out = pd.DataFrame({
        "date": pd.to_datetime(roa_rows[period_col].astype(str), errors="coerce"),
        "roa": pd.to_numeric(roa_rows[value_col], errors="coerce") / 100.0,  # % -> fraction
    })
    out["stock"] = code
    return out.dropna(subset=["date"])


def build_fundamentals_table(codes: list[str], prices_df: pd.DataFrame, start: str, end: str, pause: float = 0.3) -> pd.DataFrame:
    """
    Builds fundamentals_df with columns: stock, date, size, bm, roa
    - size = ln(market_cap), forward-filled daily from prices_df's market_cap
    - bm   = 1 / PB (book-to-market = inverse of price-to-book), forward-filled daily
    - roa  = quarterly ROA, forward-filled to daily between report dates
    """
    roa_frames = []
    for i, code in enumerate(codes):
        print(f"[{i+1}/{len(codes)}] fetching fundamentals for {code} ...")
        f = fetch_fundamentals_for_stock(code)
        if f is not None:
            roa_frames.append(f)
        time.sleep(pause)

    roa_df = pd.concat(roa_frames, ignore_index=True) if roa_frames else pd.DataFrame(columns=["stock", "date", "roa"])

    all_dates = pd.date_range(start, end, freq="B")
    out_frames = []
    for code in codes:
        stock_px = prices_df[prices_df["stock"] == code].set_index("date")
        stock_px = stock_px.reindex(all_dates).ffill()
        size = np.log(stock_px["market_cap"].replace(0, np.nan))
        bm = 1.0 / stock_px["pb"].replace(0, np.nan)

        stock_roa = roa_df[roa_df["stock"] == code].set_index("date")["roa"]
        stock_roa = stock_roa.reindex(all_dates).ffill().bfill()

        frame = pd.DataFrame({"date": all_dates, "stock": code, "size": size.values,
                               "bm": bm.values, "roa": stock_roa.values})
        out_frames.append(frame)

    fundamentals = pd.concat(out_frames, ignore_index=True)
    fundamentals = fundamentals.dropna(subset=["size"])
    return fundamentals


# ---------------------------------------------------------------------------
# 4. FF5-style factors (simplified 2x3 sort -- see docstring caveats)
# ---------------------------------------------------------------------------
def fetch_risk_free_rate(start: str, end: str) -> pd.DataFrame:
    """
    Risk-free rate proxy from China government bond yield (via bond_zh_us_rate, which
    returns both China and US series). Falls back to a flat approximation if the fetch
    fails or the expected column isn't found -- check your akshare version's column
    names with `ak.bond_zh_us_rate().columns` if this happens.
    """
    try:
        df = ak.bond_zh_us_rate(start_date=start.replace("-", ""))
        cn_col = next((c for c in df.columns if "中国" in c and ("10" in c or "1" in c)), None)
        date_col = next((c for c in df.columns if "日期" in c), df.columns[0])
        if cn_col is None:
            raise ValueError("could not find a China yield column")
        df = df[[date_col, cn_col]].rename(columns={date_col: "date", cn_col: "cn_yield_pct"})
        df["date"] = pd.to_datetime(df["date"])
        df["rf"] = (df["cn_yield_pct"].astype(float) / 100.0) / 252.0  # annualized % -> daily rate
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df[["date", "rf"]]
    except Exception as e:
        print(f"  [warn] risk-free rate fetch failed ({e}); falling back to flat 2.5%/yr approximation")
        dates = pd.date_range(start, end, freq="B")
        return pd.DataFrame({"date": dates, "rf": 0.025 / 252.0})


def _monthly_2x3_sort(cross_section: pd.DataFrame, size_col: str, factor_col: str, ret_col: str):
    """
    One month's 2x3 (size x factor) sort. Returns the long-short factor return for that month:
    (avg return of small,{high,low}) etc. Standard Fama-French construction, condensed.
    """
    cs = cross_section.dropna(subset=[size_col, factor_col, ret_col]).copy()
    if len(cs) < 10:
        return np.nan

    size_median = cs[size_col].median()
    cs["size_bucket"] = np.where(cs[size_col] <= size_median, "S", "B")

    f_low, f_high = cs[factor_col].quantile([0.3, 0.7])
    cs["factor_bucket"] = np.select(
        [cs[factor_col] <= f_low, cs[factor_col] >= f_high],
        ["L", "H"], default="M",
    )

    grp_ret = cs.groupby(["size_bucket", "factor_bucket"])[ret_col].mean()

    def g(sz, fb):
        return grp_ret.get((sz, fb), np.nan)

    high = np.nanmean([g("S", "H"), g("B", "H")])
    low = np.nanmean([g("S", "L"), g("B", "L")])
    return high - low  # HML/RMW/CMA-style long-short factor for this month


def build_ff5_factors(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame, rf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified monthly 2x3 sort producing SMB/HML/RMW/CMA-style factors, broadcast to
    daily frequency (each day in a month gets that month's factor realization, converted
    to a daily-equivalent rate). This is an approximation of the full CSMAR/CRSP-style
    construction -- notably:
      - profitability proxy = ROA (paper's own control variable) rather than operating
        profitability from the income statement
      - investment proxy = market-cap growth rather than total-asset growth (asset growth
        needs a balance-sheet fetch not wired in here)
    Good enough for research/demo purposes; swap in a stricter construction if you need
    publication-grade factors.
    """
    px = prices_df.sort_values(["stock", "date"]).copy()
    px["month"] = px["date"].dt.to_period("M")
    px["ret"] = px.groupby("stock")["close"].pct_change()

    fnd = fundamentals_df.sort_values(["stock", "date"]).copy()
    fnd["month"] = fnd["date"].dt.to_period("M")

    monthly_ret = px.groupby(["stock", "month"])["ret"].apply(lambda s: (1 + s).prod() - 1).reset_index()
    monthly_size = px.groupby(["stock", "month"])["close"].last().reset_index()  # placeholder, replaced by fnd size below
    monthly_fnd = fnd.groupby(["stock", "month"]).last().reset_index()[["stock", "month", "size", "bm", "roa"]]

    monthly_fnd["mcap_growth"] = monthly_fnd.groupby("stock")["size"].diff()  # ln(mcap) diff ~= growth rate

    panel = monthly_ret.merge(monthly_fnd, on=["stock", "month"], how="inner")

    # Market factor: cap-weighted-ish average return across the universe each month
    mkt = panel.groupby("month")["ret"].mean().rename("mkt_ret")

    hml_series, rmw_series, cma_series, smb_series = {}, {}, {}, {}
    for month, cs in panel.groupby("month"):
        hml_series[month] = _monthly_2x3_sort(cs, "size", "bm", "ret")
        rmw_series[month] = _monthly_2x3_sort(cs, "size", "roa", "ret")
        cma_series[month] = _monthly_2x3_sort(cs, "size", "mcap_growth", "ret")
        size_median = cs["size"].median()
        small = cs[cs["size"] <= size_median]["ret"].mean()
        big = cs[cs["size"] > size_median]["ret"].mean()
        smb_series[month] = small - big

    monthly_factors = pd.DataFrame({
        "hml": pd.Series(hml_series), "rmw": pd.Series(rmw_series),
        "cma": pd.Series(cma_series), "smb": pd.Series(smb_series),
    })
    monthly_factors = monthly_factors.join(mkt)
    monthly_factors.index = monthly_factors.index.to_timestamp()

    # Broadcast monthly factor realization to a daily-equivalent rate (compounding assumption)
    daily_dates = pd.date_range(prices_df["date"].min(), prices_df["date"].max(), freq="B")
    daily = pd.DataFrame({"date": daily_dates})
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    out_rows = []
    for month_start, month_factors in monthly_factors.iterrows():
        n_days_in_month = (daily["month_start"] == month_start).sum()
        if n_days_in_month == 0:
            continue
        daily_equiv = {
            k: (1 + v) ** (1 / n_days_in_month) - 1 if pd.notna(v) else 0.0
            for k, v in month_factors.items()
        }
        days = daily.loc[daily["month_start"] == month_start, "date"]
        for d in days:
            out_rows.append({"date": d, **daily_equiv})

    ff5 = pd.DataFrame(out_rows)
    ff5 = ff5.rename(columns={"mkt_ret": "mkt_rf_raw"})
    ff5 = ff5.merge(rf_df, on="date", how="left")
    ff5["rf"] = ff5["rf"].fillna(ff5["rf"].mean())
    ff5["mkt_rf"] = ff5["mkt_rf_raw"] - ff5["rf"]
    return ff5[["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--n_stocks", type=int, default=25, help="pilot universe size (recommend 20-30, not full 300)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2020-12-31")
    p.add_argument("--pause", type=float, default=0.3, help="seconds between API calls, be polite to the server")
    args = parser.parse_args()

    out_dir = DATA_DIR / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Fetching CSI300 constituent list ==")
    constituents = get_csi300_constituents(n_stocks=args.n_stocks)
    codes = constituents["code"].tolist()
    print(f"Selected {len(codes)} stocks: {codes}")

    print("== Fetching prices + valuation ==")
    prices_df = build_prices_table(codes, args.start, args.end, pause=args.pause)
    prices_df.to_csv(out_dir / "prices.csv", index=False)
    print(f"Wrote {len(prices_df)} price rows to {out_dir / 'prices.csv'}")

    print("== Fetching fundamentals ==")
    fundamentals_df = build_fundamentals_table(codes, prices_df, args.start, args.end, pause=args.pause)
    fundamentals_df.to_csv(out_dir / "fundamentals.csv", index=False)
    print(f"Wrote {len(fundamentals_df)} fundamentals rows to {out_dir / 'fundamentals.csv'}")

    print("== Fetching risk-free rate & building FF5-style factors ==")
    rf_df = fetch_risk_free_rate(args.start, args.end)
    ff5_df = build_ff5_factors(prices_df, fundamentals_df, rf_df)
    ff5_df.to_csv(out_dir / "ff5_factors.csv", index=False)
    print(f"Wrote {len(ff5_df)} FF5 factor rows to {out_dir / 'ff5_factors.csv'}")

    print("\nDone. NOTE: data/raw/posts.csv (Guba posts) is NOT fetched by this script --")
    print("see the README section on scraping Guba separately (e.g. EastMoney_Crawler).")


if __name__ == "__main__":
    _cli()