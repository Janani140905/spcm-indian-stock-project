"""
SPCM: Sentiment and Price Combined Model (paper Sec III.D, Figure 3).

Two modules feed a fusion matrix consumed by a final classifier (KNN / RF / NN):
  1. Stock Price Module  -> price features with 1/5/10/30-day delay/lag processing
  2. Text Sentiment Module -> Sent, PV, AI, Homo (already built in src/features/*)

Label: top 30% of forward N-day return -> 1 ("valid bullish"), bottom 70% -> 0.
Recommendation: rank by predicted P(label=1), take top-K stocks.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from src.models.baseline_models import MODEL_REGISTRY, scale_to_unit_range
from src.utils.data_loader import TARGET_STOCKS, normalize_indian_symbol

PRICE_LAGS = (1, 5, 10, 30)
FORWARD_HORIZON = 10   # trading days ahead, used to construct labels (paper uses "future rise/fall")
TOP_PCT_LABEL = 0.30   # top 30% of forward returns -> label 1


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_price_features(prices: pd.DataFrame, lags=PRICE_LAGS) -> pd.DataFrame:
    """Stock Price Module: base price features + lagged (delayed) versions."""
    df = prices.sort_values(["stock", "date"]).copy()
    df["stock"] = df["stock"].apply(normalize_indian_symbol)
    df["range_co"] = df["close"] - df["open"]
    df["range_hl"] = df["high"] - df["low"]
    df["ret_1d"] = df.groupby("stock")["close"].pct_change()

    base_cols = ["open", "high", "low", "close", "volume", "turnover",
                 "range_co", "range_hl", "ret_1d", "liquidity", "pe", "pcf"]
    base_cols = [c for c in base_cols if c in df.columns]

    for lag in lags:
        for col in base_cols:
            df[f"{col}_lag{lag}"] = df.groupby("stock")[col].shift(lag)

    return df


def build_labels(prices: pd.DataFrame, horizon: int = FORWARD_HORIZON, top_pct: float = TOP_PCT_LABEL) -> pd.DataFrame:
    """
    Forward return over `horizon` trading days; cross-sectional (per date) ranking into
    top `top_pct` (label=1) vs rest (label=0), matching the paper's "top 30% / bottom 70%".
    """
    df = prices.sort_values(["stock", "date"]).copy()
    df["stock"] = df["stock"].apply(normalize_indian_symbol)
    df["fwd_ret"] = df.groupby("stock")["close"].shift(-horizon) / df["close"] - 1

    def _label_cross_section(g: pd.DataFrame) -> pd.Series:
        if g["fwd_ret"].notna().sum() < 5:
            return pd.Series(np.nan, index=g.index)
        cutoff = g["fwd_ret"].quantile(1 - top_pct)
        out = pd.Series(np.nan, index=g.index)
        valid = g["fwd_ret"].notna()
        out[valid] = (g.loc[valid, "fwd_ret"] >= cutoff).astype(float)
        return out

    df["label"] = df.groupby("date", group_keys=False).apply(_label_cross_section)
    return df[["stock", "date", "fwd_ret", "label"]]


def shift_sentiment_dates(sentiment_ind: pd.DataFrame, price_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Because the raw Indian news dataset lacks reliable intraday timestamps, news is lagged 
    to the next available trading day to prevent post-close information leakage.
    """
    if sentiment_ind is None or sentiment_ind.empty:
        return sentiment_ind
        
    s_df = sentiment_ind.copy()
    s_df['date'] = pd.to_datetime(s_df['date']).dt.normalize()
    p_df = price_feat[['stock', 'date']].drop_duplicates().sort_values(['stock', 'date'])
    p_df['date'] = pd.to_datetime(p_df['date']).dt.normalize()
    
    s_df = s_df.sort_values('date')
    p_df = p_df.sort_values('date')
    
    shifted_dfs = []
    for stock in s_df['stock'].unique():
        s_stock = s_df[s_df['stock'] == stock].copy()
        p_stock = p_df[p_df['stock'] == stock].copy()
        if p_stock.empty:
            continue
            
        # Add 1 day to strictly enforce "next trading day"
        s_stock['join_date'] = s_stock['date'] + pd.Timedelta(days=1)
        merged = pd.merge_asof(s_stock, p_stock, left_on='join_date', right_on='date', direction='forward', suffixes=('_orig', ''))
        
        # 'date' is now the next trading date. Keep 'date_orig' for debugging.
        shifted_dfs.append(merged)
        
    if not shifted_dfs:
        return pd.DataFrame()
        
    shifted_all = pd.concat(shifted_dfs, ignore_index=True)
    
    # Aggregate multiple calendar dates falling on the same next trading date (e.g., weekend -> Monday)
    grp = shifted_all.groupby(['stock', 'date'])
    agg_df = grp.agg({
        'headline_count': 'sum',
        'bullish_count': 'sum',
        'bearish_count': 'sum',
        'neutral_count': 'sum',
        'date_orig': 'min',
    }).reset_index()
    
    agg_df['news_available'] = (agg_df['headline_count'] > 0).astype(int)
    denom = agg_df['bullish_count'] + agg_df['bearish_count']
    agg_df['Sent'] = np.where(denom > 0, (agg_df['bullish_count'] - agg_df['bearish_count']) / denom, np.nan)
    agg_df['PV'] = np.where(denom > 0, np.log(denom + 1), np.nan)
    agg_df['AI'] = np.where(denom > 0, 1 - np.sqrt(1 - agg_df['Sent']**2), np.nan)
    
    return agg_df

def build_fusion_matrix(
    price_feat: pd.DataFrame,
    sentiment_ind: pd.DataFrame | None,
    homophily_ind: pd.DataFrame | None,
    labels: pd.DataFrame,
    control_feat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge price features + sentiment indicators (Sent, PV, AI) + Homo + labels + AR/CAR into one matrix."""
    m = price_feat.copy()
    m["stock"] = m["stock"].apply(normalize_indian_symbol)
    if sentiment_ind is not None and not sentiment_ind.empty:
        s_ind = sentiment_ind.copy()
        s_ind["stock"] = s_ind["stock"].apply(normalize_indian_symbol)

        cols_to_merge = ["stock", "date", "Sent", "PV", "AI"]
        for c in ["news_available", "headline_count"]:
            if c in s_ind.columns:
                cols_to_merge.append(c)

        m = m.merge(
            s_ind[cols_to_merge],
            on=["stock", "date"],
            how="left"
        )

        if "news_available" not in m.columns:
            m["news_available"] = m["Sent"].notna().astype(int)

            
    if homophily_ind is not None and not homophily_ind.empty and "Homo" in homophily_ind.columns:
        h_ind = homophily_ind.copy()
        h_ind["stock"] = h_ind["stock"].apply(normalize_indian_symbol)
        m = m.merge(h_ind[["stock", "date", "Homo"]], on=["stock", "date"], how="left")
        m["Homo"] = m["Homo"].fillna(0.0)


    l_df = labels.copy()
    l_df["stock"] = l_df["stock"].apply(normalize_indian_symbol)
    m = m.merge(l_df, on=["stock", "date"], how="left")

    if control_feat is not None and not control_feat.empty:
        control_feat = control_feat.copy()
        control_feat["stock"] = control_feat["stock"].apply(normalize_indian_symbol)

        # Exclude duplicate feature columns while retaining merge keys.
        cols_to_merge = [
            c for c in control_feat.columns
            if c not in m.columns or c in ["stock", "date"]
        ]
        m = m.merge(control_feat[cols_to_merge], on=["stock", "date"], how="left")

    l_df = labels.copy()
    l_df["stock"] = l_df["stock"].apply(normalize_indian_symbol)
    m = m.merge(l_df, on=["stock", "date"], how="left")
    return m


def get_feature_columns(fusion_df: pd.DataFrame, include_sentiment: bool = True) -> list[str]:
    exclude = {"stock", "date", "fwd_ret", "label", "news_available", "headline_count"}
    price_cols = [c for c in fusion_df.columns if c not in exclude and c not in ("Sent", "PV", "AI", "Homo")]
    sentiment_cols = [c for c in ["Sent", "PV", "AI", "Homo", "news_available", "headline_count"] if c in fusion_df.columns]
    return price_cols + (sentiment_cols if include_sentiment else [])


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

def train_spcm(
    fusion_df: pd.DataFrame,
    classifier: str = "nn",
    include_sentiment: bool = True,
    train_mask: np.ndarray | None = None,
):
    """
    Fit a classifier on `fusion_df` (optionally restricted to `train_mask` rows).
    Returns (fitted_model, scaler, imputer, feature_cols).
    """
    feature_cols = get_feature_columns(fusion_df, include_sentiment=include_sentiment)
    
    # Drop rows only where the target label is missing
    work = fusion_df.dropna(subset=["label"]).copy()
    if train_mask is not None:
        work = work[train_mask.reindex(work.index).fillna(False)]

    X = work[feature_cols].values.astype(float)
    y = work["label"].values.astype(int)

    # Impute missing values (e.g. Sent, PV, AI NaNs, and early lag NaNs)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_imputed)

    model = MODEL_REGISTRY[classifier]()
    model.fit(X_scaled, y)
    return model, scaler, imputer, feature_cols


def predict_proba(model, scaler: MinMaxScaler, imputer: SimpleImputer, feature_cols: list[str], fusion_df: pd.DataFrame) -> pd.DataFrame:
    work = fusion_df.copy()
    X = work[feature_cols].values.astype(float)
    
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    proba = model.predict_proba(X_scaled)[:, 1]
    work["proba"] = proba
    return work[["stock", "date", "proba"]]


def recommend_top_k(proba_df: pd.DataFrame, date, k: int = 50) -> pd.DataFrame:
    day = proba_df[proba_df["date"] == date].sort_values("proba", ascending=False)
    return day.head(k)


def compute_auc(fusion_df: pd.DataFrame, proba_df: pd.DataFrame) -> float:
    merged = fusion_df.merge(proba_df, on=["stock", "date"]).dropna(subset=["label", "proba"])
    if merged["label"].nunique() < 2:
        return float("nan")
    return roc_auc_score(merged["label"], merged["proba"])


# ---------------------------------------------------------------------------
# Sensitivity analysis (paper Sec IV.C, Figure 7)
# ---------------------------------------------------------------------------
def sensitivity_curve(model, scaler, feature_cols, fusion_df: pd.DataFrame, feature: str, n_points: int = 21):
    """
    Hold all features at their sample mean except `feature`, sweep `feature` across
    [-1, 1] (post-scaling range), and record the change in mean predicted P(label=1)
    relative to the feature's midpoint -- a proxy for the paper's "cumulative return
    change" marginal-effect plot (Figure 7).
    """
    work = fusion_df.dropna(subset=feature_cols).copy()
    X = work[feature_cols].values.astype(float)
    X_scaled, _ = scale_to_unit_range(X, scaler=scaler)

    means = X_scaled.mean(axis=0)
    feat_idx = feature_cols.index(feature)

    grid = np.linspace(-1, 1, n_points)
    base_row = means.copy()
    preds = []
    for val in grid:
        row = base_row.copy()
        row[feat_idx] = val
        p = model.predict_proba(row.reshape(1, -1))[0, 1]
        preds.append(p)
    preds = np.array(preds)
    change = preds - preds[n_points // 2]  # relative to midpoint (0), mirrors Fig. 7's y-axis
    return pd.DataFrame({feature: grid, "pred_change": change})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_all(data_dir: Path):
    prices = pd.read_csv(data_dir / "raw" / "prices.csv", parse_dates=["date"])
    sentiment_ind = pd.read_csv(data_dir / "processed" / "sentiment_indicators.csv", parse_dates=["date"])
    homophily_ind = pd.read_csv(data_dir / "processed" / "homophily.csv", parse_dates=["date"])
    return prices, sentiment_ind, homophily_ind


def _build_pipeline(data_dir: Path):
    prices, sentiment_ind, homophily_ind = _load_all(data_dir)
    price_feat = build_price_features(prices)
    labels = build_labels(prices)
    fusion = build_fusion_matrix(price_feat, sentiment_ind, homophily_ind, labels)
    return fusion


def _cli():
    from src.utils.data_loader import DATA_DIR

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--start", default="2018-01-01")
    p_train.add_argument("--end", default="2020-12-31")
    p_train.add_argument("--classifier", default="nn", choices=list(MODEL_REGISTRY.keys()))

    p_sens = sub.add_parser("sensitivity")
    p_sens.add_argument("--feature", required=True, choices=["PV", "AI", "Homo"])

    args = parser.parse_args()
    fusion = _build_pipeline(DATA_DIR)
    fusion = fusion[(fusion["date"] >= args.start) & (fusion["date"] <= args.end)] if hasattr(args, "start") else fusion

    if args.cmd == "train":
        model, scaler, feature_cols = train_spcm(fusion, classifier=args.classifier)
        proba = predict_proba(model, scaler, feature_cols, fusion)
        auc = compute_auc(fusion, proba)
        print(f"Trained SPCM ({args.classifier}) on {len(fusion)} rows. In-sample AUC: {auc:.4f}")

        out_dir = DATA_DIR / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        proba.to_csv(out_dir / "spcm_predictions.csv", index=False)
        print(f"Predictions saved to {out_dir / 'spcm_predictions.csv'}")

    elif args.cmd == "sensitivity":
        model, scaler, feature_cols = train_spcm(fusion, classifier="nn")
        curve = sensitivity_curve(model, scaler, feature_cols, fusion, args.feature)
        out_path = DATA_DIR / "processed" / f"sensitivity_{args.feature}.csv"
        curve.to_csv(out_path, index=False)
        print(f"Sensitivity curve for {args.feature} saved to {out_path}")


if __name__ == "__main__":
    _cli()
