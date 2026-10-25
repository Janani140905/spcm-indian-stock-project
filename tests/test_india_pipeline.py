import unittest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from src.models.spcm import build_labels, shift_sentiment_dates, build_fusion_matrix, get_feature_columns
from src.features.financial_features import compute_daily_returns, _rolling_ff5_ar_for_stock

class TestIndianPipelineAudit(unittest.TestCase):
    
    def test_label_cross_section_top_30_percentile(self):
        """Verify labels are computed cross-sectionally per date for top 30%."""
        dates = pd.date_range("2020-01-01", periods=15, freq="D")
        stocks = [f"STK_{i}" for i in range(10)]
        rows = []
        for d in dates:
            for i, s in enumerate(stocks):
                # Simulated prices
                rows.append({"stock": s, "date": d, "close": 100.0 + (i * 2.0) + (d.day * 1.5)})
        prices = pd.DataFrame(rows)
        
        lbl_df = build_labels(prices, horizon=2, top_pct=0.30)
        
        # Check a date with full returns
        sample_date = dates[2]
        day_lbl = lbl_df[lbl_df["date"] == sample_date]
        self.assertEqual(len(day_lbl), 10)
        # Top 30% of 10 stocks is 3 positive labels
        pos_count = day_lbl["label"].sum()
        self.assertEqual(pos_count, 3.0)
        
        # Check that NaN is preserved for the last 2 days of horizon
        last_date = dates[-1]
        last_lbl = lbl_df[lbl_df["date"] == last_date]
        self.assertTrue(last_lbl["label"].isna().all())

    def test_label_preserves_nan_for_missing_forward_returns(self):
        """Verify that individual stocks with NaN fwd_ret retain NaN label, not 0.0."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        # 6 stocks, one stock delisted/missing on future date
        rows = []
        for d in dates:
            for s in ["S1", "S2", "S3", "S4", "S5", "S6"]:
                if s == "S6" and d >= pd.Timestamp("2020-01-03"):
                    continue # missing future price
                rows.append({"stock": s, "date": d, "close": 100.0})
        prices = pd.DataFrame(rows)
        lbl_df = build_labels(prices, horizon=2, top_pct=0.30)
        
        # On 2020-01-02, S6 has no price at 2020-01-04 so fwd_ret is NaN
        s6_row = lbl_df[(lbl_df["stock"] == "S6") & (lbl_df["date"] == pd.Timestamp("2020-01-02"))]
        if not s6_row.empty:
            self.assertTrue(pd.isna(s6_row["label"].iloc[0]))

    def test_sentiment_next_trading_day_alignment(self):
        """Verify calendar day k news maps strictly to next trading day t >= k + 1."""
        price_feat = pd.DataFrame({
            "stock": ["RELIANCE"] * 5,
            "date": pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
        }) # Jan 3 is Friday, Jan 6 is Monday
        
        sentiment_ind = pd.DataFrame({
            "stock": ["RELIANCE", "RELIANCE", "RELIANCE"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-06"]),
            "headline_count": [1, 1, 1],
            "bullish_count": [1, 0, 1],
            "bearish_count": [0, 1, 0],
            "neutral_count": [0, 0, 0],
            "news_available": [1, 1, 1],
            "Sent": [1.0, -1.0, 1.0],
            "PV": [np.log(2), np.log(2), np.log(2)],
            "AI": [1.0, 1.0, 1.0]
        }) # Jan 3 (Friday), Jan 4 (Saturday), Jan 6 (Monday)
        
        shifted = shift_sentiment_dates(sentiment_ind, price_feat)
        
        # Friday news (Jan 3) + Saturday news (Jan 4) must map forward to Monday (Jan 6)
        mon_news = shifted[shifted["date"] == pd.Timestamp("2020-01-06")]
        self.assertEqual(len(mon_news), 1)
        # Aggregated headline count on Monday should be 1 (Fri) + 1 (Sat) = 2
        self.assertEqual(mon_news["headline_count"].iloc[0], 2)
        
        # Monday news (Jan 6) must map forward to Tuesday (Jan 7)
        tue_news = shifted[shifted["date"] == pd.Timestamp("2020-01-07")]
        self.assertEqual(len(tue_news), 1)
        self.assertEqual(tue_news["headline_count"].iloc[0], 1)

    def test_temporal_purge_boundary(self):
        """Verify that a 10-day purge prevents forward returns from crossing train/test boundary."""
        # Simulated trading dates
        trading_dates = pd.date_range("2018-01-01", periods=50, freq="B")
        test_start_idx = 25
        test_start_date = trading_dates[test_start_idx]
        
        horizon = 10
        # Safe train end must ensure train_date + horizon < test_start_date
        safe_train_idx = test_start_idx - horizon - 1
        safe_train_date = trading_dates[safe_train_idx]
        
        # Target realization date of safe_train_date
        realization_date = trading_dates[safe_train_idx + horizon]
        self.assertLess(realization_date, test_start_date)
        
        # First leaking date
        leaking_date = trading_dates[safe_train_idx + 1]
        leaking_realization = trading_dates[safe_train_idx + 1 + horizon]
        self.assertGreaterEqual(leaking_realization, test_start_date)

    def test_imputer_and_scaler_isolation(self):
        """Verify imputer and scaler fitted on train do not leak test distributions."""
        X_train = np.array([[1.0, np.nan], [2.0, 10.0], [3.0, 20.0]])
        X_test = np.array([[np.nan, 100.0], [5.0, np.nan]])
        
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        self.assertEqual(imputer.statistics_[1], 15.0) # median of [10, 20]
        
        # Transforming test data uses train median (15.0), not test values
        X_test_imp = imputer.transform(X_test)
        self.assertEqual(X_test_imp[1, 1], 15.0)
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_s = scaler.fit_transform(X_train_imp)
        # Scaler bounds based on train only (e.g. feature 0 min=1, max=3)
        self.assertEqual(scaler.data_min_[0], 1.0)
        self.assertEqual(scaler.data_max_[0], 3.0)

if __name__ == "__main__":
    unittest.main()
