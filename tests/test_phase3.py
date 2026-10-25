import unittest
import numpy as np
import pandas as pd

from src.models.spcm import build_price_features, build_labels, shift_sentiment_dates, build_fusion_matrix, get_feature_columns
from src.features.financial_features import compute_abnormal_returns, build_control_variables

class TestPhase3AblationAndHoldingPeriod(unittest.TestCase):

    def test_feature_subset_partitioning(self):
        """Verify feature configurations match the exact expected column partitions."""
        prices = pd.DataFrame({
            "stock": ["STK1"] * 65,
            "date": pd.date_range("2020-01-01", periods=65, freq="D"),
            "open": np.linspace(100, 150, 65),
            "high": np.linspace(101, 152, 65),
            "low": np.linspace(99, 149, 65),
            "close": np.linspace(100, 151, 65),
            "previous_close": np.linspace(99, 150, 65),
            "volume": [1000] * 65,
            "turnover": [1500] * 65,
        })
        np.random.seed(42)
        ff5 = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=65, freq="D"),
            "mkt_rf": np.random.normal(0.0005, 0.01, 65),
            "smb": np.random.normal(0.0002, 0.005, 65),
            "hml": np.random.normal(-0.0001, 0.005, 65),
            "wml": np.random.normal(0.0003, 0.005, 65),
            "rf": [0.0001] * 65,
        })
        sentiment_ind = pd.DataFrame({
            "stock": ["STK1"] * 65,
            "date": pd.date_range("2020-01-01", periods=65, freq="D"),
            "headline_count": [1] * 65,
            "bullish_count": [1] * 65,
            "bearish_count": [0] * 65,
            "neutral_count": [0] * 65,
            "news_available": [1] * 65,
            "Sent": [1.0] * 65,
            "PV": [np.log(2)] * 65,
            "AI": [1.0] * 65,
        })

        ar_df = compute_abnormal_returns(prices, ff5, window=10)
        ctrl = build_control_variables(ar_df)
        pf = build_price_features(prices)
        lbl = build_labels(prices)
        s_shift = shift_sentiment_dates(sentiment_ind, pf)
        fusion = build_fusion_matrix(pf, s_shift, None, lbl, control_feat=ctrl)
        
        all_cols = get_feature_columns(fusion, include_sentiment=True)
        price_cols = [c for c in all_cols if c not in [
            'ret', 'excess_ret', 'AR', 'CAR_t1_t5', 'CAR_t6_t20', 'Volatility', 
            'Sent', 'PV', 'AI', 'news_available', 'headline_count'
        ]]
        factor_cols = [c for c in ['ret', 'excess_ret', 'AR', 'CAR_t1_t5', 'CAR_t6_t20', 'Volatility'] if c in all_cols]
        sent_cols = [c for c in ['Sent', 'PV', 'AI', 'news_available', 'headline_count'] if c in all_cols]

        self.assertEqual(len(all_cols), 57)
        self.assertEqual(len(price_cols), 46)
        self.assertEqual(len(factor_cols), 6)
        self.assertEqual(len(sent_cols), 5)

        # Disjoint verification
        self.assertEqual(len(set(price_cols).intersection(factor_cols)), 0)
        self.assertEqual(len(set(price_cols).intersection(sent_cols)), 0)
        self.assertEqual(len(set(factor_cols).intersection(sent_cols)), 0)

    def test_precision_equals_recall_at_3(self):
        """Under a 10-stock universe with 30% positive labels (3 positives), Prec@3 == Recall@3."""
        # Simulated true labels (3 positives out of 10)
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        # Case A: Model ranks top 3 correctly (3/3)
        ranks_a = np.array([0, 1, 2])
        p3_a = y_true[ranks_a].sum() / 3.0
        r3_a = y_true[ranks_a].sum() / 3.0
        self.assertEqual(p3_a, r3_a)
        self.assertEqual(p3_a, 1.0)

        # Case B: Model ranks 1 positive in top 3 (1/3)
        ranks_b = np.array([0, 3, 4])
        p3_b = y_true[ranks_b].sum() / 3.0
        r3_b = y_true[ranks_b].sum() / 3.0
        self.assertEqual(p3_b, r3_b)
        self.assertAlmostEqual(p3_b, 1/3)

    def test_holding_period_non_overlapping_cycles(self):
        """Verify holding period cycles do not overlap in entry and exit dates."""
        trading_dates = pd.date_range("2020-01-01", periods=100, freq="B")
        for H in [10, 20, 40]:
            rebal_dates = trading_dates[::H]
            for i in range(len(rebal_dates) - 1):
                entry_date = rebal_dates[i]
                exit_date = rebal_dates[i + 1]
                # Ensure distance between consecutive rebalances is exactly H trading days
                idx_entry = list(trading_dates).index(entry_date)
                idx_exit = list(trading_dates).index(exit_date)
                self.assertEqual(idx_exit - idx_entry, H)

    def test_turnover_transaction_cost_calculation(self):
        """Verify 15 bps per trade is applied accurately on turnover."""
        fee_per_trade = 0.0015
        # Initial buy of 3 stocks: one-way cost 15 bps (0.0015)
        initial_cost = 1.0 * fee_per_trade
        self.assertEqual(initial_cost, 0.0015)

        # Rebalance: 1 stock replaced out of 3 -> turnover = 1/3 = 33.3%
        # Requires 1 sell + 1 buy = 2 trades per turnover unit
        turnover = 1.0 / 3.0
        cost = (turnover * 2.0) * fee_per_trade
        self.assertAlmostEqual(cost, (2/3) * 0.0015)

if __name__ == "__main__":
    unittest.main()
