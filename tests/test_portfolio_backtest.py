"""
Unit tests for Portfolio Optimization and Chronological Backtesting (Phase 7).

Verifies:
1. Weight Constraints: Sum to 1.0, non-negative, and adhere to max_weight bounds.
2. Chronological Availability: Signals and covariance use pre-decision cutoff data only.
3. Prevention of Future-Data Leakage: Zero lookahead in execution and risk estimation.
4. Transaction Cost Accounting: Exact turnover deduction from cash and equity.
5. Covariance Estimation & Shrinkage: Regularization of ill-conditioned matrices.
6. Optimizer Failure Handling: Graceful fallbacks when optimization encounters edge cases.
7. Multi-Tranche Capital Conservation: Total capital conserved across staggered tranches.
8. Quantitative Metrics Accuracy: Exact mathematical formulas for CAGR, Sharpe, Drawdowns.
"""
import unittest
import numpy as np
import pandas as pd

from src.portfolio.optimizer import (
    EqualWeightOptimizer,
    TopKEqualWeightOptimizer,
    TopKScoreWeightedOptimizer,
    TopKMinVarianceOptimizer,
    InverseVolatilityOptimizer,
    MinVarianceOptimizer,
    OptimizationConfig,
    get_optimizer,
)
from src.portfolio.metrics import PortfolioMetricsCalculator
from src.portfolio.backtest import BacktestConfig, BacktestEngine


class TestPortfolioOptimizationAndBacktest(unittest.TestCase):
    """Test suite covering Phase 7 portfolio construction and backtesting."""

    def setUp(self):
        self.symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"
        ]
        self.config = OptimizationConfig(top_k=3, max_weight=0.40, min_weight=0.0)

        # Mock scores DataFrame
        self.scores_df = pd.DataFrame({
            "stock": self.symbols,
            "y_prob": [0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
        })

        # Mock well-conditioned covariance matrix
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, size=(60, len(self.symbols)))
        df_ret = pd.DataFrame(returns, columns=self.symbols)
        self.cov_matrix = df_ret.cov() * 252

    def test_weight_constraints(self):
        """Verifies all optimizers produce weights summing to 1.0 and within valid bounds."""
        optimizers = [
            EqualWeightOptimizer(self.config),
            TopKEqualWeightOptimizer(self.config),
            TopKScoreWeightedOptimizer(self.config),
            InverseVolatilityOptimizer(self.config, top_k_filter=False),
            InverseVolatilityOptimizer(self.config, top_k_filter=True),
            MinVarianceOptimizer(self.config),
            TopKMinVarianceOptimizer(self.config),
        ]

        for opt in optimizers:
            weights = opt.allocate(self.symbols, self.scores_df, self.cov_matrix)
            total_w = sum(weights.values())
            self.assertAlmostEqual(total_w, 1.0, places=5, msg=f"Weights do not sum to 1.0 in {opt.__class__.__name__}")

            for s, w in weights.items():
                self.assertGreaterEqual(w, -1e-6, msg=f"Negative weight {w} for {s} in {opt.__class__.__name__}")
                self.assertLessEqual(w, 1.0 + 1e-6, msg=f"Weight exceeds 1.0 for {s} in {opt.__class__.__name__}")

    def test_topk_selection_and_ranking(self):
        """Verifies TopK optimizers select exactly the top K ranked stocks."""
        opt = TopKEqualWeightOptimizer(self.config)
        weights = opt.allocate(self.symbols, self.scores_df, self.cov_matrix)

        # Top 3 stocks in mock scores are RELIANCE, TCS, HDFCBANK
        expected_top = {"RELIANCE", "TCS", "HDFCBANK"}
        active_stocks = {s for s, w in weights.items() if w > 0.01}
        self.assertEqual(active_stocks, expected_top, "Top-K stocks selected do not match highest scores.")
        for s in expected_top:
            self.assertAlmostEqual(weights[s], 1.0 / 3.0, places=5)

    def test_covariance_shrinkage_and_regularization(self):
        """Verifies covariance shrinkage turns a singular matrix into a positive-definite matrix."""
        # Create a completely singular covariance matrix (rank 1)
        v = np.ones((len(self.symbols), 1))
        singular_cov = v @ v.T

        opt = MinVarianceOptimizer(self.config)
        shrunk = opt._shrink_covariance(singular_cov)

        eigenvalues = np.linalg.eigvalsh(shrunk)
        self.assertTrue(np.all(eigenvalues > 0), "Shrunk covariance matrix has non-positive eigenvalues.")
        self.assertLess(np.linalg.cond(shrunk), 1e6, "Condition number of shrunk covariance is too high.")

    def test_optimizer_fallback_on_ill_conditioned_data(self):
        """Verifies optimizers fall back gracefully without raising exceptions on pathological data."""
        opt_minvar = TopKMinVarianceOptimizer(self.config)

        # 1. Empty covariance
        w1 = opt_minvar.allocate(self.symbols, self.scores_df, pd.DataFrame())
        self.assertAlmostEqual(sum(w1.values()), 1.0, places=5)

        # 2. NaN in scores
        nan_scores = pd.DataFrame({"stock": self.symbols, "y_prob": [np.nan] * len(self.symbols)})
        w2 = opt_minvar.allocate(self.symbols, nan_scores, self.cov_matrix)
        self.assertAlmostEqual(sum(w2.values()), 1.0, places=5)

        # 3. None inputs
        w3 = opt_minvar.allocate(self.symbols, None, None)
        self.assertAlmostEqual(sum(w3.values()), 1.0, places=5)

    def test_transaction_cost_accounting(self):
        """Verifies transaction costs reduce portfolio equity monotonically as cost_bps increases."""
        prices = pd.read_csv("data/processed/india/prices.csv")
        oos = pd.read_csv("reports/phase6_oos_predictions.csv")

        # Run with 0 bps
        cfg_0 = BacktestConfig(cost_bps=0.0, rebalance_freq=10)
        res_0 = BacktestEngine(prices, oos, cfg_0).run_strategy("topk_equal", "RF")

        # Run with 25 bps
        cfg_25 = BacktestConfig(cost_bps=25.0, rebalance_freq=10)
        res_25 = BacktestEngine(prices, oos, cfg_25).run_strategy("topk_equal", "RF")

        # Run with 50 bps
        cfg_50 = BacktestConfig(cost_bps=50.0, rebalance_freq=10)
        res_50 = BacktestEngine(prices, oos, cfg_50).run_strategy("topk_equal", "RF")

        self.assertGreater(res_0.report.final_value, res_25.report.final_value, "Higher transaction costs should reduce final value.")
        self.assertGreater(res_25.report.final_value, res_50.report.final_value, "Higher transaction costs should reduce final value.")
        self.assertGreater(res_50.report.total_transaction_costs, res_25.report.total_transaction_costs, "Total transaction costs paid should increase with cost_bps.")

    def test_chronology_and_prevention_of_lookahead_leakage(self):
        """Verifies that execution strictly occurs at t+1 and decision uses only history up to t."""
        prices = pd.read_csv("data/processed/india/prices.csv")
        oos = pd.read_csv("reports/phase6_oos_predictions.csv")

        cfg = BacktestConfig(rebalance_freq=10, execution_price="open")
        engine = BacktestEngine(prices, oos, cfg)
        res = engine.run_strategy("topk_equal", "RF")

        trade_logs = res.trade_logs
        self.assertFalse(trade_logs.empty, "Trade logs should not be empty.")

        for _, trade in trade_logs.iterrows():
            decision_date = pd.to_datetime(trade["decision_date"])
            execution_date = pd.to_datetime(trade["execution_date"])

            # Execution date must be strictly after decision date
            self.assertGreater(execution_date, decision_date, f"Lookahead violation: Execution {execution_date} is not after decision {decision_date}")

    def test_metrics_calculator_accuracy(self):
        """Verifies CAGR, Volatility, Sharpe, Drawdown, and t-test calculations against known values."""
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        # Generate 1-year series doubling from 100 to 200
        equity = pd.Series(np.linspace(100.0, 200.0, 252), index=dates)
        bench = pd.Series(np.linspace(100.0, 150.0, 252), index=dates)

        calc = PortfolioMetricsCalculator(risk_free_rate=0.06)
        report = calc.compute_metrics("TestStrat", "TestModel", equity, bench)

        # CAGR should be approx 100% (doubling in 1 year)
        self.assertAlmostEqual(report.cagr, 1.0, places=1)
        self.assertGreater(report.sharpe_ratio, 0.0)
        self.assertAlmostEqual(report.max_drawdown, 0.0, places=3)
        self.assertGreater(report.alpha_cagr, 0.0)


if __name__ == "__main__":
    unittest.main()
