"""
Unit tests for Phase 6: Chronological Walk-Forward Validation & Prediction Outcome Tracking.

Verifies:
1. Mathematical temporal purge enforcement (last training label realization strictly < test start).
2. Preprocessing isolation (SimpleImputer and MinMaxScaler fitted strictly on training slices).
3. Metric calculations (ROC-AUC, PR-AUC, Balanced Accuracy, Confusion Matrix, Precision@3).
4. Outcome tracker pending status lifecycle (zero premature outcome calculation).
5. Outcome tracker resolution over full 10-trading-day horizon.
6. Cross-sectional ranking logic and handling of sparse universe data.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.chronological_validation import (
    ChronologicalFold,
    ChronologicalSplitter,
    compute_comprehensive_metrics,
)
from src.evaluation.outcome_tracker import (
    OutcomeStatus,
    PredictionOutcomeTracker,
    TrackedPrediction,
)


class TestChronologicalValidation(unittest.TestCase):
    """Tests for chronological walk-forward splitting and preprocessing isolation."""

    def setUp(self):
        # Create a synthetic trading calendar across 3 years (2021, 2022, 2023)
        dates_2021 = pd.date_range("2021-01-01", "2021-12-31", freq="B")
        dates_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="B")
        dates_2023 = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        self.all_dates = sorted(list(dates_2021) + list(dates_2022) + list(dates_2023))

    def test_chronological_splitter_no_leakage(self):
        """Verify that last training label realization strictly precedes test start."""
        splitter = ChronologicalSplitter(self.all_dates, forward_horizon=10)
        folds = splitter.create_annual_folds()

        self.assertEqual(len(folds), 2)  # Fold 1 (test 2022), Fold 2 (test 2023)

        for fold in folds:
            # 1. Purged train end must be strictly before test start
            self.assertLess(fold.train_end_purged, fold.test_start)

            # 2. Realization of forward 10-day label for the last training day
            #    MUST strictly precede test start (Zero-Leakage Guarantee)
            self.assertLess(fold.last_train_label_realization, fold.test_start)

            # 3. Exact 10-trading-day purge length
            date_to_idx = splitter.date_to_idx
            test_start_idx = date_to_idx[fold.test_start]
            purged_train_end_idx = date_to_idx[fold.train_end_purged]
            purge_gap = test_start_idx - purged_train_end_idx - 1
            self.assertEqual(purge_gap, 10)

    def test_preprocessing_isolation(self):
        """Verify scaler and imputer fit exclusively on training data without test leakage."""
        rng = np.random.default_rng(42)
        X_train = rng.uniform(10.0, 50.0, size=(100, 5))
        # Add NaNs in training data
        X_train[0, 0] = np.nan
        X_train[5, 2] = np.nan

        # Test data with extreme out-of-distribution values
        X_test = rng.uniform(100.0, 500.0, size=(20, 5))
        X_test[1, 0] = np.nan

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train_imp)

        # Store training parameters
        train_medians = imputer.statistics_.copy()
        train_min = scaler.data_min_.copy()
        train_max = scaler.data_max_.copy()

        # Transform test data (zero-refit)
        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        # Check training parameters are unaltered
        np.testing.assert_array_equal(imputer.statistics_, train_medians)
        np.testing.assert_array_equal(scaler.data_min_, train_min)
        np.testing.assert_array_equal(scaler.data_max_, train_max)

        # Confirm test imputation applied training medians
        self.assertEqual(X_test_imp[1, 0], train_medians[0])

    def test_metrics_calculation_correctness(self):
        """Verify accurate computation of ROC-AUC, PR-AUC, balanced accuracy, and Precision@3."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        # High probability for true positives
        y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.02])

        eval_df = pd.DataFrame({
            "date": [pd.Timestamp("2026-10-21")] * 10,
            "stock": [f"STOCK_{i}" for i in range(10)],
            "fwd_ret": [0.05, 0.04, 0.03, -0.01, -0.02, 0.0, -0.05, 0.01, -0.03, -0.04],
            "label": y_true,
        })

        train_info = {
            "start": pd.Timestamp("2026-01-01"),
            "end": pd.Timestamp("2026-06-30"),
            "rows": 100,
            "pos_count": 30,
        }
        test_info = {
            "start": pd.Timestamp("2026-07-01"),
            "end": pd.Timestamp("2026-10-21"),
            "rows": 10,
            "pos_count": 3,
        }

        res, oos_df = compute_comprehensive_metrics(
            y_true=y_true,
            y_prob=y_prob,
            df_eval=eval_df,
            fold_id=1,
            model_name="TEST_MODEL",
            feature_set="Price-Only",
            train_info=train_info,
            test_info=test_info,
        )

        # Perfect ranking: ROC-AUC = 1.0, Precision@3 = 1.0
        self.assertEqual(res.roc_auc, 1.0)
        self.assertEqual(res.pr_auc, 1.0)
        self.assertEqual(res.precision_at_3, 1.0)
        self.assertEqual(res.accuracy, 1.0)
        self.assertEqual(res.balanced_accuracy, 1.0)
        self.assertEqual(res.tp, 3)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.fn, 0)
        self.assertEqual(res.tn, 7)


class TestOutcomeTracker(unittest.TestCase):
    """Tests for prediction outcome tracking, pending lifecycle, and 10-day resolution."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_outcomes.csv"
        self.json_path = Path(self.temp_dir.name) / "test_outcomes.json"
        self.universe = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "BHARTIARTL", "KOTAKBANK", "HINDUNILVR"]
        self.tracker = PredictionOutcomeTracker(
            storage_path=self.csv_path,
            json_storage_path=self.json_path,
            eligible_universe=self.universe,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_prediction_initial_pending_state(self):
        """Verify new predictions initialize as PENDING with zero realized metrics."""
        rec = self.tracker.record_prediction(
            stock="RELIANCE",
            close_price=1200.0,
            predicted_probability=0.35,
            market_data_cutoff="2026-10-01",
            rank=1,
            is_recommended=True,
        )

        self.assertEqual(rec.status, OutcomeStatus.PENDING.value)
        self.assertEqual(rec.days_elapsed, 0)
        self.assertIsNone(rec.realized_return_10d)
        self.assertIsNone(rec.realized_label)
        self.assertIsNone(rec.prediction_correct)
        self.assertTrue(self.csv_path.exists())
        self.assertTrue(self.json_path.exists())

    def test_partial_horizon_remains_strictly_pending(self):
        """Verify prediction remains PENDING when fewer than 10 forward days exist (Zero Fabrication)."""
        self.tracker.record_prediction(
            stock="RELIANCE",
            close_price=1200.0,
            predicted_probability=0.35,
            market_data_cutoff="2026-10-01",
            rank=1,
            is_recommended=True,
        )

        # Simulate 6 forward trading days (fewer than 10)
        fwd_dates = pd.date_range("2026-10-02", periods=6, freq="B")
        prices_df = pd.DataFrame({
            "stock": ["RELIANCE"] * 6,
            "date": fwd_dates,
            "close": [1210.0, 1215.0, 1220.0, 1218.0, 1225.0, 1230.0],
        })

        stats = self.tracker.update_outcomes(prices_df)
        self.assertEqual(stats["pending"], 1)
        self.assertEqual(stats["completed"], 0)

        # Retrieve record
        rec = list(self.tracker.records.values())[0]
        self.assertEqual(rec.status, OutcomeStatus.PENDING.value)
        self.assertEqual(rec.days_elapsed, 6)
        self.assertIsNone(rec.realized_return_10d)
        self.assertIsNone(rec.realized_label)

    def test_full_horizon_resolves_and_labels_correctly(self):
        """Verify that exactly 10 forward trading days resolve return and top-30% label."""
        cutoff = "2026-09-01"
        self.tracker.record_prediction(
            stock="RELIANCE",
            close_price=1000.0,
            predicted_probability=0.45,
            market_data_cutoff=cutoff,
            rank=1,
            is_recommended=True,
            predicted_class=1,
        )

        # Construct 10 forward trading days across universe of 10 stocks
        all_dates = [pd.Timestamp(cutoff)] + list(pd.date_range("2026-09-02", periods=10, freq="B"))
        records = []
        for stock in self.universe:
            base_p = 1000.0
            # RELIANCE outperforms (+20%), others gain modest 2%
            ret = 0.20 if stock == "RELIANCE" else 0.02
            for d in all_dates:
                px = base_p if d == all_dates[0] else base_p * (1.0 + ret)
                records.append({"stock": stock, "date": d, "close": px})

        prices_df = pd.DataFrame(records)
        stats = self.tracker.update_outcomes(prices_df)

        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["pending"], 0)

        rec = list(self.tracker.records.values())[0]
        self.assertEqual(rec.status, OutcomeStatus.COMPLETED.value)
        self.assertEqual(rec.days_elapsed, 10)
        self.assertAlmostEqual(rec.realized_return_10d, 0.20, places=4)
        # RELIANCE has highest return in universe (top 30%), so realized_label must be 1
        self.assertEqual(rec.realized_label, 1)
        self.assertTrue(rec.prediction_correct)

    def test_sparse_universe_handling(self):
        """Verify sparse universe (< 5 stocks) marks INSUFFICIENT_DATA rather than guessing."""
        cutoff = "2026-09-01"
        self.tracker.record_prediction(
            stock="RELIANCE",
            close_price=1000.0,
            predicted_probability=0.45,
            market_data_cutoff=cutoff,
        )

        # Only 2 stocks present on cutoff date
        all_dates = [pd.Timestamp(cutoff)] + list(pd.date_range("2026-09-02", periods=10, freq="B"))
        records = []
        for stock in ["RELIANCE", "TCS"]:
            for d in all_dates:
                records.append({"stock": stock, "date": d, "close": 1000.0})

        prices_df = pd.DataFrame(records)
        stats = self.tracker.update_outcomes(prices_df)

        self.assertEqual(stats["insufficient"], 1)
        rec = list(self.tracker.records.values())[0]
        self.assertEqual(rec.status, OutcomeStatus.INSUFFICIENT_DATA.value)
        self.assertIn("sparse_cross_section_less_than_5_stocks", rec.data_quality_flags)


if __name__ == "__main__":
    unittest.main()
