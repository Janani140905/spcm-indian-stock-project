"""
SPCM Live Prediction Outcome Tracking Engine (Phase 6).

Implements rigorous, zero-fabrication prediction tracking over 10-trading-day horizons:
1. Records symbol, prediction timestamp, model version, market cutoff date, predicted probability,
   cross-sectional rank, recommendation flag, and data-quality metadata.
2. Keeps outcomes strictly PENDING until the complete 10-trading-day forward window has closed.
3. Automatically evaluates actual outcomes using the verified project label definition:
   fwd_ret = (close_{t+10} - close_t) / close_t, with cross-sectional top 30% cutoff (label=1).
4. Employs file-based persistence (CSV / JSON) so PostgreSQL is not required for testing.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.inference.spcm_inference_pipeline import SPCMInferenceResult
from src.utils.fetch_indian_prices import STOCK_UNIVERSE
from src.utils.indian_market_data import extract_stock_code, normalize_symbol
from src.utils.indian_news_fetcher import IST_TIMEZONE

logger = logging.getLogger("outcome_tracker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_STORAGE_CSV = "reports/prediction_outcomes.csv"
DEFAULT_STORAGE_JSON = "reports/prediction_outcomes.json"


class OutcomeStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class TrackedPrediction:
    """Represents a recorded model prediction and its pending/completed lifecycle."""
    prediction_id: str
    stock: str
    symbol: str
    prediction_timestamp: str
    market_data_cutoff: str
    close_price_at_prediction: float
    predicted_probability: float
    predicted_class: int
    rank: int
    is_recommended: bool
    model_version: str
    target_definition: str = "top_30pct_fwd_10d_ret"
    horizon_trading_days: int = 10
    status: str = OutcomeStatus.PENDING.value
    days_elapsed: int = 0
    maturity_date: Optional[str] = None
    realized_price_t10: Optional[float] = None
    realized_return_10d: Optional[float] = None
    universe_cutoff_return: Optional[float] = None
    realized_label: Optional[int] = None
    prediction_correct: Optional[bool] = None
    resolved_at: Optional[str] = None
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class PredictionOutcomeTracker:
    """
    Tracks real and historical predictions and resolves their 10-trading-day outcomes.
    """

    def __init__(
        self,
        storage_path: str | Path = DEFAULT_STORAGE_CSV,
        json_storage_path: str | Path = DEFAULT_STORAGE_JSON,
        eligible_universe: Sequence[str] | None = None,
    ):
        self.storage_path = Path(storage_path)
        self.json_storage_path = Path(json_storage_path)
        self.eligible_universe = [extract_stock_code(s) for s in (eligible_universe or STOCK_UNIVERSE)]
        self.records: dict[str, TrackedPrediction] = {}
        self.load_records()

    def record_prediction(
        self,
        stock: str,
        close_price: float,
        predicted_probability: float,
        market_data_cutoff: str | pd.Timestamp,
        rank: int = 1,
        is_recommended: bool = False,
        predicted_class: int | None = None,
        model_version: str = "spcm_nn_price_and_sentiment",
        prediction_timestamp: str | None = None,
        data_quality_flags: list[str] | None = None,
    ) -> TrackedPrediction:
        """
        Records a new prediction in PENDING status.
        """
        clean_stock = extract_stock_code(stock)
        symbol = f"{clean_stock}.NS"
        cutoff_str = pd.to_datetime(market_data_cutoff).strftime("%Y-%m-%d")
        now_ts = prediction_timestamp or datetime.now(tz=IST_TIMEZONE).isoformat()

        if predicted_class is None:
            predicted_class = 1 if is_recommended or predicted_probability >= 0.5 else 0

        # Unique deterministic or sequential ID
        pred_id = f"pred_{cutoff_str}_{clean_stock}_{int(predicted_probability * 10000):04d}"

        record = TrackedPrediction(
            prediction_id=pred_id,
            stock=clean_stock,
            symbol=symbol,
            prediction_timestamp=now_ts,
            market_data_cutoff=cutoff_str,
            close_price_at_prediction=round(float(close_price), 2),
            predicted_probability=round(float(predicted_probability), 4),
            predicted_class=int(predicted_class),
            rank=int(rank),
            is_recommended=bool(is_recommended),
            model_version=model_version,
            data_quality_flags=data_quality_flags or [],
        )

        self.records[pred_id] = record
        self.save_records()
        return record

    def record_inference_result(self, result: SPCMInferenceResult) -> list[TrackedPrediction]:
        """
        Records all stock predictions from an SPCMInferenceResult bundle.
        """
        recorded = []
        model_name = result.model_metadata.get("model_name", "spcm_nn_price_and_sentiment")

        for p in result.predictions:
            flags = []
            if p.news_count == 0 or p.sentiment_label == "no_news":
                flags.append("no_news_imputed_sentiment")
            if p.sentiment_label != "no_news":
                flags.append(f"sentiment_{p.sentiment_label}")

            rec = self.record_prediction(
                stock=p.stock,
                close_price=p.close_price,
                predicted_probability=p.probability,
                market_data_cutoff=p.trading_date,
                rank=p.rank,
                is_recommended=p.is_recommended,
                predicted_class=1 if p.is_recommended else 0,
                model_version=model_name,
                prediction_timestamp=result.inference_timestamp.isoformat(),
                data_quality_flags=flags,
            )
            recorded.append(rec)

        return recorded

    def update_outcomes(self, prices_df: pd.DataFrame) -> dict[str, int]:
        """
        Inspects all PENDING prediction records and evaluates actual 10-day outcomes
        if and only if at least 10 subsequent trading days exist in prices_df.
        """
        prices = prices_df.copy()
        prices["stock"] = prices["stock"].apply(extract_stock_code)
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

        stats = {"pending": 0, "completed": 0, "insufficient": 0, "unchanged": 0}

        # Cache trading calendar
        all_dates = sorted(prices["date"].unique())
        date_series = pd.Series(all_dates)

        for pred_id, rec in self.records.items():
            if rec.status == OutcomeStatus.COMPLETED.value:
                stats["completed"] += 1
                continue

            cutoff_date = pd.to_datetime(rec.market_data_cutoff).normalize()

            # Find subsequent trading dates for this stock
            stock_prices = prices[prices["stock"] == rec.stock].sort_values("date").reset_index(drop=True)
            fwd_prices = stock_prices[stock_prices["date"] > cutoff_date].reset_index(drop=True)

            days_available = len(fwd_prices)
            rec.days_elapsed = days_available

            if days_available < rec.horizon_trading_days:
                # Horizon not yet reached - keep strictly PENDING
                rec.status = OutcomeStatus.PENDING.value
                stats["pending"] += 1
                continue

            # 10 trading sessions have closed - evaluate outcome
            t10_row = fwd_prices.iloc[rec.horizon_trading_days - 1]
            maturity_date_str = t10_row["date"].strftime("%Y-%m-%d")
            p_t10 = float(t10_row["close"])
            p_t = float(rec.close_price_at_prediction)

            if p_t <= 0:
                rec.status = OutcomeStatus.INSUFFICIENT_DATA.value
                stats["insufficient"] += 1
                continue

            ret_10d = (p_t10 - p_t) / p_t
            rec.realized_price_t10 = round(p_t10, 2)
            rec.realized_return_10d = round(ret_10d, 4)
            rec.maturity_date = maturity_date_str

            # Calculate cross-sectional top 30% label on date t across universe
            universe_prices_t = prices[prices["date"] == cutoff_date]
            universe_prices_t10 = prices[prices["date"] == t10_row["date"]]

            m = universe_prices_t.merge(universe_prices_t10, on="stock", suffixes=("_t0", "_t10"))
            m = m[m["stock"].isin(self.eligible_universe)]

            if len(m) >= 5:
                m["ret"] = (m["close_t10"] - m["close_t0"]) / m["close_t0"]
                cutoff_ret = float(m["ret"].quantile(0.70))
                realized_label = 1 if ret_10d >= cutoff_ret else 0
                rec.universe_cutoff_return = round(cutoff_ret, 4)
                rec.realized_label = realized_label
                rec.prediction_correct = bool(rec.predicted_class == realized_label)
                rec.status = OutcomeStatus.COMPLETED.value
                rec.resolved_at = datetime.now(tz=IST_TIMEZONE).isoformat()
                stats["completed"] += 1
            else:
                # If universe is too sparse on that date, record return but flag universe limitation
                rec.status = OutcomeStatus.INSUFFICIENT_DATA.value
                rec.data_quality_flags.append("sparse_cross_section_less_than_5_stocks")
                stats["insufficient"] += 1

        self.save_records()
        return stats

    def get_summary(self) -> dict[str, Any]:
        """Returns structured tracking summary metrics."""
        records = list(self.records.values())
        total = len(records)
        pending = [r for r in records if r.status == OutcomeStatus.PENDING.value]
        completed = [r for r in records if r.status == OutcomeStatus.COMPLETED.value]
        insufficient = [r for r in records if r.status == OutcomeStatus.INSUFFICIENT_DATA.value]

        acc = None
        prec = None
        top3_ret_mean = None
        other_ret_mean = None

        if completed:
            correct = sum(1 for r in completed if r.prediction_correct is True)
            acc = round(correct / len(completed), 4)

            pos_preds = [r for r in completed if r.predicted_class == 1]
            if pos_preds:
                pos_correct = sum(1 for r in pos_preds if r.realized_label == 1)
                prec = round(pos_correct / len(pos_preds), 4)

            rec_stocks = [r.realized_return_10d for r in completed if r.is_recommended and r.realized_return_10d is not None]
            other_stocks = [r.realized_return_10d for r in completed if not r.is_recommended and r.realized_return_10d is not None]

            if rec_stocks:
                top3_ret_mean = round(float(np.mean(rec_stocks)), 4)
            if other_stocks:
                other_ret_mean = round(float(np.mean(other_stocks)), 4)

        return {
            "total_tracked_predictions": total,
            "pending_count": len(pending),
            "completed_count": len(completed),
            "insufficient_data_count": len(insufficient),
            "accuracy_on_completed": acc,
            "precision_on_completed": prec,
            "mean_10d_return_recommended": top3_ret_mean,
            "mean_10d_return_non_recommended": other_ret_mean,
            "target_definition": "top_30pct_fwd_10d_ret",
            "horizon_trading_days": 10,
        }

    def load_records(self) -> None:
        """Loads records from JSON if available, else CSV."""
        if self.json_storage_path.exists():
            try:
                with open(self.json_storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        rec = TrackedPrediction(**item)
                        self.records[rec.prediction_id] = rec
                return
            except Exception as e:
                logger.warning(f"Failed to load JSON tracking records: {e}")

        if self.storage_path.exists():
            try:
                df = pd.read_csv(self.storage_path)
                for _, row in df.iterrows():
                    d = row.to_dict()
                    # Clean NaNs
                    d_clean = {k: (None if pd.isna(v) else v) for k, v in d.items()}
                    if isinstance(d_clean.get("data_quality_flags"), str):
                        try:
                            d_clean["data_quality_flags"] = json.loads(d_clean["data_quality_flags"])
                        except Exception:
                            d_clean["data_quality_flags"] = []
                    rec = TrackedPrediction(**d_clean)
                    self.records[rec.prediction_id] = rec
            except Exception as e:
                logger.warning(f"Failed to load CSV tracking records: {e}")

    def save_records(self) -> None:
        """Persists records to both CSV and JSON."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_storage_path.parent.mkdir(parents=True, exist_ok=True)

        recs = [r.to_dict() for r in self.records.values()]
        if not recs:
            return

        # Save JSON
        with open(self.json_storage_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2)

        # Save CSV
        df = pd.DataFrame(recs)
        if "data_quality_flags" in df.columns:
            df["data_quality_flags"] = df["data_quality_flags"].apply(json.dumps)
        df.to_csv(self.storage_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 6: SPCM Prediction Outcome Tracking.")
    parser.add_argument("--status", action="store_true", help="Display status summary of tracked predictions")
    parser.add_argument("--update", action="store_true", help="Update outcomes using historical/current price data")
    parser.add_argument("--prices", default="data/processed/india/prices.csv", help="Path to price CSV for updating")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to display")
    args = parser.parse_args()

    tracker = PredictionOutcomeTracker()

    if args.update:
        print(f"Updating prediction outcomes using '{args.prices}'...")
        if Path(args.prices).exists():
            prices_df = pd.read_csv(args.prices)
            stats = tracker.update_outcomes(prices_df)
            print(f"Update results: {stats}")
        else:
            print(f"Error: Price file '{args.prices}' does not exist.")

    summary = tracker.get_summary()
    print("=" * 80)
    print(" SPCM PREDICTION OUTCOME TRACKING SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")
    print("=" * 80)

    records = list(tracker.records.values())
    if records:
        print(f"\nRecent Tracked Predictions (Displaying {min(len(records), args.limit)} of {len(records)}):")
        df = pd.DataFrame([r.to_dict() for r in records[:args.limit]])
        display_cols = ["prediction_id", "stock", "market_data_cutoff", "predicted_probability", "status", "days_elapsed", "realized_return_10d", "realized_label", "prediction_correct"]
        avail_cols = [c for c in display_cols if c in df.columns]
        print(df[avail_cols].to_string(index=False))
    else:
        print("\nNo predictions tracked yet. Use 'tracker.record_prediction()' or inference pipeline.")


if __name__ == "__main__":
    main()
