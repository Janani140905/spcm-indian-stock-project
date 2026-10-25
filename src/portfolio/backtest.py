"""
Walk-Forward Chronological Portfolio Backtest Engine for Indian Equities (Phase 7).

Features:
1. Strict Point-in-Time Chronology:
   - Decision cutoff at date t close.
   - Covariance estimated strictly using history up to date t (no lookahead).
   - Rebalance executed at date t+1 Open (or Close), modeling realistic operational delay.
2. Friction & Transaction Costs:
   - Configurable one-way transaction cost basis points (default 15 bps = 0.15%).
   - Exact physical share tracking and cash deduction on rebalance.
3. Overlapping / Multi-Tranche Support:
   - Periodic Cycle Mode: Non-overlapping rebalancing every N trading days (default 10 days).
   - Staggered Overlapping Mode: 10 concurrent subportfolios (each 10% capital) rebalanced on
     rotating days to completely eliminate start-date timing luck and smooth capital deployment.
4. Comprehensive reporting integrating institutional metrics from PortfolioMetricsCalculator.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.portfolio.metrics import PortfolioMetricsCalculator, PortfolioPerformanceReport
from src.portfolio.optimizer import (
    BasePortfolioOptimizer,
    OptimizationConfig,
    get_optimizer,
)

logger = logging.getLogger("portfolio_backtest")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class BacktestConfig:
    """Configurable execution parameters for portfolio backtesting."""
    initial_capital: float = 1_000_000.0  # INR
    rebalance_freq: int = 10              # Trading sessions (matches 10-day forward horizon)
    execution_price: str = "open"         # 'open' or 'close' on day t+1
    cost_bps: float = 15.0                # 15 basis points = 0.15% per trade turnover
    risk_free_rate: float = 0.06          # 6.0% annual Indian T-bill rate
    lookback_cov: int = 60                # Trading days for historical covariance matrix
    top_k: int = 3                        # Number of top stocks for model strategies
    max_weight: float = 0.40              # Diversification constraint: max weight per stock
    staggered_tranches: bool = False      # If True, runs 10 overlapping subportfolios


@dataclass
class BacktestResult:
    """Encapsulates full backtest simulation outputs."""
    strategy_name: str
    model_name: str
    config: BacktestConfig
    equity_curve: pd.DataFrame
    trade_logs: pd.DataFrame
    daily_weights: pd.DataFrame
    report: PortfolioPerformanceReport

    def to_summary_dict(self) -> Dict[str, Any]:
        d = self.report.to_dict()
        d["rebalance_freq"] = self.config.rebalance_freq
        d["cost_bps"] = self.config.cost_bps
        d["execution_price"] = self.config.execution_price
        return d


class BacktestEngine:
    """
    Executes walk-forward chronological portfolio simulations.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        oos_predictions_df: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Parameters:
            prices_df: DataFrame with ['stock', 'date', 'open', 'high', 'low', 'close', 'volume'].
            oos_predictions_df: DataFrame from Phase 6 with ['fold', 'model', 'feature_set', 'date', 'stock', 'y_prob'].
            config: Backtest configuration options.
        """
        self.config = config or BacktestConfig()
        self.prices = prices_df.copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"]).dt.normalize()
        self.prices = self.prices.sort_values(["stock", "date"]).reset_index(drop=True)

        self.oos = oos_predictions_df.copy()
        self.oos["date"] = pd.to_datetime(self.oos["date"]).dt.normalize()

        # Build pivot tables for vector access
        self.piv_close = self.prices.pivot(index="date", columns="stock", values="close").ffill()
        self.piv_open = self.prices.pivot(index="date", columns="stock", values="open").ffill()
        self.trading_dates = sorted(self.piv_close.index)
        self.date_to_idx = {d: i for i, d in enumerate(self.trading_dates)}
        self.symbols = sorted(list(self.piv_close.columns))

        # Metrics calculator
        self.metrics_calc = PortfolioMetricsCalculator(risk_free_rate=self.config.risk_free_rate)

    def run_strategy(
        self,
        strategy_name: str,
        model_name: str = "RF",
        feature_set: str = "Price + Sentiment",
        benchmark_equity_curve: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Runs a backtest for the specified strategy and model combination.
        """
        if self.config.staggered_tranches:
            return self._run_staggered_strategy(strategy_name, model_name, feature_set, benchmark_equity_curve)
        else:
            return self._run_cycle_strategy(strategy_name, model_name, feature_set, benchmark_equity_curve)

    def _run_cycle_strategy(
        self,
        strategy_name: str,
        model_name: str,
        feature_set: str,
        benchmark_equity_curve: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Executes non-overlapping periodic cycle rebalancing."""
        opt_cfg = OptimizationConfig(
            top_k=self.config.top_k,
            max_weight=self.config.max_weight,
        )
        optimizer = get_optimizer(strategy_name, opt_cfg)

        # Filter OOS predictions for this model and feature set
        model_oos = self.oos[
            (self.oos["model"] == model_name) & (self.oos["feature_set"] == feature_set)
        ]
        oos_dates = sorted(model_oos["date"].unique())
        if not oos_dates:
            raise ValueError(f"No OOS predictions found for model '{model_name}' and feature set '{feature_set}'.")

        # Select decision dates every rebalance_freq days
        rebal_freq = self.config.rebalance_freq
        decision_dates = oos_dates[::rebal_freq]

        initial_cap = self.config.initial_capital
        cash = initial_cap
        positions: Dict[str, float] = {s: 0.0 for s in self.symbols}
        current_weights: Dict[str, float] = {s: 0.0 for s in self.symbols}

        start_decision_date = decision_dates[0]
        start_idx = self.date_to_idx[start_decision_date]
        exec_start_idx = start_idx + 1  # Next day execution
        last_date = oos_dates[-1]
        last_idx = self.date_to_idx[last_date]

        decision_set = set(decision_dates)
        daily_records: List[Dict[str, Any]] = []
        trade_records: List[Dict[str, Any]] = []
        weights_records: List[Dict[str, Any]] = []

        # Iterate chronologically day by day
        for idx in range(exec_start_idx, last_idx + 1):
            curr_date = self.trading_dates[idx]
            prev_date = self.trading_dates[idx - 1]

            # Check if prev_date was a decision cutoff date
            if prev_date in decision_set:
                # 1. Mark portfolio value at execution price
                if self.config.execution_price == "open":
                    exec_prices = {s: float(self.piv_open.loc[curr_date, s]) for s in self.symbols}
                else:
                    exec_prices = {s: float(self.piv_close.loc[curr_date, s]) for s in self.symbols}

                pre_rebal_val = cash + sum(positions[s] * exec_prices[s] for s in self.symbols)

                # 2. Estimate Covariance strictly using returns prior to decision date
                cov_start_idx = max(0, idx - 1 - self.config.lookback_cov)
                hist_returns = self.piv_close.iloc[cov_start_idx : idx].pct_change().dropna()
                cov_matrix = hist_returns.cov() * 252

                # 3. Get decision day model scores
                scores_df = model_oos[model_oos["date"] == prev_date][["stock", "y_prob"]]

                # 4. Optimize target weights
                target_weights = optimizer.allocate(
                    symbols=self.symbols,
                    scores_df=scores_df,
                    cov_matrix=cov_matrix,
                    current_weights=current_weights,
                )

                # 5. Execute trades and deduct transaction costs
                target_dollars = {s: target_weights.get(s, 0.0) * pre_rebal_val for s in self.symbols}
                current_dollars = {s: positions[s] * exec_prices[s] for s in self.symbols}
                trade_dollars = {s: target_dollars[s] - current_dollars[s] for s in self.symbols}

                turnover = sum(abs(trade_dollars[s]) for s in self.symbols)
                trade_cost = turnover * (self.config.cost_bps / 10000.0)

                # Post-cost net equity to allocate
                net_alloc_val = pre_rebal_val - trade_cost
                new_positions: Dict[str, float] = {}
                for s in self.symbols:
                    s_dollars = target_weights.get(s, 0.0) * net_alloc_val
                    new_positions[s] = float(s_dollars / exec_prices[s]) if exec_prices[s] > 0 else 0.0

                positions = new_positions
                cash = 0.0
                current_weights = target_weights

                trade_records.append({
                    "decision_date": str(prev_date.date()),
                    "execution_date": str(curr_date.date()),
                    "pre_val": round(pre_rebal_val, 2),
                    "turnover": round(turnover, 2),
                    "cost": round(trade_cost, 2),
                    "weights": target_weights,
                })

            # End-of-day mark-to-market at CLOSE
            close_prices = {s: float(self.piv_close.loc[curr_date, s]) for s in self.symbols}
            eod_asset_val = sum(positions[s] * close_prices[s] for s in self.symbols)
            eod_val = cash + eod_asset_val

            # Record daily weights
            w_dict = {"date": curr_date}
            for s in self.symbols:
                w_dict[s] = float((positions[s] * close_prices[s]) / eod_val) if eod_val > 0 else 0.0
            weights_records.append(w_dict)

            daily_records.append({
                "date": curr_date,
                "portfolio_value": round(eod_val, 2),
                "cash": round(cash, 2),
            })

        df_daily = pd.DataFrame(daily_records).set_index("date")
        df_daily["daily_return"] = df_daily["portfolio_value"].pct_change().fillna(0.0)
        cummax = df_daily["portfolio_value"].cummax()
        df_daily["drawdown"] = (df_daily["portfolio_value"] - cummax) / cummax

        df_trades = pd.DataFrame(trade_records)
        df_weights = pd.DataFrame(weights_records)

        # Benchmark comparison
        if benchmark_equity_curve is None:
            # If no benchmark supplied, benchmark is EqualWeight
            if strategy_name.lower() in ("equal_weight", "benchmark_equal_weight", "ew"):
                benchmark_equity_curve = df_daily["portfolio_value"]
            else:
                ew_res = self._run_cycle_strategy("equal_weight", model_name, feature_set)
                benchmark_equity_curve = ew_res.equity_curve.set_index("date")["portfolio_value"]

        # Compute institutional report
        report = self.metrics_calc.compute_metrics(
            strategy_name=strategy_name,
            model_name=model_name,
            equity_series=df_daily["portfolio_value"],
            benchmark_series=benchmark_equity_curve,
            trade_logs=df_trades,
            weights_df=df_weights,
            benchmark_name="EqualWeight",
        )

        return BacktestResult(
            strategy_name=strategy_name,
            model_name=model_name,
            config=self.config,
            equity_curve=df_daily.reset_index(),
            trade_logs=df_trades,
            daily_weights=df_weights,
            report=report,
        )

    def _run_staggered_strategy(
        self,
        strategy_name: str,
        model_name: str,
        feature_set: str,
        benchmark_equity_curve: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Executes multi-tranche staggered backtest with N = rebalance_freq subportfolios.
        Each tranche is allocated 1/N of total capital and rebalances every N days,
        staggered by 1 day. This eliminates start-date timing luck and models
        continuous 10-day holding capital deployment.
        """
        rebal_freq = self.config.rebalance_freq
        sub_cap = self.config.initial_capital / rebal_freq

        # Filter OOS dates
        model_oos = self.oos[
            (self.oos["model"] == model_name) & (self.oos["feature_set"] == feature_set)
        ]
        oos_dates = sorted(model_oos["date"].unique())
        sub_config = BacktestConfig(
            initial_capital=sub_cap,
            rebalance_freq=rebal_freq,
            execution_price=self.config.execution_price,
            cost_bps=self.config.cost_bps,
            risk_free_rate=self.config.risk_free_rate,
            lookback_cov=self.config.lookback_cov,
            top_k=self.config.top_k,
            max_weight=self.config.max_weight,
            staggered_tranches=False,
        )

        sub_engine = BacktestEngine(self.prices, self.oos, sub_config)
        tranche_equity_curves: List[pd.Series] = []
        all_trade_records: List[pd.DataFrame] = []

        # Run rebalance cycles for each tranche offset (0, 1, ..., rebal_freq - 1)
        for offset in range(rebal_freq):
            tranche_dates = oos_dates[offset::rebal_freq]
            if len(tranche_dates) < 3:
                continue

            # Run custom cycle with this tranche's rebalance dates
            sub_res = self._run_custom_cycle(
                optimizer_strat=strategy_name,
                model_name=model_name,
                feature_set=feature_set,
                decision_dates=tranche_dates,
                config=sub_config,
            )
            tranche_equity_curves.append(sub_res.equity_curve.set_index("date")["portfolio_value"])
            if not sub_res.trade_logs.empty:
                all_trade_records.append(sub_res.trade_logs)

        # Aggregate tranches by date
        df_combined = pd.concat(tranche_equity_curves, axis=1).dropna()
        combined_val = df_combined.sum(axis=1)

        df_daily = pd.DataFrame({"date": combined_val.index, "portfolio_value": combined_val.values})
        df_daily["daily_return"] = df_daily["portfolio_value"].pct_change().fillna(0.0)
        cummax = df_daily["portfolio_value"].cummax()
        df_daily["drawdown"] = (df_daily["portfolio_value"] - cummax) / cummax

        combined_trades = pd.concat(all_trade_records, ignore_index=True) if all_trade_records else pd.DataFrame()

        if benchmark_equity_curve is None or benchmark_equity_curve.empty:
            if strategy_name.lower() in ("equal_weight", "benchmark_equal_weight", "ew"):
                benchmark_equity_curve = df_daily.set_index("date")["portfolio_value"]
            else:
                bench_config = BacktestConfig(
                    initial_capital=self.config.initial_capital,
                    rebalance_freq=rebal_freq,
                    staggered_tranches=True,
                )
                bench_engine = BacktestEngine(self.prices, self.oos, bench_config)
                # Pass df_daily portfolio value as dummy benchmark to avoid recursion
                bench_res = bench_engine._run_staggered_strategy(
                    "equal_weight", model_name, feature_set, benchmark_equity_curve=df_daily.set_index("date")["portfolio_value"]
                )
                benchmark_equity_curve = bench_res.equity_curve.set_index("date")["portfolio_value"]

        report = self.metrics_calc.compute_metrics(
            strategy_name=f"{strategy_name}_Staggered",
            model_name=model_name,
            equity_series=df_daily.set_index("date")["portfolio_value"],
            benchmark_series=benchmark_equity_curve,
            trade_logs=combined_trades,
            weights_df=None,
            benchmark_name="EqualWeight_Staggered",
        )

        return BacktestResult(
            strategy_name=f"{strategy_name}_Staggered",
            model_name=model_name,
            config=self.config,
            equity_curve=df_daily,
            trade_logs=combined_trades,
            daily_weights=pd.DataFrame(),
            report=report,
        )

    def _run_custom_cycle(
        self,
        optimizer_strat: str,
        model_name: str,
        feature_set: str,
        decision_dates: Sequence[pd.Timestamp],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Helper to run a simulation with specific decision dates."""
        opt_cfg = OptimizationConfig(top_k=config.top_k, max_weight=config.max_weight)
        optimizer = get_optimizer(optimizer_strat, opt_cfg)

        model_oos = self.oos[
            (self.oos["model"] == model_name) & (self.oos["feature_set"] == feature_set)
        ]

        cash = config.initial_capital
        positions: Dict[str, float] = {s: 0.0 for s in self.symbols}
        current_weights: Dict[str, float] = {s: 0.0 for s in self.symbols}

        start_decision_date = decision_dates[0]
        start_idx = self.date_to_idx[start_decision_date]
        exec_start_idx = start_idx + 1
        last_decision_date = decision_dates[-1]
        last_idx = min(self.date_to_idx[last_decision_date] + config.rebalance_freq, len(self.trading_dates) - 1)

        decision_set = set(decision_dates)
        daily_records: List[Dict[str, Any]] = []
        trade_records: List[Dict[str, Any]] = []

        for idx in range(exec_start_idx, last_idx + 1):
            curr_date = self.trading_dates[idx]
            prev_date = self.trading_dates[idx - 1]

            if prev_date in decision_set:
                exec_prices = {s: float(self.piv_open.loc[curr_date, s]) for s in self.symbols}
                pre_rebal_val = cash + sum(positions[s] * exec_prices[s] for s in self.symbols)

                cov_start_idx = max(0, idx - 1 - config.lookback_cov)
                hist_returns = self.piv_close.iloc[cov_start_idx : idx].pct_change().dropna()
                cov_matrix = hist_returns.cov() * 252

                scores_df = model_oos[model_oos["date"] == prev_date][["stock", "y_prob"]]

                target_weights = optimizer.allocate(
                    symbols=self.symbols,
                    scores_df=scores_df,
                    cov_matrix=cov_matrix,
                    current_weights=current_weights,
                )

                target_dollars = {s: target_weights.get(s, 0.0) * pre_rebal_val for s in self.symbols}
                current_dollars = {s: positions[s] * exec_prices[s] for s in self.symbols}
                trade_dollars = {s: target_dollars[s] - current_dollars[s] for s in self.symbols}

                turnover = sum(abs(trade_dollars[s]) for s in self.symbols)
                trade_cost = turnover * (config.cost_bps / 10000.0)

                net_alloc_val = pre_rebal_val - trade_cost
                new_positions: Dict[str, float] = {}
                for s in self.symbols:
                    s_dollars = target_weights.get(s, 0.0) * net_alloc_val
                    new_positions[s] = float(s_dollars / exec_prices[s]) if exec_prices[s] > 0 else 0.0

                positions = new_positions
                cash = 0.0
                current_weights = target_weights

                trade_records.append({
                    "decision_date": str(prev_date.date()),
                    "execution_date": str(curr_date.date()),
                    "pre_val": pre_rebal_val,
                    "turnover": turnover,
                    "cost": trade_cost,
                })

            close_prices = {s: float(self.piv_close.loc[curr_date, s]) for s in self.symbols}
            eod_val = cash + sum(positions[s] * close_prices[s] for s in self.symbols)

            daily_records.append({
                "date": curr_date,
                "portfolio_value": eod_val,
            })

        df_daily = pd.DataFrame(daily_records)
        df_trades = pd.DataFrame(trade_records)

        dummy_report = PortfolioPerformanceReport(
            strategy_name=optimizer_strat,
            model_name=model_name,
            start_date=str(df_daily["date"].iloc[0].date()),
            end_date=str(df_daily["date"].iloc[-1].date()),
            trading_days=len(df_daily),
            initial_value=config.initial_capital,
            final_value=float(df_daily["portfolio_value"].iloc[-1]),
            cumulative_return=float((df_daily["portfolio_value"].iloc[-1] / config.initial_capital) - 1.0),
            cagr=0.0,
            annualized_volatility=0.0,
            downside_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            calmar_ratio=0.0,
            benchmark_name="None",
            benchmark_cumulative_return=0.0,
            benchmark_cagr=0.0,
            benchmark_sharpe=0.0,
            alpha_cagr=0.0,
            beta_to_bench=1.0,
            tracking_error=0.0,
            information_ratio=0.0,
            total_turnover=float(df_trades["turnover"].sum()) if not df_trades.empty else 0.0,
            annualized_turnover=0.0,
            total_transaction_costs=float(df_trades["cost"].sum()) if not df_trades.empty else 0.0,
            cost_drag_cagr=0.0,
            mean_effective_n=1.0,
            mean_hhi=1.0,
            max_single_weight=1.0,
            excess_return_t_stat=0.0,
            excess_return_p_value=1.0,
            is_statistically_significant_5pct=False,
            annual_breakdowns=[],
        )

        return BacktestResult(
            strategy_name=optimizer_strat,
            model_name=model_name,
            config=config,
            equity_curve=df_daily,
            trade_logs=df_trades,
            daily_weights=pd.DataFrame(),
            report=dummy_report,
        )
