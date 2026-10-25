"""
Comprehensive Portfolio Performance, Risk, and Attribution Metrics (Phase 7).

Computes institutional-grade quantitative finance metrics:
1. Returns: Cumulative Return, CAGR (Compound Annual Growth Rate).
2. Risk: Annualized Volatility, Downside Deviation, Maximum Drawdown, Max Drawdown Duration.
3. Risk-Adjusted: Sharpe Ratio (with configurable risk-free rate), Sortino Ratio, Calmar Ratio.
4. Benchmark Comparison: Active Return (Alpha), Beta, Tracking Error, Information Ratio.
5. Trading & Friction: Total Turnover, Annualized Turnover, Cumulative Transaction Costs.
6. Diversification & Concentration: Herfindahl-Hirschman Index (HHI), Effective N (Inverse HHI).
7. Statistical Significance: Paired Student's t-test and p-value on daily excess returns.
8. Periodic Attribution: Calendar year breakdowns (2022, 2023, 2024).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_RISK_FREE_RATE = 0.06  # 6.0% annual Indian 91-day T-bill / RBI policy proxy
TRADING_DAYS_PER_YEAR = 252


@dataclass
class AnnualPerformance:
    """Performance breakdown for a single calendar year."""
    year: int
    total_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    alpha_vs_bench: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioPerformanceReport:
    """Comprehensive performance and risk evaluation report for a portfolio strategy."""
    strategy_name: str
    model_name: str
    start_date: str
    end_date: str
    trading_days: int
    initial_value: float
    final_value: float
    cumulative_return: float
    cagr: float
    annualized_volatility: float
    downside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    # Benchmark Comparison
    benchmark_name: str
    benchmark_cumulative_return: float
    benchmark_cagr: float
    benchmark_sharpe: float
    alpha_cagr: float
    beta_to_bench: float
    tracking_error: float
    information_ratio: float
    # Turnover & Costs
    total_turnover: float
    annualized_turnover: float
    total_transaction_costs: float
    cost_drag_cagr: float
    # Diversification & Concentration
    mean_effective_n: float
    mean_hhi: float
    max_single_weight: float
    # Statistical Significance
    excess_return_t_stat: float
    excess_return_p_value: float
    is_statistically_significant_5pct: bool
    # Annual breakdown
    annual_breakdowns: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioMetricsCalculator:
    """Calculates all quantitative metrics from daily portfolio equity curves and trade logs."""

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    def compute_metrics(
        self,
        strategy_name: str,
        model_name: str,
        equity_series: pd.Series,
        benchmark_series: pd.Series,
        trade_logs: Optional[pd.DataFrame] = None,
        weights_df: Optional[pd.DataFrame] = None,
        benchmark_name: str = "EqualWeight",
    ) -> PortfolioPerformanceReport:
        """
        Computes all portfolio metrics given daily marked-to-market equity curves.

        Parameters:
            strategy_name: Name of portfolio allocation strategy.
            model_name: Name of underlying predictive model.
            equity_series: Daily portfolio value indexed by date.
            benchmark_series: Daily benchmark value indexed by date.
            trade_logs: DataFrame of rebalances with 'turnover' and 'cost'.
            weights_df: DataFrame of daily or rebalance portfolio weights across symbols.
            benchmark_name: Label for benchmark.
        """
        # Align series
        aligned = pd.DataFrame({"strat": equity_series, "bench": benchmark_series}).dropna()
        if len(aligned) < 5:
            raise ValueError(f"Insufficient aligned trading days ({len(aligned)}) for metrics calculation.")

        strat_vals = aligned["strat"]
        bench_vals = aligned["bench"]
        dates = aligned.index

        initial_val = float(strat_vals.iloc[0])
        final_val = float(strat_vals.iloc[-1])
        n_days = len(aligned)
        years = max(n_days / TRADING_DAYS_PER_YEAR, 0.01)

        # 1. Returns
        cum_ret = (final_val / initial_val) - 1.0
        cagr = (final_val / initial_val) ** (1.0 / years) - 1.0

        daily_returns = strat_vals.pct_change().dropna()
        daily_bench_returns = bench_vals.pct_change().dropna()

        # 2. Risk
        ann_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns) > 1 else 0.0

        # Downside deviation
        excess_daily = daily_returns - self.daily_rf
        downside_diff = np.minimum(excess_daily, 0.0)
        downside_vol = float(np.sqrt(np.mean(downside_diff**2)) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside_diff) > 0 else 0.0

        # Sharpe and Sortino
        sharpe = float((cagr - self.risk_free_rate) / ann_vol) if ann_vol > 1e-6 else 0.0
        sortino = float((cagr - self.risk_free_rate) / downside_vol) if downside_vol > 1e-6 else 0.0

        # Drawdowns
        cummax = strat_vals.cummax()
        drawdowns = (strat_vals - cummax) / cummax
        mdd = float(drawdowns.min())

        # Max drawdown duration
        is_underwater = drawdowns < 0.0
        max_duration = 0
        curr_duration = 0
        for uw in is_underwater:
            if uw:
                curr_duration += 1
                if curr_duration > max_duration:
                    max_duration = curr_duration
            else:
                curr_duration = 0

        calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-6 else 0.0

        # 3. Benchmark Comparisons
        bench_init = float(bench_vals.iloc[0])
        bench_final = float(bench_vals.iloc[-1])
        bench_cum_ret = (bench_final / bench_init) - 1.0
        bench_cagr = (bench_final / bench_init) ** (1.0 / years) - 1.0
        bench_vol = float(daily_bench_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_bench_returns) > 1 else 0.0
        bench_sharpe = float((bench_cagr - self.risk_free_rate) / bench_vol) if bench_vol > 1e-6 else 0.0

        alpha_cagr = cagr - bench_cagr

        # Beta and Tracking Error
        aligned_rets = pd.DataFrame({"strat": daily_returns, "bench": daily_bench_returns}).dropna()
        if len(aligned_rets) > 2 and aligned_rets["bench"].var() > 1e-9:
            cov_sb = float(aligned_rets["strat"].cov(aligned_rets["bench"]))
            var_b = float(aligned_rets["bench"].var())
            beta = float(cov_sb / var_b)
        else:
            beta = 1.0

        diff_returns = aligned_rets["strat"] - aligned_rets["bench"]
        tracking_error = float(diff_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(diff_returns) > 1 else 0.0
        information_ratio = float(alpha_cagr / tracking_error) if tracking_error > 1e-6 else 0.0

        # Statistical significance: paired t-test on daily excess returns
        if len(diff_returns) > 5 and diff_returns.std() > 1e-9:
            t_res = stats.ttest_1samp(diff_returns, 0.0)
            t_stat = float(t_res.statistic)
            p_val = float(t_res.pvalue)
        else:
            t_stat = 0.0
            p_val = 1.0
        is_sig = bool(p_val < 0.05 and t_stat > 0)

        # 4. Turnover & Transaction Costs
        if trade_logs is not None and not trade_logs.empty:
            total_turnover = float(trade_logs["turnover"].sum())
            annual_turnover = float(total_turnover / years)
            total_costs = float(trade_logs["cost"].sum())
            # Estimate cost drag on CAGR
            gross_final = final_val + total_costs
            gross_cagr = (gross_final / initial_val) ** (1.0 / years) - 1.0
            cost_drag = gross_cagr - cagr
        else:
            total_turnover = 0.0
            annual_turnover = 0.0
            total_costs = 0.0
            cost_drag = 0.0

        # 5. Diversification & Concentration
        if weights_df is not None and not weights_df.empty:
            weight_cols = [c for c in weights_df.columns if c not in ("date", "cycle", "rebalance_date")]
            w_vals = weights_df[weight_cols].values
            hhi_series = np.sum(w_vals**2, axis=1)
            mean_hhi = float(np.mean(hhi_series))
            eff_n_series = 1.0 / np.maximum(hhi_series, 1e-8)
            mean_eff_n = float(np.mean(eff_n_series))
            max_w = float(np.max(w_vals))
        else:
            mean_hhi = 1.0
            mean_eff_n = 1.0
            max_w = 1.0

        # 6. Annual Breakdowns
        annual_records: List[Dict[str, Any]] = []
        df_merged = pd.DataFrame({
            "date": pd.to_datetime(aligned.index),
            "strat": strat_vals.values,
            "bench": bench_vals.values,
        })
        df_merged["year"] = df_merged["date"].dt.year

        for yr, group in df_merged.groupby("year"):
            yr_int = int(yr)
            yr_s = group["strat"].values
            yr_b = group["bench"].values
            yr_ret = float((yr_s[-1] / yr_s[0]) - 1.0)
            yr_bench_ret = float((yr_b[-1] / yr_b[0]) - 1.0)
            yr_daily = pd.Series(yr_s).pct_change().dropna()
            yr_vol = float(yr_daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(yr_daily) > 1 else 0.0
            yr_sharpe = float((yr_ret - self.risk_free_rate) / yr_vol) if yr_vol > 1e-6 else 0.0

            yr_cummax = pd.Series(yr_s).cummax()
            yr_dd = (pd.Series(yr_s) - yr_cummax) / yr_cummax
            yr_mdd = float(yr_dd.min())
            yr_alpha = yr_ret - yr_bench_ret

            annual_records.append(
                AnnualPerformance(
                    year=yr_int,
                    total_return=round(yr_ret, 4),
                    annualized_vol=round(yr_vol, 4),
                    sharpe_ratio=round(yr_sharpe, 4),
                    max_drawdown=round(yr_mdd, 4),
                    alpha_vs_bench=round(yr_alpha, 4),
                ).to_dict()
            )

        return PortfolioPerformanceReport(
            strategy_name=strategy_name,
            model_name=model_name,
            start_date=str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0])[:10],
            end_date=str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])[:10],
            trading_days=n_days,
            initial_value=round(initial_val, 2),
            final_value=round(final_val, 2),
            cumulative_return=round(cum_ret, 4),
            cagr=round(cagr, 4),
            annualized_volatility=round(ann_vol, 4),
            downside_volatility=round(downside_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(mdd, 4),
            max_drawdown_duration_days=int(max_duration),
            calmar_ratio=round(calmar, 4),
            benchmark_name=benchmark_name,
            benchmark_cumulative_return=round(bench_cum_ret, 4),
            benchmark_cagr=round(bench_cagr, 4),
            benchmark_sharpe=round(bench_sharpe, 4),
            alpha_cagr=round(alpha_cagr, 4),
            beta_to_bench=round(beta, 4),
            tracking_error=round(tracking_error, 4),
            information_ratio=round(information_ratio, 4),
            total_turnover=round(total_turnover, 2),
            annualized_turnover=round(annual_turnover, 2),
            total_transaction_costs=round(total_costs, 2),
            cost_drag_cagr=round(cost_drag, 4),
            mean_effective_n=round(mean_eff_n, 2),
            mean_hhi=round(mean_hhi, 4),
            max_single_weight=round(max_w, 4),
            excess_return_t_stat=round(t_stat, 4),
            excess_return_p_value=round(p_val, 4),
            is_statistically_significant_5pct=is_sig,
            annual_breakdowns=annual_records,
        )
