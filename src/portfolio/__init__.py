"""
Portfolio optimization and backtesting package for Indian SPCM (Phase 7).
"""
from src.portfolio.optimizer import (
    BasePortfolioOptimizer,
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
    MinVarianceOptimizer,
    OptimizationConfig,
    TopKEqualWeightOptimizer,
    TopKMinVarianceOptimizer,
    TopKScoreWeightedOptimizer,
    get_optimizer,
)
from src.portfolio.metrics import PortfolioMetricsCalculator, PortfolioPerformanceReport
from src.portfolio.backtest import BacktestEngine, BacktestConfig, BacktestResult

__all__ = [
    "BasePortfolioOptimizer",
    "EqualWeightOptimizer",
    "InverseVolatilityOptimizer",
    "MinVarianceOptimizer",
    "OptimizationConfig",
    "TopKEqualWeightOptimizer",
    "TopKMinVarianceOptimizer",
    "TopKScoreWeightedOptimizer",
    "get_optimizer",
    "PortfolioMetricsCalculator",
    "PortfolioPerformanceReport",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
]
