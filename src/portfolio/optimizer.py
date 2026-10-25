"""
Modular Portfolio Optimizers for Indian SPCM (Phase 7).

Provides mathematical allocation algorithms for Indian equities:
1. EqualWeight: Uniform baseline allocation across eligible assets (1/N).
2. TopKEqualWeight: Model-informed selection allocating 1/K to top-ranked assets.
3. TopKScoreWeighted: Model-informed selection tilting weights by model score.
4. InverseVolatility: Risk-parity proxy allocating inversely to rolling volatility (1/sigma).
5. MinVariance: Full-universe minimum variance optimizing w^T Sigma w under box constraints.
6. TopKMinVariance: Risk-aware model portfolio minimizing variance among top-K model picks.

Includes robust covariance shrinkage (Ledoit-Wolf / diagonal shrinkage) and
graceful fallbacks to prevent runtime failures during optimization.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("portfolio_optimizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class OptimizationConfig:
    """Configuration parameters for portfolio optimizers."""
    top_k: int = 3
    min_weight: float = 0.0
    max_weight: float = 0.40
    topk_min_weight: float = 0.10
    topk_max_weight: float = 0.60
    shrinkage_factor: float = 0.10
    ridge_epsilon: float = 1e-6
    max_iter: int = 500
    tolerance: float = 1e-7


class BasePortfolioOptimizer(ABC):
    """Abstract base class for all portfolio allocation strategies."""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    @abstractmethod
    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Computes portfolio weights summing to 1.0 for the given universe.

        Parameters:
            symbols: List of eligible stock tickers.
            scores_df: DataFrame with 'stock' and 'y_prob' (or model ranking).
            cov_matrix: Covariance matrix of historical returns up to decision cutoff.
            current_weights: Existing portfolio weights before rebalance (for turnover control).

        Returns:
            Dictionary mapping stock ticker to portfolio weight (summing to 1.0).
        """
        pass

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensures weights are strictly positive and sum to exactly 1.0."""
        clean = {s: max(0.0, float(w)) for s, w in weights.items()}
        total = sum(clean.values())
        if total <= 1e-9:
            # Fallback to equal weight
            n = len(clean)
            return {s: 1.0 / n for s in clean}
        return {s: w / total for s, w in clean.items()}

    def _shrink_covariance(self, cov: np.ndarray) -> np.ndarray:
        """
        Applies diagonal shrinkage and ridge regularization to guarantee
        positive-definiteness and numerical stability:
            Sigma_reg = (1 - delta) * Sigma + delta * diag(Sigma) + eps * I
        """
        cov_arr = np.array(cov, dtype=float)
        p = cov_arr.shape[0]
        diag_elements = np.diag(np.diag(cov_arr))
        shrunk = (1.0 - self.config.shrinkage_factor) * cov_arr + self.config.shrinkage_factor * diag_elements
        regularized = shrunk + self.config.ridge_epsilon * np.eye(p)
        return regularized


class EqualWeightOptimizer(BasePortfolioOptimizer):
    """Equal-weight allocation: w_i = 1 / N across all eligible assets."""

    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not symbols:
            return {}
        n = len(symbols)
        return {s: 1.0 / n for s in symbols}


class TopKEqualWeightOptimizer(BasePortfolioOptimizer):
    """
    Model-informed Top-K selection:
    Selects top K stocks by model score (y_prob) and allocates 1 / K to each.
    """

    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not symbols:
            return {}
        k = min(self.config.top_k, len(symbols))

        if scores_df is None or scores_df.empty or "y_prob" not in scores_df.columns:
            logger.warning("No scores provided for TopKEqualWeight; falling back to EqualWeight.")
            return {s: 1.0 / len(symbols) for s in symbols}

        filtered = scores_df[scores_df["stock"].isin(symbols)].sort_values("y_prob", ascending=False)
        top_stocks = list(filtered["stock"].head(k))

        if len(top_stocks) < k:
            remaining = [s for s in symbols if s not in top_stocks]
            top_stocks.extend(remaining[: k - len(top_stocks)])

        weights = {s: (1.0 / k if s in top_stocks else 0.0) for s in symbols}
        return self._normalize_weights(weights)


class TopKScoreWeightedOptimizer(BasePortfolioOptimizer):
    """
    Model-informed Top-K Score-Weighted allocation:
    Selects top K stocks by score and allocates proportional to model score,
    subject to max position weight capping.
    """

    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not symbols:
            return {}
        k = min(self.config.top_k, len(symbols))

        if scores_df is None or scores_df.empty or "y_prob" not in scores_df.columns:
            logger.warning("No scores provided for TopKScoreWeighted; falling back to EqualWeight.")
            return {s: 1.0 / len(symbols) for s in symbols}

        filtered = scores_df[scores_df["stock"].isin(symbols)].sort_values("y_prob", ascending=False)
        top_df = filtered.head(k).copy()

        scores = top_df["y_prob"].values
        tot_score = float(np.sum(scores))

        if tot_score <= 1e-9:
            return TopKEqualWeightOptimizer(self.config).allocate(symbols, scores_df, cov_matrix, current_weights)

        raw_weights = scores / tot_score
        # Cap at max_weight
        capped = np.minimum(raw_weights, self.config.topk_max_weight)
        norm_capped = capped / np.sum(capped)

        weights = {s: 0.0 for s in symbols}
        for s, w in zip(top_df["stock"], norm_capped):
            weights[s] = float(w)

        return self._normalize_weights(weights)


class InverseVolatilityOptimizer(BasePortfolioOptimizer):
    """
    Risk-based Inverse Volatility allocation:
    w_i proportional to 1 / sigma_i.
    Can be run across universe or filtered to model Top-K if scores are provided.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None, top_k_filter: bool = False):
        super().__init__(config)
        self.top_k_filter = top_k_filter

    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not symbols:
            return {}

        target_symbols = list(symbols)
        if self.top_k_filter and scores_df is not None and not scores_df.empty:
            k = min(self.config.top_k, len(symbols))
            filtered = scores_df[scores_df["stock"].isin(symbols)].sort_values("y_prob", ascending=False)
            target_symbols = list(filtered["stock"].head(k))

        if cov_matrix is None or cov_matrix.empty:
            logger.warning("Covariance missing for InverseVolatility; falling back to EqualWeight.")
            return {s: (1.0 / len(target_symbols) if s in target_symbols else 0.0) for s in symbols}

        available = [s for s in target_symbols if s in cov_matrix.index and s in cov_matrix.columns]
        if not available:
            return {s: (1.0 / len(target_symbols) if s in target_symbols else 0.0) for s in symbols}

        sub_cov = cov_matrix.loc[available, available].values
        variances = np.diag(sub_cov)
        volatilities = np.sqrt(np.maximum(variances, 1e-8))
        inv_vols = 1.0 / volatilities
        sum_inv = np.sum(inv_vols)

        weights = {s: 0.0 for s in symbols}
        for s, inv_v in zip(available, inv_vols):
            weights[s] = float(inv_v / sum_inv)

        return self._normalize_weights(weights)


class MinVarianceOptimizer(BasePortfolioOptimizer):
    """
    Global Minimum Variance allocation on the full universe:
        min_w w^T Sigma w
        subject to: sum(w) = 1, min_weight <= w_i <= max_weight.
    Uninformed by model scores; serves as a pure risk-minimizing baseline.
    """

    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not symbols:
            return {}
        n = len(symbols)

        if cov_matrix is None or cov_matrix.empty:
            logger.warning("Covariance missing for MinVariance; falling back to EqualWeight.")
            return {s: 1.0 / n for s in symbols}

        available = [s for s in symbols if s in cov_matrix.index and s in cov_matrix.columns]
        p = len(available)
        if p < 2:
            return {s: 1.0 / n for s in symbols}

        sub_cov = cov_matrix.loc[available, available].values
        reg_cov = self._shrink_covariance(sub_cov)

        init_w = np.ones(p) / p
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(p)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w: np.ndarray) -> float:
            return float(w @ reg_cov @ w)

        try:
            res = minimize(
                objective,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.config.max_iter, "ftol": self.config.tolerance},
            )

            if res.success and not np.isnan(res.x).any():
                opt_w = res.x
            else:
                logger.warning(f"MinVariance optimization did not converge ({res.message}); falling back to InverseVol.")
                return InverseVolatilityOptimizer(self.config).allocate(symbols, scores_df, cov_matrix, current_weights)
        except Exception as exc:
            logger.warning(f"MinVariance optimization error ({exc}); falling back to InverseVol.")
            return InverseVolatilityOptimizer(self.config).allocate(symbols, scores_df, cov_matrix, current_weights)

        weights = {s: 0.0 for s in symbols}
        for s, w in zip(available, opt_w):
            weights[s] = float(w)

        return self._normalize_weights(weights)


class TopKMinVarianceOptimizer(BasePortfolioOptimizer):
    """
    Model-informed Risk-Aware allocation:
    Selects Top K stocks by model score (y_prob), then solves for the
    Minimum Variance portfolio among those K assets:
        min_w w^T Sigma_K w
        subject to: sum(w) = 1, topk_min_weight <= w_i <= topk_max_weight.
    Combines model alpha with risk-aware diversification.
    """

    def allocate(
        self,
        symbols: Sequence[str],
        scores_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if not symbols:
            return {}
        k = min(self.config.top_k, len(symbols))

        if scores_df is None or scores_df.empty or "y_prob" not in scores_df.columns:
            logger.warning("No scores provided for TopKMinVariance; falling back to EqualWeight.")
            return {s: 1.0 / len(symbols) for s in symbols}

        filtered = scores_df[scores_df["stock"].isin(symbols)].sort_values("y_prob", ascending=False)
        top_stocks = list(filtered["stock"].head(k))

        if cov_matrix is None or cov_matrix.empty:
            return TopKEqualWeightOptimizer(self.config).allocate(symbols, scores_df, cov_matrix, current_weights)

        available = [s for s in top_stocks if s in cov_matrix.index and s in cov_matrix.columns]
        p = len(available)
        if p < 2:
            return TopKEqualWeightOptimizer(self.config).allocate(symbols, scores_df, cov_matrix, current_weights)

        sub_cov = cov_matrix.loc[available, available].values
        reg_cov = self._shrink_covariance(sub_cov)

        init_w = np.ones(p) / p
        bounds = [(self.config.topk_min_weight, self.config.topk_max_weight) for _ in range(p)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w: np.ndarray) -> float:
            return float(w @ reg_cov @ w)

        try:
            res = minimize(
                objective,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.config.max_iter, "ftol": self.config.tolerance},
            )

            if res.success and not np.isnan(res.x).any():
                opt_w = res.x
            else:
                logger.warning(f"TopKMinVariance did not converge ({res.message}); falling back to InverseVol on TopK.")
                return InverseVolatilityOptimizer(self.config, top_k_filter=True).allocate(
                    symbols, scores_df, cov_matrix, current_weights
                )
        except Exception as exc:
            logger.warning(f"TopKMinVariance error ({exc}); falling back to InverseVol on TopK.")
            return InverseVolatilityOptimizer(self.config, top_k_filter=True).allocate(
                symbols, scores_df, cov_matrix, current_weights
            )

        weights = {s: 0.0 for s in symbols}
        for s, w in zip(available, opt_w):
            weights[s] = float(w)

        return self._normalize_weights(weights)


def get_optimizer(strategy_name: str, config: Optional[OptimizationConfig] = None) -> BasePortfolioOptimizer:
    """Factory function returning the optimizer instance for a given strategy name."""
    strat = strategy_name.lower().strip()
    if strat in ("equal_weight", "benchmark_equal_weight", "ew"):
        return EqualWeightOptimizer(config)
    elif strat in ("topk_equal", "topk_equal_weight", "top_k_equal"):
        return TopKEqualWeightOptimizer(config)
    elif strat in ("topk_score", "topk_score_weighted", "score_weighted"):
        return TopKScoreWeightedOptimizer(config)
    elif strat in ("inverse_volatility", "inv_vol", "universe_inv_vol"):
        return InverseVolatilityOptimizer(config, top_k_filter=False)
    elif strat in ("topk_inv_vol", "topk_inverse_volatility"):
        return InverseVolatilityOptimizer(config, top_k_filter=True)
    elif strat in ("universe_minvar", "min_variance", "minvar"):
        return MinVarianceOptimizer(config)
    elif strat in ("topk_minvar", "topk_min_variance"):
        return TopKMinVarianceOptimizer(config)
    else:
        raise ValueError(
            f"Unknown portfolio strategy: '{strategy_name}'. "
            f"Available: equal_weight, topk_equal, topk_score, topk_minvar, topk_inv_vol, universe_minvar, inverse_volatility"
        )
