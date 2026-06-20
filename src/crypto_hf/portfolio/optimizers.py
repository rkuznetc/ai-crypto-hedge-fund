from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from crypto_hf.portfolio.base import (
    BasePortfolioOptimizer,
    OptimizationError,
    PortfolioWeights,
    normalize_weights,
    validate_portfolio_weights,
)
from crypto_hf.portfolio.estimators import (
    historical_mean_returns,
    ledoit_wolf_covariance,
    sample_covariance,
)


@dataclass
class OptimizerConfig:
    """Shared constraints and estimation settings for portfolio optimizers."""

    min_weight: float = 0.0
    max_weight: float = 1.0
    allow_short: bool = False
    risk_free_rate: float = 0.0
    annualization_factor: int = 365
    covariance_method: str = "ledoit_wolf"
    expected_return_method: str = "historical_mean"
    ewm_span: int = 60


def _estimate_covariance(returns: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    if config.covariance_method == "sample":
        return sample_covariance(returns, config.annualization_factor)
    if config.covariance_method == "ledoit_wolf":
        return ledoit_wolf_covariance(returns, config.annualization_factor)
    raise ValueError(f"Unsupported covariance_method: {config.covariance_method}")


def _estimate_expected_returns(returns: pd.DataFrame, config: OptimizerConfig) -> pd.Series:
    if config.expected_return_method == "historical_mean":
        return historical_mean_returns(returns, config.annualization_factor)
    if config.expected_return_method == "ewm_mean":
        from crypto_hf.portfolio.estimators import exponentially_weighted_mean_returns

        return exponentially_weighted_mean_returns(
            returns,
            span=config.ewm_span,
            annualization_factor=config.annualization_factor,
        )
    raise ValueError(f"Unsupported expected_return_method: {config.expected_return_method}")


def _clip_and_normalize(
    raw: pd.Series,
    config: OptimizerConfig,
) -> pd.Series:
    clipped = raw.clip(lower=0.0 if not config.allow_short else None, upper=config.max_weight)
    if clipped.sum() <= 0:
        raise OptimizationError("All weights clipped to zero")
    weights = normalize_weights(clipped)
    validate_portfolio_weights(
        weights,
        allow_short=config.allow_short,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
    )
    return weights


def _optimize_bounded_weights(
    objective,
    n_assets: int,
    config: OptimizerConfig,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    if x0 is None:
        x0 = np.full(n_assets, 1.0 / n_assets)
    bounds = [(config.min_weight, config.max_weight)] * n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise OptimizationError(f"Optimization failed: {result.message}")
    return result.x


class EqualWeightOptimizer(BasePortfolioOptimizer):
    name = "equal_weight"

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def optimize(self, returns_train: pd.DataFrame) -> PortfolioWeights:
        n_assets = len(returns_train.columns)
        weights = pd.Series(1.0 / n_assets, index=returns_train.columns)
        weights = _clip_and_normalize(weights, self.config)
        return PortfolioWeights(
            name=self.name,
            weights=weights,
            metadata={
                "method": self.name,
                "covariance_method": self.config.covariance_method,
                "expected_return_method": self.config.expected_return_method,
                "max_weight": self.config.max_weight,
            },
        )


class InverseVolatilityOptimizer(BasePortfolioOptimizer):
    name = "inverse_volatility"

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def optimize(self, returns_train: pd.DataFrame) -> PortfolioWeights:
        vol = returns_train.std()
        if (vol <= 0).any() or vol.isna().any():
            raise OptimizationError("Invalid volatilities for inverse-volatility weights")
        raw = 1.0 / vol
        weights = _clip_and_normalize(raw, self.config)
        return PortfolioWeights(
            name=self.name,
            weights=weights,
            metadata={
                "method": self.name,
                "covariance_method": self.config.covariance_method,
                "expected_return_method": self.config.expected_return_method,
                "max_weight": self.config.max_weight,
            },
        )


class MinVarianceOptimizer(BasePortfolioOptimizer):
    name = "min_variance"

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def optimize(self, returns_train: pd.DataFrame) -> PortfolioWeights:
        cov = _estimate_covariance(returns_train, self.config).to_numpy()
        n_assets = len(returns_train.columns)

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        solution = _optimize_bounded_weights(objective, n_assets, self.config)
        weights = pd.Series(solution, index=returns_train.columns)
        validate_portfolio_weights(
            weights,
            allow_short=self.config.allow_short,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        return PortfolioWeights(
            name=self.name,
            weights=weights,
            metadata={
                "method": self.name,
                "covariance_method": self.config.covariance_method,
                "expected_return_method": self.config.expected_return_method,
                "max_weight": self.config.max_weight,
            },
        )


class MaxSharpeOptimizer(BasePortfolioOptimizer):
    name = "max_sharpe"

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def optimize(self, returns_train: pd.DataFrame) -> PortfolioWeights:
        mu = _estimate_expected_returns(returns_train, self.config).to_numpy()
        cov = _estimate_covariance(returns_train, self.config).to_numpy()
        rf = self.config.risk_free_rate
        n_assets = len(returns_train.columns)

        def objective(w: np.ndarray) -> float:
            port_return = float(w @ mu) - rf
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol <= 0:
                return 1e6
            return -port_return / port_vol

        solution = _optimize_bounded_weights(objective, n_assets, self.config)
        weights = pd.Series(solution, index=returns_train.columns)
        validate_portfolio_weights(
            weights,
            allow_short=self.config.allow_short,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        return PortfolioWeights(
            name=self.name,
            weights=weights,
            metadata={
                "method": self.name,
                "covariance_method": self.config.covariance_method,
                "expected_return_method": self.config.expected_return_method,
                "risk_free_rate": self.config.risk_free_rate,
                "max_weight": self.config.max_weight,
            },
        )


class HierarchicalRiskParityOptimizer(BasePortfolioOptimizer):
    name = "hrp"

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def optimize(self, returns_train: pd.DataFrame) -> PortfolioWeights:
        cov = _estimate_covariance(returns_train, self.config)
        corr = returns_train.corr().clip(-1.0, 1.0)
        dist = np.sqrt(0.5 * (1.0 - corr.to_numpy()))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = leaves_list(link)
        ordered = returns_train.columns[sort_ix]
        weights = self._recursive_bisection(cov.loc[ordered, ordered])
        weights = weights.reindex(returns_train.columns)
        weights = _clip_and_normalize(weights, self.config)
        return PortfolioWeights(
            name=self.name,
            weights=weights,
            metadata={
                "method": self.name,
                "covariance_method": self.config.covariance_method,
                "expected_return_method": self.config.expected_return_method,
                "max_weight": self.config.max_weight,
            },
        )

    def _recursive_bisection(self, cov: pd.DataFrame) -> pd.Series:
        weights = pd.Series(1.0, index=cov.index)
        clusters = [list(cov.index)]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) == 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            var_left = self._cluster_variance(cov, left)
            var_right = self._cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            clusters.extend([left, right])
        return normalize_weights(weights)

    @staticmethod
    def _cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
        sub = cov.loc[assets, assets]
        inv_var = 1.0 / np.diag(sub.to_numpy())
        w = inv_var / inv_var.sum()
        return float(w @ sub.to_numpy() @ w)
