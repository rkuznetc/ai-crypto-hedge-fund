from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.portfolio.base import OptimizationError
from crypto_hf.portfolio.optimizers import (
    EqualWeightOptimizer,
    HierarchicalRiskParityOptimizer,
    InverseVolatilityOptimizer,
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    OptimizerConfig,
)


def _returns(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    data = rng.normal(0.0005, 0.02, size=(n, 5))
    cols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
    return pd.DataFrame(data, index=idx, columns=cols)


def _config(**overrides: object) -> OptimizerConfig:
    base = dict(min_weight=0.0, max_weight=0.35, allow_short=False)
    base.update(overrides)
    return OptimizerConfig(**base)


def _assert_valid_weights(weights: pd.Series, config: OptimizerConfig) -> None:
    assert np.isclose(weights.sum(), 1.0)
    assert not weights.isna().any()
    assert (weights >= -1e-8).all()
    assert (weights <= config.max_weight + 1e-6).all()


@pytest.mark.parametrize(
    "optimizer_cls",
    [
        EqualWeightOptimizer,
        InverseVolatilityOptimizer,
        MinVarianceOptimizer,
        MaxSharpeOptimizer,
        HierarchicalRiskParityOptimizer,
    ],
)
def test_optimizers_return_valid_weights(optimizer_cls: type) -> None:
    config = _config()
    optimizer = optimizer_cls(config)
    weights = optimizer.optimize(_returns()).weights
    _assert_valid_weights(weights, config)


def test_min_variance_portfolio_variance_positive() -> None:
    returns = _returns()
    config = _config()
    weights = MinVarianceOptimizer(config).optimize(returns).weights
    cov = returns.cov().to_numpy()
    variance = float(weights.to_numpy() @ cov @ weights.to_numpy())
    assert variance >= 0.0
