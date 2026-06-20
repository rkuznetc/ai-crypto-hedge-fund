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


def _deterministic_vol_returns() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=120, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    low = rng.normal(0.0002, 0.005, size=len(idx))
    mid = rng.normal(0.0002, 0.015, size=len(idx))
    high = rng.normal(0.0002, 0.030, size=len(idx))
    return pd.DataFrame(
        {
            "asset_low_vol": low,
            "asset_mid_vol": mid,
            "asset_high_vol": high,
        },
        index=idx,
    )


def test_inverse_volatility_prefers_low_vol_assets() -> None:
    returns = _deterministic_vol_returns()
    config = OptimizerConfig(min_weight=0.0, max_weight=1.0, annualization_factor=365)
    weights = InverseVolatilityOptimizer(config).optimize(returns).weights
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert weights["asset_low_vol"] > weights["asset_mid_vol"] > weights["asset_high_vol"]


def test_inverse_volatility_differs_from_equal_weight() -> None:
    returns = _deterministic_vol_returns()
    config = OptimizerConfig(min_weight=0.0, max_weight=1.0, annualization_factor=365)
    inv = InverseVolatilityOptimizer(config).optimize(returns).weights
    eq = EqualWeightOptimizer(config).optimize(returns).weights
    assert not np.allclose(inv.to_numpy(), eq.to_numpy(), atol=1e-4)


def test_inverse_volatility_respects_max_weight() -> None:
    returns = _deterministic_vol_returns()
    config = OptimizerConfig(min_weight=0.0, max_weight=0.45, annualization_factor=365)
    weights = InverseVolatilityOptimizer(config).optimize(returns).weights
    assert (weights <= 0.45 + 1e-6).all()
    assert weights["asset_low_vol"] > weights["asset_high_vol"]


def test_inverse_volatility_invalid_vol_raises() -> None:
    idx = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    returns = pd.DataFrame({"flat": 0.0, "ok": np.linspace(-0.01, 0.01, 10)}, index=idx)
    config = OptimizerConfig(min_weight=0.0, max_weight=1.0)
    with pytest.raises(OptimizationError, match="Invalid volatilities"):
        InverseVolatilityOptimizer(config).optimize(returns)
