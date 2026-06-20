from __future__ import annotations

import pandas as pd
import pytest

from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.config import SingleAssetModelsConfig
from crypto_hf.pipeline.single_asset_models import _return_diagnostics
from crypto_hf.pipeline.threshold_calibration import calibrate_probability_threshold
from crypto_hf.strategies.ensemble import EnsembleMajorityVoteStrategy
from crypto_hf.strategies.statistical import VolatilityRegimeFilterStrategy


def _ohlcv(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000.0,
            "returns": close.pct_change().fillna(0.0),
        },
        index=idx,
    )


def test_gross_total_return_is_at_least_net_with_fees() -> None:
    data = _ohlcv(20)
    prices = data["close"]
    signal = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 2, index=data.index)
    positions = signal.shift(1).fillna(0.0)
    net_engine = VectorbtBacktester(initial_cash=10_000, fee_rate=0.001, slippage=0.0)
    gross_engine = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0, slippage=0.0)
    net_result = net_engine.run(prices, positions, strategy_name="fee_strategy")
    diagnostics = _return_diagnostics(net_result, gross_engine, prices, positions)
    assert diagnostics["gross_total_return"] >= diagnostics["net_total_return"]
    assert diagnostics["estimated_cost_impact"] >= 0.0


def test_ensemble_majority_vote_logic() -> None:
    data = _ohlcv(4)
    components = {
        "a": pd.Series([1.0, 1.0, 0.0, 1.0], index=data.index),
        "b": pd.Series([1.0, 0.0, 0.0, 1.0], index=data.index),
        "c": pd.Series([0.0, 1.0, 1.0, 1.0], index=data.index),
    }
    out = EnsembleMajorityVoteStrategy(components, min_votes=2).generate_signals(data)
    assert out["signal"].tolist() == [1.0, 1.0, 0.0, 1.0]
    expected = out["signal"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(out["position"], expected, check_names=False)


def test_ensemble_missing_component_raises() -> None:
    data = _ohlcv(3)
    components = {"a": pd.Series([1.0, 0.0], index=data.index[:2])}
    with pytest.raises(ValueError, match="missing"):
        EnsembleMajorityVoteStrategy(components, min_votes=1).generate_signals(data)


def test_volatility_regime_filter_binary_shifted_positions() -> None:
    data = _ohlcv(15)
    strategy = VolatilityRegimeFilterStrategy(window=3, threshold=999.0)
    out = strategy.generate_signals(data)
    assert set(out["signal"].unique()).issubset({0.0, 1.0})
    expected = out["signal"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(out["position"], expected, check_names=False)


def test_volatility_threshold_calibration_uses_train_validation_only() -> None:
    train_val = _ohlcv(20)
    test = _ohlcv(10)
    test.index = pd.date_range("2024-02-01", periods=10, freq="D", tz="UTC")
    threshold = VolatilityRegimeFilterStrategy.calibrate_threshold(
        train_val,
        window=3,
        annualization_factor=365,
        use_quantile=True,
        fixed_threshold=0.5,
        quantile=0.7,
    )
    assert threshold > 0.0
    assert threshold != VolatilityRegimeFilterStrategy.calibrate_threshold(
        test,
        window=3,
        annualization_factor=365,
        use_quantile=True,
        fixed_threshold=0.5,
        quantile=0.7,
    )


def test_volatility_regime_passes_backtester() -> None:
    data = _ohlcv(25)
    strategy = VolatilityRegimeFilterStrategy(window=5, threshold=1.0)
    positions = strategy.generate_signals(data)["position"]
    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0).run(data["close"], positions)
    assert not result.equity_curve.empty
