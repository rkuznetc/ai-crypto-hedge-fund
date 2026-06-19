from __future__ import annotations

import pandas as pd
import pytest

from crypto_hf.backtesting.vectorbt_engine import (
    BINARY_POSITIONS_MSG,
    VectorbtBacktester,
)
from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy


def test_rejects_non_binary_positions() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    prices = pd.Series([100.0, 100.0, 100.0], index=idx)
    positions = pd.Series([0.0, 0.5, 1.0], index=idx)

    with pytest.raises(ValueError, match=BINARY_POSITIONS_MSG):
        VectorbtBacktester(initial_cash=10_000, fee_rate=0.0).run(prices, positions)


def test_strict_alignment_rejects_missing_positions() -> None:
    price_idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    position_idx = price_idx[:3]
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=price_idx)
    positions = pd.Series([0.0, 0.0, 1.0], index=position_idx)

    with pytest.raises(ValueError, match="strict_alignment=True"):
        VectorbtBacktester(initial_cash=10_000, fee_rate=0.0, strict_alignment=True).run(
            prices,
            positions,
        )


def test_relaxed_alignment_fills_missing_positions() -> None:
    price_idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    position_idx = price_idx[:3]
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=price_idx)
    positions = pd.Series([0.0, 0.0, 1.0], index=position_idx)

    result = VectorbtBacktester(
        initial_cash=10_000,
        fee_rate=0.0,
        strict_alignment=False,
    ).run(prices, positions)

    assert not result.equity_curve.empty
    assert len(result.positions) == len(prices)


def test_slippage_parameter_accepted() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    positions = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx)

    result = VectorbtBacktester(
        initial_cash=10_000,
        fee_rate=0.0,
        slippage=0.001,
    ).run(prices, positions)

    assert not result.equity_curve.empty
    assert "total_return" in result.metrics


def test_zero_slippage_matches_prior_behavior() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    positions = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx)

    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0, slippage=0.0).run(
        prices,
        positions,
    )

    assert result.equity_curve.iloc[-1] == pytest.approx(10_000)


def test_entry_timing_proves_fill_at_next_bar_close() -> None:
    """Buy at close[1]=200, not close[0]=100; mark-to-market at 300 on bar 2."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.Series([100.0, 200.0, 300.0, 300.0], index=idx)

    signals = BuyAndHoldStrategy().generate_signals(pd.DataFrame({"close": prices}, index=idx))
    assert signals["signal"].iloc[0] == 1.0
    assert signals["position"].iloc[0] == 0.0
    assert signals["position"].iloc[1] == 1.0

    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0, slippage=0.0).run(
        prices,
        signals["position"],
        strategy_name="buy_and_hold",
    )

    assert result.equity_curve.iloc[1] == pytest.approx(10_000)
    assert result.equity_curve.iloc[2] == pytest.approx(15_000)
