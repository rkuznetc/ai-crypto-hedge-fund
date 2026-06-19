from __future__ import annotations

import pandas as pd
import pytest

from crypto_hf.backtesting.vectorbt_engine import (
    VectorbtBacktester,
    positions_to_entries_exits,
)
from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy


def test_entry_executes_at_next_bar_close() -> None:
    """signal[t] -> position[t+1] -> fill at close[t+1]."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.Series([100.0, 200.0, 200.0, 200.0], index=idx)

    signals = BuyAndHoldStrategy().generate_signals(pd.DataFrame({"close": prices}, index=idx))
    assert signals["signal"].iloc[0] == 1.0
    assert signals["position"].iloc[0] == 0.0
    assert signals["position"].iloc[1] == 1.0

    entries, exits = positions_to_entries_exits(signals["position"])
    assert not entries.iloc[0]
    assert entries.iloc[1]
    assert not exits.any()

    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0).run(
        prices,
        signals["position"],
        strategy_name="buy_and_hold",
    )

    # Bar 0: still in cash.
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    # Bar 1: bought at close=200, not at close=100.
    assert result.equity_curve.iloc[1] == pytest.approx(10_000)
    # Bar 2+: mark-to-market at 200.
    assert result.equity_curve.iloc[2] == pytest.approx(10_000)


def test_backtest_smoke() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    close = pd.Series(range(100, 150), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )

    signals = BuyAndHoldStrategy().generate_signals(df)
    result = VectorbtBacktester(
        initial_cash=10_000,
        fee_rate=0.001,
        annualization_factor=365,
    ).run(df["close"], signals["position"], strategy_name="buy_and_hold")

    assert not result.equity_curve.empty
    assert not result.returns.empty
    assert "total_return" in result.metrics
    assert "sharpe_ratio" in result.metrics
