from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.portfolio.backtesting import run_static_portfolio_backtest


def _prices(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.DataFrame(
        {
            "BTC/USDT": np.linspace(100, 110, n),
            "ETH/USDT": np.linspace(50, 55, n),
            "BNB/USDT": np.linspace(300, 310, n),
        },
        index=idx,
    )
    return close


def test_static_portfolio_buy_and_hold_no_rebalance() -> None:
    prices = _prices()
    weights = pd.Series({"BTC/USDT": 1 / 3, "ETH/USDT": 1 / 3, "BNB/USDT": 1 / 3})
    result = run_static_portfolio_backtest(
        "equal_weight",
        weights,
        prices,
        initial_cash=10_000,
        fee_rate=0.0,
        slippage=0.0,
    )
    shares = result.asset_values.iloc[0] / prices.iloc[0]
    for i in range(1, len(prices)):
        expected = (shares * prices.iloc[i]).sum()
        assert result.equity_curve.iloc[i] == pytest.approx(expected, rel=1e-6)


def test_fees_reduce_return() -> None:
    prices = _prices()
    weights = pd.Series({"BTC/USDT": 0.5, "ETH/USDT": 0.5, "BNB/USDT": 0.0})
    no_fee = run_static_portfolio_backtest(
        "test",
        weights,
        prices,
        initial_cash=10_000,
        fee_rate=0.0,
        slippage=0.0,
    )
    with_fee = run_static_portfolio_backtest(
        "test",
        weights,
        prices,
        initial_cash=10_000,
        fee_rate=0.01,
        slippage=0.0,
    )
    assert with_fee.metrics["total_return"] <= no_fee.metrics["total_return"]


def test_metrics_non_empty() -> None:
    result = run_static_portfolio_backtest(
        "test",
        pd.Series({"BTC/USDT": 1.0}),
        _prices()[["BTC/USDT"]],
        initial_cash=10_000,
        fee_rate=0.0,
        slippage=0.0,
    )
    assert result.metrics["total_return"] != 0.0 or result.equity_curve.iloc[-1] > 0
    assert "sharpe_ratio" in result.metrics
    assert "concentration_hhi" in result.diagnostics
