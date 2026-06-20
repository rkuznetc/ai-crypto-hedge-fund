from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.portfolio.benchmarks import (
    BTC_ONLY_BENCHMARK_NAME,
    CASH_BENCHMARK_NAME,
    run_btc_only_benchmark,
    run_cash_benchmark,
)
from crypto_hf.portfolio.backtesting import run_static_portfolio_backtest


def _close_prices(symbols: list[str], n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    data = {symbol: 100.0 + np.arange(n) * (i + 1) for i, symbol in enumerate(symbols)}
    return pd.DataFrame(data, index=idx)


def test_btc_only_matches_buy_and_hold_btc() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    prices = _close_prices(symbols)
    weights = pd.Series(0.0, index=symbols)
    weights["BTC/USDT"] = 1.0
    manual = run_static_portfolio_backtest(
        name="manual_btc",
        weights=weights,
        test_close_prices=prices,
        initial_cash=10_000.0,
        fee_rate=0.001,
        slippage=0.0005,
    )
    benchmark = run_btc_only_benchmark(
        symbols,
        prices,
        initial_cash=10_000.0,
        fee_rate=0.001,
        slippage=0.0005,
        annualization_factor=365,
        risk_free_rate=0.0,
    )
    assert benchmark.name == BTC_ONLY_BENCHMARK_NAME
    assert np.isclose(benchmark.weights["BTC/USDT"], 1.0)
    assert np.isclose(
        benchmark.metrics["total_return"],
        manual.metrics["total_return"],
        rtol=1e-9,
    )


def test_cash_benchmark_is_flat() -> None:
    symbols = ["BTC/USDT", "ETH/USDT"]
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    result = run_cash_benchmark(symbols, idx, initial_cash=10_000.0)
    assert result.name == CASH_BENCHMARK_NAME
    assert np.allclose(result.equity_curve.to_numpy(), 10_000.0)
    assert result.metrics["total_return"] == 0.0
    assert result.metrics["max_drawdown"] == 0.0
    assert result.metrics["annualized_volatility"] == 0.0
    assert result.metrics["turnover"] == 0.0
    assert result.diagnostics["is_benchmark"] == 1.0
    assert result.diagnostics["is_investable_crypto_portfolio"] == 0.0
