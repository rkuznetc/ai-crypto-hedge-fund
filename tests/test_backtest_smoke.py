from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy


def test_backtest_smoke() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    close = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.001, 0.01, 50))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(50) * 1000,
        },
        index=idx,
    )

    signals = BuyAndHoldStrategy().generate_signals(df)
    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.001).run(
        df["close"],
        signals["position"],
        strategy_name="buy_and_hold",
    )

    assert not result.equity_curve.empty
    assert not result.returns.empty
    assert "total_return" in result.metrics
    assert "sharpe_ratio" in result.metrics
