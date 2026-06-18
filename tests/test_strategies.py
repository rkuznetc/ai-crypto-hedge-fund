from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy
from crypto_hf.strategies.sma_crossover import SMACrossoverStrategy


def _ohlcv_with_sma(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100, 140, n) + np.sin(np.linspace(0, 6, n)) * 5
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.ones(n) * 100,
            "sma_3": pd.Series(close).rolling(3).mean().values,
            "sma_5": pd.Series(close).rolling(5).mean().values,
        },
        index=idx,
    )
    return df


def test_buy_and_hold_positions() -> None:
    df = _ohlcv_with_sma(10)
    signals = BuyAndHoldStrategy().generate_signals(df)
    assert signals["signal"].iloc[-1] == 1.0
    assert signals["position"].iloc[0] == 0.0
    assert (signals["position"].iloc[1:] == 1.0).all()


def test_sma_crossover_signals() -> None:
    df = _ohlcv_with_sma()
    strategy = SMACrossoverStrategy(fast_window=3, slow_window=5)
    signals = strategy.generate_signals(df)
    expected = (df["sma_3"] > df["sma_5"]).astype(float)
    pd.testing.assert_series_equal(signals["signal"], expected, check_names=False)


def test_position_shifted_by_one_bar() -> None:
    df = _ohlcv_with_sma()
    strategy = SMACrossoverStrategy(fast_window=3, slow_window=5)
    signals = strategy.generate_signals(df)
    expected_position = signals["signal"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(signals["position"], expected_position, check_names=False)
