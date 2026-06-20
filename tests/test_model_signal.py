from __future__ import annotations

import pandas as pd
import pytest

from crypto_hf.strategies.model_signal import signals_from_values


def _frame(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"close": range(n)}, index=idx)


def test_strict_alignment_raises_on_missing_signal() -> None:
    data = _frame()
    signals = pd.Series([1.0, 0.0], index=data.index[:2])
    with pytest.raises(ValueError, match="strict_alignment=True"):
        signals_from_values(data, signals, strict_alignment=True)


def test_non_binary_signal_values_raise() -> None:
    data = _frame(3)
    for bad in [0.4, 0.7, 2.0, -1.0]:
        signals = pd.Series([1.0, bad, 0.0], index=data.index)
        with pytest.raises(ValueError, match="binary"):
            signals_from_values(data, signals)


def test_valid_binary_signals_and_shifted_positions() -> None:
    data = _frame(4)
    signals = pd.Series([1.0, 1.0, 0.0, 1.0], index=data.index)
    out = signals_from_values(data, signals)
    pd.testing.assert_series_equal(out["signal"], signals, check_names=False)
    expected_position = signals.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(out["position"], expected_position, check_names=False)


def test_non_strict_alignment_fills_missing_with_cash() -> None:
    data = _frame(3)
    signals = pd.Series([1.0], index=data.index[:1])
    out = signals_from_values(data, signals, strict_alignment=False)
    assert out["signal"].tolist() == [1.0, 0.0, 0.0]
