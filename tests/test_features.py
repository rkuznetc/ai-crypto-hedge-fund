from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.features.technical import (
    add_moving_averages,
    add_returns,
    add_rolling_volatility,
)


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 110.0, 105.0, 115.0, 120.0], index=idx)
    return pd.DataFrame({"close": close}, index=idx)


def test_returns_known_values() -> None:
    df = _sample_df()
    out = add_returns(df)
    assert np.isnan(out["returns"].iloc[0])
    assert out["returns"].iloc[1] == pytest.approx(0.1)


def test_sma_known_values() -> None:
    df = _sample_df()
    out = add_moving_averages(df, windows=[2])
    assert np.isnan(out["sma_2"].iloc[0])
    assert out["sma_2"].iloc[1] == pytest.approx(105.0)
    assert out["sma_2"].iloc[4] == pytest.approx(117.5)


def test_rolling_volatility_shape_and_nan() -> None:
    df = _sample_df()
    out = add_returns(df)
    out = add_rolling_volatility(out, window=3)
    assert len(out) == len(df)
    assert out["volatility_3"].iloc[:3].isna().all()
    assert out["volatility_3"].iloc[3:].notna().all()
