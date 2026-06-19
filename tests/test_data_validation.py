from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.data.validation import DataValidationError, validate_ohlcv


def _valid_df(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n) * 100,
        },
        index=idx,
    )


def test_valid_data_passes() -> None:
    result = validate_ohlcv(_valid_df(), timeframe="1d")
    assert result.row_count == 10


def test_missing_columns_raises() -> None:
    df = _valid_df().drop(columns=["volume"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_ohlcv(df)


def test_duplicate_timestamps_raises() -> None:
    df = _valid_df()
    duplicate_row = df.iloc[[1]].copy()
    df = pd.concat([df, duplicate_row]).sort_index()
    with pytest.raises(DataValidationError, match="Duplicate timestamps"):
        validate_ohlcv(df)


def test_negative_prices_raises() -> None:
    df = _valid_df()
    df.loc[df.index[0], "close"] = -1
    with pytest.raises(DataValidationError, match="Non-positive"):
        validate_ohlcv(df)


def test_timestamp_frequency_mismatch_raises() -> None:
    df = _valid_df()
    df = df.iloc[[0, 2, 3, 4, 5, 6, 7, 8, 9]]
    with pytest.raises(DataValidationError, match="Timestamp frequency mismatch"):
        validate_ohlcv(df, timeframe="1d")


def test_high_below_open_raises() -> None:
    df = _valid_df()
    df.loc[df.index[0], "high"] = df.loc[df.index[0], "open"] - 1
    with pytest.raises(DataValidationError, match="high is below open or close"):
        validate_ohlcv(df)


def test_low_above_close_raises() -> None:
    df = _valid_df()
    df.loc[df.index[0], "low"] = df.loc[df.index[0], "close"] + 1
    with pytest.raises(DataValidationError, match="low is above open or close"):
        validate_ohlcv(df)
