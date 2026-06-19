from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.features.targets import (
    TARGET_COLUMNS,
    drop_unknown_target_rows,
    make_next_return_target,
    model_feature_columns,
)


def _price_df(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100, 107, n)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_next_return_uses_future_bar_and_last_row_removed() -> None:
    df = make_next_return_target(_price_df())
    assert df["next_return"].iloc[0] == pytest.approx(df["close"].iloc[1] / df["close"].iloc[0] - 1)
    assert pd.isna(df["next_return"].iloc[-1])

    cleaned = drop_unknown_target_rows(df)
    assert len(cleaned) == len(df) - 1
    assert cleaned["next_return"].isna().sum() == 0


def test_target_up_is_classification_label() -> None:
    df = drop_unknown_target_rows(make_next_return_target(_price_df()))
    assert set(df["target_up"].unique()).issubset({0, 1})


def test_feature_columns_exclude_targets() -> None:
    df = make_next_return_target(_price_df())
    df["return_lag_1"] = df["returns"].shift(1)
    cols = model_feature_columns(df)
    for target_col in TARGET_COLUMNS:
        assert target_col not in cols
