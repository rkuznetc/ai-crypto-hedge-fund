from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.features.model_features import build_model_features, drop_rows_with_missing_features
from crypto_hf.features.targets import make_next_return_target, model_feature_columns
from crypto_hf.pipeline.baseline import split_train_test


def _ohlcv(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.ones(n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )


def test_model_features_are_causal() -> None:
    raw = _ohlcv()
    featured = build_model_features(
        raw,
        return_lags=[1, 2],
        rolling_mean_windows=[3],
        rolling_vol_windows=[3],
        momentum_windows=[3],
        sma_ratio_windows=[3],
        rsi_window=3,
    )
    assert featured["return_lag_1"].iloc[2] == pytest.approx(featured["returns"].iloc[1])


def test_chronological_train_test_split() -> None:
    raw = _ohlcv()
    featured = build_model_features(
        raw,
        return_lags=[1],
        rolling_mean_windows=[3],
        rolling_vol_windows=[3],
        momentum_windows=[3],
        sma_ratio_windows=[3],
        rsi_window=3,
    )
    featured = make_next_return_target(featured).dropna(subset=["next_return"])
    feature_cols = model_feature_columns(featured)
    dataset = drop_rows_with_missing_features(featured, feature_cols)
    train, test = split_train_test(dataset, train_size=0.7)
    assert train.index.max() < test.index.min()
