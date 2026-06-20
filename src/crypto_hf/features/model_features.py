from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.features.technical import add_returns


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_model_features(
    df: pd.DataFrame,
    return_lags: list[int],
    rolling_mean_windows: list[int],
    rolling_vol_windows: list[int],
    momentum_windows: list[int],
    sma_ratio_windows: list[int],
    rsi_window: int,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """Build causal model features for single-asset direction models."""
    out = add_returns(df) if "returns" not in df.columns else df.copy()

    for lag in return_lags:
        out[f"return_lag_{lag}"] = out["returns"].shift(lag)

    for window in rolling_mean_windows:
        out[f"rolling_mean_return_{window}"] = out["returns"].rolling(
            window=window,
            min_periods=window,
        ).mean()

    for window in rolling_vol_windows:
        out[f"rolling_vol_{window}"] = (
            out["returns"].rolling(window=window, min_periods=window).std()
            * np.sqrt(annualization_factor)
        )

    for window in momentum_windows:
        out[f"momentum_{window}"] = out["close"] / out["close"].shift(window) - 1.0

    for window in sma_ratio_windows:
        sma = out["close"].rolling(window=window, min_periods=window).mean()
        out[f"sma_ratio_{window}"] = out["close"] / sma - 1.0

    if "volume" in out.columns:
        out["volume_change"] = out["volume"].pct_change()

    out[f"rsi_{rsi_window}"] = _rsi(out["close"], window=rsi_window)
    return out.replace([np.inf, -np.inf], np.nan)


def drop_rows_with_missing_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_up",
) -> pd.DataFrame:
    """Drop rows with NaN in features or target."""
    required = feature_cols + [target_col, "returns"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for model dataset: {missing}")
    return df.dropna(subset=required).copy()
