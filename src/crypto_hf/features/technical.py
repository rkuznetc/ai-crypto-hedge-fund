from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add simple percentage returns (no look-ahead)."""
    out = df.copy()
    out["returns"] = out[price_col].pct_change()
    return out


def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add log returns (no look-ahead)."""
    out = df.copy()
    out["log_returns"] = np.log(out[price_col] / out[price_col].shift(1))
    return out


def add_moving_averages(
    df: pd.DataFrame,
    windows: list[int],
    price_col: str = "close",
) -> pd.DataFrame:
    """Add simple moving averages for given window sizes."""
    out = df.copy()
    for window in windows:
        out[f"sma_{window}"] = out[price_col].rolling(window=window, min_periods=window).mean()
    return out


def add_rolling_volatility(
    df: pd.DataFrame,
    window: int,
    annualization_factor: int = 365,
    returns_col: str = "returns",
) -> pd.DataFrame:
    """Add rolling annualized volatility from daily returns."""
    out = df.copy()
    if returns_col not in out.columns:
        out = add_returns(out)
    out[f"volatility_{window}"] = (
        out[returns_col].rolling(window=window, min_periods=window).std()
        * np.sqrt(annualization_factor)
    )
    return out


def add_drawdown(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add drawdown series from running peak (no look-ahead)."""
    out = df.copy()
    running_max = out[price_col].cummax()
    out["drawdown"] = out[price_col] / running_max - 1.0
    return out
