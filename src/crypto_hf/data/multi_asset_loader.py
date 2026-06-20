from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crypto_hf.data.loader import load_ohlcv_csv
from crypto_hf.data.validation import validate_ohlcv


@dataclass
class MultiAssetLoadResult:
    """Aligned multi-asset OHLCV and derived price/return matrices."""

    data_by_symbol: dict[str, pd.DataFrame]
    close_prices: pd.DataFrame
    returns: pd.DataFrame
    aligned_rows: int
    rows_before_alignment: int


def symbol_to_filename(symbol: str, timeframe: str) -> str:
    """Map exchange symbol to CSV filename."""
    base_quote = symbol.replace("/", "_")
    return f"{base_quote}_{timeframe}.csv"


def load_multi_asset_ohlcv(
    symbols: list[str],
    data_dir: str | Path,
    timeframe: str,
) -> MultiAssetLoadResult:
    """Load and inner-align OHLCV for multiple symbols."""
    if len(symbols) < 2:
        raise ValueError("At least two symbols are required for multi-asset loading")

    data_dir = Path(data_dir)
    data_by_symbol: dict[str, pd.DataFrame] = {}
    close_frames: list[pd.Series] = []

    for symbol in symbols:
        path = data_dir / symbol_to_filename(symbol, timeframe)
        raw = load_ohlcv_csv(path)
        validate_ohlcv(raw, timeframe=timeframe)
        data_by_symbol[symbol] = raw
        close_frames.append(raw["close"].rename(symbol))

    close_prices = pd.concat(close_frames, axis=1, join="inner").sort_index()
    rows_before = max(len(df) for df in data_by_symbol.values())
    aligned_rows = len(close_prices)
    if aligned_rows == 0:
        raise ValueError("No overlapping timestamps after inner alignment")

    aligned_data = {
        symbol: data_by_symbol[symbol].reindex(close_prices.index)
        for symbol in symbols
    }
    returns = build_returns_matrix(close_prices)

    return MultiAssetLoadResult(
        data_by_symbol=aligned_data,
        close_prices=close_prices,
        returns=returns,
        aligned_rows=aligned_rows,
        rows_before_alignment=rows_before,
    )


def build_close_price_matrix(data_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build aligned close price matrix from per-symbol DataFrames."""
    frames = [df["close"].rename(symbol) for symbol, df in data_by_symbol.items()]
    return pd.concat(frames, axis=1, join="inner").sort_index()


def build_returns_matrix(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns for all assets."""
    returns = close_prices.pct_change()
    return returns.iloc[1:].copy()
