from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crypto_hf.data.multi_asset_loader import (
    build_close_price_matrix,
    build_returns_matrix,
    load_multi_asset_ohlcv,
    symbol_to_filename,
)
from helpers import make_synthetic_ohlcv, write_ohlcv_csv


def test_symbol_to_filename() -> None:
    assert symbol_to_filename("BTC/USDT", "1d") == "BTC_USDT_1d.csv"


def test_multi_asset_alignment(tmp_path: Path) -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    write_ohlcv_csv(make_synthetic_ohlcv(30, seed=1), tmp_path / symbol_to_filename(symbols[0], "1d"))
    write_ohlcv_csv(make_synthetic_ohlcv(25, start="2024-01-06", seed=2), tmp_path / symbol_to_filename(symbols[1], "1d"))
    write_ohlcv_csv(make_synthetic_ohlcv(20, start="2024-01-11", seed=3), tmp_path / symbol_to_filename(symbols[2], "1d"))

    result = load_multi_asset_ohlcv(symbols, tmp_path, "1d")
    assert list(result.close_prices.columns) == symbols
    assert result.aligned_rows == 20
    assert result.close_prices.isna().sum().sum() == 0
    assert not result.returns.isna().all().any()


def test_returns_matrix_shape() -> None:
    data = {
        "BTC/USDT": make_synthetic_ohlcv(10, seed=1),
        "ETH/USDT": make_synthetic_ohlcv(10, seed=2),
    }
    close = build_close_price_matrix(data)
    returns = build_returns_matrix(close)
    assert returns.shape == (9, 2)
    assert list(returns.columns) == ["BTC/USDT", "ETH/USDT"]
