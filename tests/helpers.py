from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    n: int,
    start: str = "2024-01-01",
    *,
    seed: int = 0,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Build valid daily OHLCV data indexed by UTC timestamp."""
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    close = base_price + np.cumsum(rng.normal(0.0, 1.0, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def write_ohlcv_csv(df: pd.DataFrame, path: Path) -> None:
    """Write OHLCV DataFrame to CSV with timestamp column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "timestamp"
    out.reset_index().to_csv(path, index=False)
