from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV and return a clean DataFrame indexed by timestamp."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df[REQUIRED_COLUMNS[1:]]
