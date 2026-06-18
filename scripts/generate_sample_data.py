#!/usr/bin/env python3
"""Generate synthetic BTC/USDT daily OHLCV for offline development."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_days: int = 500,
    seed: int = 42,
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """Create realistic-looking synthetic daily OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D", tz="UTC")

    daily_returns = rng.normal(loc=0.0005, scale=0.02, size=n_days)
    close = start_price * np.cumprod(1 + daily_returns)

    open_ = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    volume = np.empty(n_days)

    open_[0] = start_price
    for i in range(n_days):
        if i > 0:
            open_[i] = close[i - 1] * (1 + rng.normal(0, 0.002))
        spread = abs(rng.normal(0.01, 0.005)) * close[i]
        high[i] = max(open_[i], close[i]) + spread * rng.uniform(0.2, 0.8)
        low[i] = min(open_[i], close[i]) - spread * rng.uniform(0.2, 0.8)
        volume[i] = rng.uniform(500, 5000)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV CSV")
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/raw/BTC_USDT_1d.csv"))
    args = parser.parse_args()

    df = generate_synthetic_ohlcv(n_days=args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
