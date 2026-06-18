#!/usr/bin/env python3
"""Optional script to download OHLCV data via ccxt (requires internet)."""

from __future__ import annotations

import argparse
from pathlib import Path

import ccxt
import pandas as pd


def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1000,
    exchange_id: str = "binance",
    output: Path = Path("data/raw/BTC_USDT_1d.csv"),
) -> Path:
    """Fetch OHLCV candles and save to CSV."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV via ccxt")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--output", type=Path, default=Path("data/raw/BTC_USDT_1d.csv"))
    args = parser.parse_args()

    path = download_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        exchange_id=args.exchange,
        output=args.output,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
