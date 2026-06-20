#!/usr/bin/env python3
"""Download OHLCV CSV files for all symbols in static portfolio config."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_hf.config import load_static_portfolio_config
from crypto_hf.data.multi_asset_loader import symbol_to_filename


def _load_download_ohlcv():
    module_path = PROJECT_ROOT / "scripts" / "download_ohlcv.py"
    spec = importlib.util.spec_from_file_location("download_ohlcv", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load download helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.download_ohlcv


def main() -> None:
    download_ohlcv = _load_download_ohlcv()
    config_path = PROJECT_ROOT / "configs" / "static_multi_asset_portfolio.yaml"
    config = load_static_portfolio_config(config_path)
    for symbol in config.symbols:
        output = config.data_dir / symbol_to_filename(symbol, config.timeframe)
        path = download_ohlcv(
            symbol=symbol,
            timeframe=config.timeframe,
            output=output,
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
