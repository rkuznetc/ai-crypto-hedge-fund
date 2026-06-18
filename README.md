# AI Crypto Hedge Fund MVP

Educational MVP for offline historical backtesting of a single crypto pair (**BTC/USDT**, 1d). This first block provides a modular Python foundation: data layer, baseline strategies, vectorbt backtesting, performance metrics, plots, tests, and a reproducible notebook.

> **Note:** The bundled CSV at `data/raw/BTC_USDT_1d.csv` is **synthetic sample data** generated for offline development. Replace it with real OHLCV (via `scripts/download_ohlcv.py` or your own source) without changing application code.

## Project structure

```
configs/                 # Experiment YAML configs
data/raw/                # OHLCV CSV (offline source of truth)
data/processed/          # Reserved for future processed datasets
notebooks/               # Reproducible analysis notebooks
reports/figures/         # Generated plots
reports/metrics/         # Generated metrics CSV
scripts/                 # CLI entrypoints
src/crypto_hf/           # Core library
tests/                   # pytest suite
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
uv sync --group dev
```

## Run tests

```bash
uv run pytest
```

## Run baseline backtest

```bash
uv run python scripts/run_baseline.py
```

Outputs:

- `reports/metrics/baseline_metrics.csv`
- `reports/figures/*.png`

## Run notebook

```bash
uv run jupyter notebook notebooks/01_baseline_single_asset.ipynb
```

Or execute top-to-bottom in JupyterLab / VS Code. The notebook imports logic from `src/crypto_hf` and should run without internet access.

## Optional: download real OHLCV

Requires internet and optionally exchange credentials in `.env`:

```bash
cp .env.example .env
uv run python scripts/download_ohlcv.py --symbol BTC/USDT --timeframe 1d
```

## Optional: regenerate synthetic sample data

```bash
uv run python scripts/generate_sample_data.py --days 500
```

## Docker (tests only)

```bash
docker build -t crypto-hf .
docker run --rm crypto-hf
```

## Design notes

- **No look-ahead bias:** strategy `position` is shifted by one bar before PnL.
- **Offline-first:** tests and notebook use local CSV only.
- **Modular:** strategies implement `BaseStrategy.generate_signals()`; backtests return `BacktestResult`.

## Roadmap (not in this block)

ML models, econometric models, LLM agents, portfolio optimization, live trading, and multi-asset expansion will build on this foundation in later phases.
