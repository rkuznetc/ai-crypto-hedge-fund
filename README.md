# AI Crypto Hedge Fund MVP

Educational MVP for offline historical backtesting of a single crypto pair (**BTC/USDT**, 1d). This block provides a modular Python foundation: data validation, baseline strategies, VectorBT backtesting, performance metrics, plots, tests, and a reproducible notebook.

Bundled OHLCV data lives at `data/raw/BTC_USDT_1d.csv` (tracked in git). Replace or refresh it via `scripts/download_ohlcv.py` without changing application code.

## Project structure

```
configs/                 # Experiment YAML configs
data/raw/                # OHLCV CSV (offline source of truth)
data/processed/          # Reserved for future processed datasets
notebooks/               # Reproducible analysis notebooks
reports/figures/         # Generated plots (not in git)
reports/metrics/         # Generated metrics CSV (not in git)
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

## Data

CSV schema: `timestamp`, `open`, `high`, `low`, `close`, `volume` (UTC, daily bars for `1d`).

To download fresh OHLCV (optional, requires internet):

```bash
cp .env.example .env
uv run python scripts/download_ohlcv.py --symbol BTC/USDT --timeframe 1d
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

The notebook imports logic from `src/crypto_hf` and runs top-to-bottom against the bundled CSV.

## Docker (tests only)

```bash
docker build -t crypto-hf .
docker run --rm crypto-hf
```

> **Note:** The first `docker build` can take **5–10 minutes**. During `uv sync` the console may show little output while large packages (vectorbt, scipy, numba, pandas) are downloaded — this is expected, not a hang. Subsequent rebuilds are fast when Docker layer cache is warm. The image installs the lightweight `test` group (pytest only), not Jupyter.

## Design notes

- **Crypto annualization:** `annualization_factor: 365` in `configs/baseline.yaml` drives CAGR, Sharpe, rolling volatility, etc.
- **No look-ahead bias:** `signal[t] → position[t+1] → entry at close[t+1]`.
- **Backtest hardening:** `VectorbtBacktester` rejects non-binary positions (`0.0`/`1.0`), enforces strict index alignment by default, and supports configurable `slippage`.
- **Modular:** strategies implement `BaseStrategy.generate_signals()`; backtests return `BacktestResult`.

## Roadmap (not in this block)

ML models, econometric models, LLM agents, portfolio optimization, live trading, and multi-asset expansion will build on this foundation in later phases.
