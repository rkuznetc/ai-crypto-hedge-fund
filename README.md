# AI Crypto Hedge Fund MVP

Educational MVP for offline historical backtesting of a single crypto pair (**BTC/USDT**, 1d). This project provides data validation, baseline strategies, econometric/ML model strategies, VectorBT backtesting, performance metrics, plots, tests, and reproducible notebooks.

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

## Run single-asset model comparison

```bash
uv run python scripts/run_single_asset_models.py
```

Compares buy-and-hold, SMA crossover, econometric AR, and ML direction models on the same test period.

Outputs:

- `reports/metrics/single_asset_model_metrics.csv`
- `reports/metrics/single_asset_model_classification.csv`
- `reports/figures/single_asset_model_*.png`

## Run notebooks

```bash
uv run jupyter notebook notebooks/01_baseline_single_asset.ipynb
uv run jupyter notebook notebooks/02_single_asset_models.ipynb
```

Notebooks import logic from `src/crypto_hf` and run top-to-bottom against the bundled CSV.

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
- **Model strategies:** AR return model (`statsmodels`) and sklearn direction classifiers share the same backtester and test-period evaluation.
- **Modular:** strategies implement `BaseStrategy.generate_signals()`; backtests return `BacktestResult`.

## Roadmap (not in this block)

LLM agents, portfolio optimization, dynamic rebalancing, live trading, and multi-asset expansion.
