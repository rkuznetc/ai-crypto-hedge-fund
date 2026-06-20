from __future__ import annotations

from pathlib import Path

from crypto_hf.config import StaticPortfolioConfig
from crypto_hf.data.multi_asset_loader import symbol_to_filename
from crypto_hf.pipeline.static_multi_asset_portfolio import run_static_multi_asset_portfolio_pipeline
from helpers import make_synthetic_ohlcv, write_ohlcv_csv


def _write_dataset(data_dir: Path, symbols: list[str], n: int = 200) -> None:
    for i, symbol in enumerate(symbols):
        write_ohlcv_csv(
            make_synthetic_ohlcv(n, seed=i + 1, base_price=100 + i * 10),
            data_dir / symbol_to_filename(symbol, "1d"),
        )


def test_static_portfolio_pipeline_smoke(tmp_path: Path) -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    data_dir = tmp_path / "raw"
    _write_dataset(data_dir, symbols)

    config = StaticPortfolioConfig(
        symbols=symbols,
        data_dir=data_dir,
        train_size=0.7,
        portfolio_lookback_days=60,
        max_weight=0.5,
    )
    outputs = run_static_multi_asset_portfolio_pipeline(
        config,
        reports_dir=tmp_path / "reports",
    )

    expected = {
        "equal_weight",
        "inverse_volatility",
        "min_variance",
        "max_sharpe",
        "hrp",
    }
    assert expected == set(outputs.results)
    assert outputs.weights.shape[0] == len(expected)
    assert outputs.metrics.shape[0] == len(expected)

    metrics_dir = tmp_path / "reports" / "metrics"
    figures_dir = tmp_path / "reports" / "figures"
    for name in [
        "static_portfolio_weights.csv",
        "static_portfolio_metrics.csv",
        "static_portfolio_diagnostics.csv",
        "static_portfolio_correlation.csv",
    ]:
        assert (metrics_dir / name).exists()
    for name in [
        "static_portfolio_equity_curves.png",
        "static_portfolio_drawdowns.png",
        "static_portfolio_weights.png",
        "static_portfolio_correlation_heatmap.png",
        "static_portfolio_risk_return_scatter.png",
    ]:
        assert (figures_dir / name).exists()
