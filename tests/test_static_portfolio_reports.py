from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_hf.config import StaticPortfolioConfig
from crypto_hf.data.multi_asset_loader import symbol_to_filename
from crypto_hf.pipeline.static_multi_asset_portfolio import run_static_multi_asset_portfolio_pipeline
from crypto_hf.portfolio.benchmarks import BTC_ONLY_BENCHMARK_NAME, CASH_BENCHMARK_NAME
from crypto_hf.portfolio.optimizers import OptimizerConfig, compute_inverse_volatility_weights
from crypto_hf.portfolio.reports import (
    build_asset_risk_report,
    build_asset_test_performance_report,
    build_train_test_metrics_report,
)
from helpers import make_synthetic_ohlcv, write_ohlcv_csv


def _write_dataset(data_dir: Path, symbols: list[str], n: int = 200) -> None:
    for i, symbol in enumerate(symbols):
        write_ohlcv_csv(
            make_synthetic_ohlcv(n, seed=i + 1, base_price=100 + i * 10),
            data_dir / symbol_to_filename(symbol, "1d"),
        )


def test_static_portfolio_pipeline_reports_and_benchmarks(tmp_path: Path) -> None:
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
        BTC_ONLY_BENCHMARK_NAME,
        CASH_BENCHMARK_NAME,
    }
    assert expected == set(outputs.results)
    assert expected == set(outputs.train_results)

    metrics_dir = tmp_path / "reports" / "metrics"
    figures_dir = tmp_path / "reports" / "figures"
    for name in [
        "static_portfolio_weights.csv",
        "static_portfolio_metrics.csv",
        "static_portfolio_diagnostics.csv",
        "static_portfolio_correlation.csv",
        "static_portfolio_asset_risk.csv",
        "static_portfolio_asset_test_performance.csv",
        "static_portfolio_train_test_metrics.csv",
    ]:
        assert (metrics_dir / name).exists(), name

    for name in [
        "static_portfolio_total_return_ranking.png",
        "static_portfolio_sharpe_ranking.png",
        "static_portfolio_max_drawdown_ranking.png",
    ]:
        assert (figures_dir / name).exists(), name

    asset_risk = pd.read_csv(metrics_dir / "static_portfolio_asset_risk.csv", index_col=0)
    assert set(asset_risk.index) == set(symbols)
    assert (asset_risk["train_annualized_volatility"] > 0).all()
    assert np.isclose(asset_risk["equal_weight"].iloc[0], 1.0 / len(symbols))

    inv = outputs.weights.loc["inverse_volatility"]
    eq = outputs.weights.loc["equal_weight"]
    assert not np.allclose(inv.to_numpy(), eq.to_numpy(), atol=1e-4)

    diagnostics = pd.read_csv(metrics_dir / "static_portfolio_diagnostics.csv", index_col=0)
    assert diagnostics.loc[CASH_BENCHMARK_NAME, "is_benchmark"] == 1.0
    assert diagnostics.loc[CASH_BENCHMARK_NAME, "is_investable_crypto_portfolio"] == 0.0
    assert diagnostics.loc[BTC_ONLY_BENCHMARK_NAME, "is_benchmark"] == 1.0
    assert diagnostics.loc[BTC_ONLY_BENCHMARK_NAME, "is_investable_crypto_portfolio"] == 1.0

    train_test = pd.read_csv(metrics_dir / "static_portfolio_train_test_metrics.csv", index_col=0)
    decay_cols = ["sharpe_decay", "return_decay", "drawdown_change"]
    for col in decay_cols:
        assert col in train_test.columns
    investable = train_test.drop(index=[CASH_BENCHMARK_NAME])
    assert not investable[["train_total_return", "test_total_return"]].isna().any().any()
    assert np.allclose(
        train_test.loc["equal_weight", "sharpe_decay"],
        train_test.loc["equal_weight", "test_sharpe_ratio"]
        - train_test.loc["equal_weight", "train_sharpe_ratio"],
    )
    assert train_test.loc[CASH_BENCHMARK_NAME, "train_total_return"] == 0.0
    assert train_test.loc[CASH_BENCHMARK_NAME, "test_total_return"] == 0.0

    asset_perf = pd.read_csv(metrics_dir / "static_portfolio_asset_test_performance.csv", index_col=0)
    assert len(asset_perf) == len(symbols)
    for col in [
        "test_weight_equal",
        "test_weight_inverse_vol",
        "test_weight_min_variance",
        "test_weight_max_sharpe",
        "test_weight_hrp",
    ]:
        assert col in asset_perf.columns
    weights = pd.read_csv(metrics_dir / "static_portfolio_weights.csv", index_col=0)
    assert asset_perf["test_weight_equal"].equals(weights.loc["equal_weight", symbols])


def test_asset_risk_inverse_vol_weights_match_optimizer() -> None:
    idx = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        {
            "BTC/USDT": rng.normal(0, 0.01, len(idx)),
            "ETH/USDT": rng.normal(0, 0.02, len(idx)),
            "SOL/USDT": rng.normal(0, 0.03, len(idx)),
        },
        index=idx,
    )
    config = OptimizerConfig(max_weight=0.6, annualization_factor=365)
    raw, final, _ = compute_inverse_volatility_weights(returns, config)
    weights = pd.DataFrame({"inverse_volatility": final})
    report = build_asset_risk_report(
        list(returns.columns),
        returns,
        returns.iloc[-20:],
        pd.DataFrame(100.0, index=returns.index, columns=returns.columns),
        weights,
        config,
    )
    for symbol in returns.columns:
        assert report.loc[symbol, "inverse_vol_raw_weight"] == pytest.approx(raw[symbol])
        assert report.loc[symbol, "inverse_vol_final_weight"] == pytest.approx(final[symbol])
