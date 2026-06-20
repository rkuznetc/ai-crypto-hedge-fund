from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crypto_hf.config import StaticPortfolioConfig
from crypto_hf.data.multi_asset_loader import load_multi_asset_ohlcv
from crypto_hf.pipeline.baseline import split_train_test
from crypto_hf.portfolio.backtesting import PortfolioBacktestResult, run_static_portfolio_backtest
from crypto_hf.portfolio.benchmarks import (
    BTC_ONLY_BENCHMARK_NAME,
    CASH_BENCHMARK_NAME,
    run_btc_only_benchmark,
    run_cash_benchmark,
)
from crypto_hf.portfolio.optimizers import (
    EqualWeightOptimizer,
    HierarchicalRiskParityOptimizer,
    InverseVolatilityOptimizer,
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    OptimizerConfig,
)
from crypto_hf.portfolio.reports import (
    build_asset_risk_report,
    build_asset_test_performance_report,
    build_train_test_metrics_report,
)
from crypto_hf.visualization.portfolio_plots import (
    plot_correlation_heatmap,
    plot_portfolio_metric_rankings,
    plot_portfolio_weights,
    plot_risk_return_scatter,
)
from crypto_hf.visualization.plots import plot_drawdown_comparison, plot_equity_curve

# Train metrics use the same optimization window as weight selection (portfolio_lookback_days).
TRAIN_METRICS_WINDOW = "optimization_window"


@dataclass
class StaticPortfolioOutputs:
    """Artifacts from the static multi-asset portfolio pipeline."""

    config: StaticPortfolioConfig
    close_prices: pd.DataFrame
    returns: pd.DataFrame
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    weights: pd.DataFrame
    results: dict[str, PortfolioBacktestResult]
    train_results: dict[str, PortfolioBacktestResult]
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    correlation_matrix: pd.DataFrame
    alignment_rows: int


def _build_optimizer_config(config: StaticPortfolioConfig) -> OptimizerConfig:
    return OptimizerConfig(
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        allow_short=config.allow_short,
        risk_free_rate=config.risk_free_rate,
        annualization_factor=config.annualization_factor,
        covariance_method=config.covariance_method,
        expected_return_method=config.expected_return_method,
        ewm_span=config.ewm_span,
    )


def _select_train_window(train_returns: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if len(train_returns) <= lookback_days:
        return train_returns
    return train_returns.iloc[-lookback_days:].copy()


def _backtest_kwargs(config: StaticPortfolioConfig) -> dict[str, float | int]:
    return {
        "initial_cash": config.initial_cash,
        "fee_rate": config.fee_rate,
        "slippage": config.slippage,
        "annualization_factor": config.annualization_factor,
        "risk_free_rate": config.risk_free_rate,
    }


def run_static_multi_asset_portfolio_pipeline(
    config: StaticPortfolioConfig,
    reports_dir: Path = Path("reports"),
) -> StaticPortfolioOutputs:
    """Run static portfolio optimization on train and backtest on test."""
    load_result = load_multi_asset_ohlcv(
        symbols=config.symbols,
        data_dir=config.data_dir,
        timeframe=config.timeframe,
    )
    close_prices = load_result.close_prices
    returns = load_result.returns
    train_returns, test_returns = split_train_test(returns, config.train_size)
    test_close = close_prices.reindex(test_returns.index)
    train_window = _select_train_window(train_returns, config.portfolio_lookback_days)
    train_close = close_prices.reindex(train_window.index)
    correlation_matrix = train_window.corr()

    optimizer_config = _build_optimizer_config(config)
    optimizers = [
        EqualWeightOptimizer(optimizer_config),
        InverseVolatilityOptimizer(optimizer_config),
        MinVarianceOptimizer(optimizer_config),
        MaxSharpeOptimizer(optimizer_config),
        HierarchicalRiskParityOptimizer(optimizer_config),
    ]

    weight_rows: dict[str, pd.Series] = {}
    results: dict[str, PortfolioBacktestResult] = {}
    train_results: dict[str, PortfolioBacktestResult] = {}
    backtest_kwargs = _backtest_kwargs(config)

    for optimizer in optimizers:
        portfolio_weights = optimizer.optimize(train_window)
        weight_rows[optimizer.name] = portfolio_weights.weights
        results[optimizer.name] = run_static_portfolio_backtest(
            name=optimizer.name,
            weights=portfolio_weights.weights,
            test_close_prices=test_close,
            **backtest_kwargs,
        )
        train_results[optimizer.name] = run_static_portfolio_backtest(
            name=optimizer.name,
            weights=portfolio_weights.weights,
            test_close_prices=train_close,
            **backtest_kwargs,
        )

    results[BTC_ONLY_BENCHMARK_NAME] = run_btc_only_benchmark(
        config.symbols,
        test_close,
        **backtest_kwargs,
    )
    train_results[BTC_ONLY_BENCHMARK_NAME] = run_btc_only_benchmark(
        config.symbols,
        train_close,
        **backtest_kwargs,
    )
    weight_rows[BTC_ONLY_BENCHMARK_NAME] = results[BTC_ONLY_BENCHMARK_NAME].weights

    results[CASH_BENCHMARK_NAME] = run_cash_benchmark(
        config.symbols,
        test_close.index,
        initial_cash=config.initial_cash,
    )
    train_results[CASH_BENCHMARK_NAME] = run_cash_benchmark(
        config.symbols,
        train_close.index,
        initial_cash=config.initial_cash,
    )
    weight_rows[CASH_BENCHMARK_NAME] = results[CASH_BENCHMARK_NAME].weights

    weights = pd.DataFrame(weight_rows).T
    metrics = pd.DataFrame({name: result.metrics for name, result in results.items()}).T
    diagnostics = pd.DataFrame({name: result.diagnostics for name, result in results.items()}).T

    asset_risk = build_asset_risk_report(
        config.symbols,
        train_window,
        test_returns,
        close_prices,
        weights,
        optimizer_config,
    )
    asset_test_performance = build_asset_test_performance_report(
        config.symbols,
        test_returns,
        weights,
        config.annualization_factor,
    )
    train_test_metrics = build_train_test_metrics_report(results, train_results)

    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    weights.to_csv(metrics_dir / "static_portfolio_weights.csv")
    metrics.to_csv(metrics_dir / "static_portfolio_metrics.csv")
    diagnostics.to_csv(metrics_dir / "static_portfolio_diagnostics.csv")
    correlation_matrix.to_csv(metrics_dir / "static_portfolio_correlation.csv")
    asset_risk.to_csv(metrics_dir / "static_portfolio_asset_risk.csv")
    asset_test_performance.to_csv(metrics_dir / "static_portfolio_asset_test_performance.csv")
    train_test_metrics.to_csv(metrics_dir / "static_portfolio_train_test_metrics.csv")

    equity_curves = {name: result.equity_curve for name, result in results.items()}
    plot_equity_curve(
        equity_curves,
        title="Static Portfolio Equity Curves (test)",
        save_path=figures_dir / "static_portfolio_equity_curves.png",
    )
    plot_drawdown_comparison(
        equity_curves,
        title="Static Portfolio Drawdowns (test)",
        save_path=figures_dir / "static_portfolio_drawdowns.png",
    )
    plot_portfolio_weights(
        weights.drop(index=[CASH_BENCHMARK_NAME], errors="ignore"),
        title="Static Portfolio Weights",
        save_path=figures_dir / "static_portfolio_weights.png",
    )
    plot_correlation_heatmap(
        correlation_matrix,
        title="Train Return Correlation",
        save_path=figures_dir / "static_portfolio_correlation_heatmap.png",
    )
    plot_risk_return_scatter(
        metrics,
        title="Static Portfolio Risk-Return (test)",
        save_path=figures_dir / "static_portfolio_risk_return_scatter.png",
    )
    plot_portfolio_metric_rankings(
        metrics,
        metric="sharpe_ratio",
        title="Sharpe Ratio Ranking (test)",
        save_path=figures_dir / "static_portfolio_metric_rankings.png",
    )
    plot_portfolio_metric_rankings(
        metrics,
        metric="total_return",
        title="Total Return Ranking (test)",
        save_path=figures_dir / "static_portfolio_total_return_ranking.png",
    )
    plot_portfolio_metric_rankings(
        metrics,
        metric="sharpe_ratio",
        title="Sharpe Ratio Ranking (test)",
        save_path=figures_dir / "static_portfolio_sharpe_ranking.png",
    )
    plot_portfolio_metric_rankings(
        metrics,
        metric="max_drawdown",
        title="Max Drawdown Ranking (test)",
        save_path=figures_dir / "static_portfolio_max_drawdown_ranking.png",
    )

    _close_all_figures()

    return StaticPortfolioOutputs(
        config=config,
        close_prices=close_prices,
        returns=returns,
        train_returns=train_returns,
        test_returns=test_returns,
        weights=weights,
        results=results,
        train_results=train_results,
        metrics=metrics,
        diagnostics=diagnostics,
        correlation_matrix=correlation_matrix,
        alignment_rows=load_result.aligned_rows,
    )


def _close_all_figures() -> None:
    import matplotlib.pyplot as plt

    for fig_num in plt.get_fignums():
        plt.close(fig_num)
