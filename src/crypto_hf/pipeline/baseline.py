from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crypto_hf.backtesting.base import BacktestResult
from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.config import BaselineConfig
from crypto_hf.data.loader import load_ohlcv_csv
from crypto_hf.data.validation import ValidationResult, validate_ohlcv
from crypto_hf.features.technical import (
    add_drawdown,
    add_log_returns,
    add_moving_averages,
    add_returns,
    add_rolling_volatility,
)
from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy
from crypto_hf.strategies.sma_crossover import SMACrossoverStrategy
from crypto_hf.visualization.plots import (
    export_metrics_table,
    plot_drawdown,
    plot_equity_curve,
    plot_metrics_table,
    plot_price_with_sma,
)


@dataclass
class BaselineOutputs:
    """Artifacts produced by the baseline backtest pipeline."""

    config: BaselineConfig
    validation: ValidationResult
    features: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    results: dict[str, BacktestResult]
    metrics_table: pd.DataFrame


def split_train_test(df: pd.DataFrame, train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    split_idx = int(len(df) * train_size)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("train_size produces an empty train or test split")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_features(df: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    """Compute baseline technical features without look-ahead."""
    windows = sorted({config.fast_window, config.slow_window})
    featured = add_returns(df)
    featured = add_log_returns(featured)
    featured = add_moving_averages(featured, windows=windows)
    featured = add_rolling_volatility(
        featured,
        window=config.volatility_window,
        annualization_factor=config.annualization_factor,
    )
    featured = add_drawdown(featured)
    return featured


def run_backtests(
    test_data: pd.DataFrame,
    config: BaselineConfig,
) -> dict[str, BacktestResult]:
    """Run buy-and-hold and SMA crossover on the test split."""
    prices = test_data["close"]
    engine = VectorbtBacktester(
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        annualization_factor=config.annualization_factor,
        slippage=config.slippage,
    )

    buy_hold = BuyAndHoldStrategy()
    bh_signals = buy_hold.generate_signals(test_data)
    bh_result = engine.run(prices, bh_signals["position"], strategy_name=buy_hold.name)

    sma = SMACrossoverStrategy(config.fast_window, config.slow_window)
    sma_signals = sma.generate_signals(test_data)
    sma_result = engine.run(prices, sma_signals["position"], strategy_name=sma.name)

    return {buy_hold.name: bh_result, sma.name: sma_result}


def run_baseline_pipeline(
    config: BaselineConfig,
    reports_dir: Path = Path("reports"),
) -> BaselineOutputs:
    """Execute the full baseline workflow."""
    raw = load_ohlcv_csv(config.data_path)
    validation = validate_ohlcv(raw, timeframe=config.timeframe)
    features = build_features(raw, config)
    train, test = split_train_test(features, config.train_size)
    results = run_backtests(test, config)

    metrics_by_strategy = {name: result.metrics for name, result in results.items()}
    metrics_table = export_metrics_table(metrics_by_strategy)

    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    export_metrics_table(metrics_by_strategy, metrics_dir / "baseline_metrics.csv")
    plot_price_with_sma(
        test,
        config.fast_window,
        config.slow_window,
        title=f"{config.symbol} — Price with SMA (test)",
        save_path=figures_dir / "price_with_sma.png",
    )
    plot_equity_curve(
        {name: result.equity_curve for name, result in results.items()},
        title="Equity Curves (test)",
        save_path=figures_dir / "equity_curves.png",
    )
    plot_drawdown(
        results["sma_crossover"].equity_curve,
        title="SMA Crossover Drawdown (test)",
        save_path=figures_dir / "drawdown_sma_crossover.png",
    )
    plot_metrics_table(
        metrics_by_strategy,
        save_path=figures_dir / "metrics_table.png",
    )

    for fig_num in plt_figures():
        plt_close(fig_num)

    return BaselineOutputs(
        config=config,
        validation=validation,
        features=features,
        train=train,
        test=test,
        results=results,
        metrics_table=metrics_table,
    )


def plt_figures():
    import matplotlib.pyplot as plt

    return plt.get_fignums()


def plt_close(num: int) -> None:
    import matplotlib.pyplot as plt

    plt.close(num)
