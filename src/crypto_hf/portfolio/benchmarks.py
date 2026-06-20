from __future__ import annotations

import pandas as pd

from crypto_hf.portfolio.backtesting import PortfolioBacktestResult, run_static_portfolio_backtest


BTC_SYMBOL = "BTC/USDT"
CASH_BENCHMARK_NAME = "cash"
BTC_ONLY_BENCHMARK_NAME = "btc_only"


def build_btc_only_weights(symbols: list[str]) -> pd.Series:
    """Build 100% BTC static weights."""
    if BTC_SYMBOL not in symbols:
        raise ValueError(f"{BTC_SYMBOL} must be in config symbols for btc_only benchmark")
    weights = pd.Series(0.0, index=symbols)
    weights[BTC_SYMBOL] = 1.0
    return weights


def run_btc_only_benchmark(
    symbols: list[str],
    test_close_prices: pd.DataFrame,
    *,
    initial_cash: float,
    fee_rate: float,
    slippage: float,
    annualization_factor: int,
    risk_free_rate: float,
) -> PortfolioBacktestResult:
    """Static 100% BTC buy-and-hold benchmark on test."""
    weights = build_btc_only_weights(symbols)
    result = run_static_portfolio_backtest(
        name=BTC_ONLY_BENCHMARK_NAME,
        weights=weights,
        test_close_prices=test_close_prices,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        annualization_factor=annualization_factor,
        risk_free_rate=risk_free_rate,
    )
    result.diagnostics.update(_benchmark_flags(is_benchmark=True, is_investable=True))
    return result


def run_cash_benchmark(
    symbols: list[str],
    index: pd.Index,
    *,
    initial_cash: float,
) -> PortfolioBacktestResult:
    """Flat cash benchmark with zero exposure."""
    equity_curve = pd.Series(float(initial_cash), index=index)
    returns = pd.Series(0.0, index=index)
    asset_values = pd.DataFrame(0.0, index=index, columns=symbols)
    weights = pd.Series(0.0, index=symbols)

    metrics = {
        "total_return": 0.0,
        "cagr": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "turnover": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
    }
    diagnostics = {
        "concentration_hhi": 0.0,
        "effective_number_of_assets": 0.0,
        "initial_turnover": 0.0,
        "max_asset_weight": 0.0,
        "min_asset_weight": 0.0,
        "diversification_ratio": 0.0,
        "largest_weight_symbol": "",
        "number_of_assets_with_weight_gt_5pct": 0.0,
        "number_of_assets_with_weight_gt_10pct": 0.0,
        **_benchmark_flags(is_benchmark=True, is_investable=False),
    }

    return PortfolioBacktestResult(
        name=CASH_BENCHMARK_NAME,
        weights=weights,
        equity_curve=equity_curve,
        returns=returns,
        asset_values=asset_values,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _benchmark_flags(*, is_benchmark: bool, is_investable: bool) -> dict[str, float]:
    return {
        "is_benchmark": float(is_benchmark),
        "is_investable_crypto_portfolio": float(is_investable),
    }
