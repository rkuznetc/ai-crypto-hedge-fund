from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_hf.metrics.performance import (
    annualized_volatility,
    calmar_ratio,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


@dataclass
class PortfolioBacktestResult:
    """Static multi-asset portfolio backtest outputs."""

    name: str
    weights: pd.Series
    equity_curve: pd.Series
    returns: pd.Series
    asset_values: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


def run_static_portfolio_backtest(
    name: str,
    weights: pd.Series,
    test_close_prices: pd.DataFrame,
    *,
    initial_cash: float,
    fee_rate: float,
    slippage: float,
    annualization_factor: int = 365,
    risk_free_rate: float = 0.0,
) -> PortfolioBacktestResult:
    """Backtest buy-and-hold static weights on the test period."""
    assets = weights.index.tolist()
    prices = test_close_prices[assets].astype(float)
    if prices.empty:
        raise ValueError("Test close prices are empty")

    entry_prices = prices.iloc[0]
    investable = weights * initial_cash * (1.0 - fee_rate)
    entry_prices_with_slippage = entry_prices * (1.0 + slippage)
    shares = investable / entry_prices_with_slippage

    asset_values = prices.mul(shares, axis=1)
    equity_curve = asset_values.sum(axis=1)
    returns = equity_curve.pct_change().fillna(0.0)

    metrics = {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, annualization_factor),
        "annualized_volatility": annualized_volatility(returns, annualization_factor),
        "sharpe_ratio": sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=annualization_factor,
        ),
        "sortino_ratio": sortino_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=annualization_factor,
        ),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve, annualization_factor),
        "turnover": float(weights.abs().sum()),
    }
    metrics["var_95"] = _value_at_risk(returns, 0.05)
    metrics["cvar_95"] = _conditional_value_at_risk(returns, 0.05)

    asset_returns = prices.pct_change().dropna()
    asset_vols = asset_returns.std()
    weighted_vol = float((weights * asset_vols).sum())
    port_vol = float(returns.std())
    diversification_ratio = weighted_vol / port_vol if port_vol > 0 else 0.0

    hhi = float((weights**2).sum())
    diagnostics = {
        "concentration_hhi": hhi,
        "effective_number_of_assets": 1.0 / hhi if hhi > 0 else 0.0,
        "initial_turnover": float(weights.abs().sum()),
        "max_asset_weight": float(weights.max()),
        "min_asset_weight": float(weights.min()),
        "diversification_ratio": diversification_ratio,
    }

    return PortfolioBacktestResult(
        name=name,
        weights=weights,
        equity_curve=equity_curve,
        returns=returns,
        asset_values=asset_values,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _value_at_risk(returns: pd.Series, alpha: float) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(np.quantile(clean, alpha))


def _conditional_value_at_risk(returns: pd.Series, alpha: float) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    var = np.quantile(clean, alpha)
    tail = clean[clean <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())
