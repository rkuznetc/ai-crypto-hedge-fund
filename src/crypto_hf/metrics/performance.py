from __future__ import annotations

import numpy as np
import pandas as pd

CRYPTO_PERIODS_PER_YEAR = 365


def total_return(equity_curve: pd.Series) -> float:
    """Total return (ROI) over the period."""
    if equity_curve.empty:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    if start == 0:
        return 0.0
    return end / start - 1.0


def cagr(equity_curve: pd.Series, periods_per_year: int = CRYPTO_PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    years = (len(equity_curve) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    roi = total_return(equity_curve)
    return (1.0 + roi) ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    """Annualized volatility from periodic returns."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    std = float(clean.std())
    if np.isnan(std):
        return 0.0
    return std * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    vol = annualized_volatility(clean, periods_per_year)
    if vol == 0:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    mean_excess = float(excess.mean())
    if np.isnan(mean_excess):
        return 0.0
    return mean_excess / float(clean.std()) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio using downside deviation."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_std = float(downside.std())
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean()) / downside_std * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative fraction."""
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = CRYPTO_PERIODS_PER_YEAR) -> float:
    """Calmar ratio: CAGR / |max drawdown|."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return cagr(equity_curve, periods_per_year) / abs(mdd)


def number_of_trades(trades: pd.DataFrame | None) -> int:
    """Count closed trades if trade records are available."""
    if trades is None or trades.empty:
        return 0
    return len(trades)


def turnover(positions: pd.Series) -> float:
    """Sum of absolute position changes (proxy for turnover)."""
    if positions.empty:
        return 0.0
    return float(positions.diff().abs().fillna(0.0).sum())


def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    positions: pd.Series | None = None,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
) -> dict[str, float]:
    """Compute full performance metric set."""
    pos = positions if positions is not None else pd.Series(dtype=float)
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve, periods_per_year),
        "number_of_trades": float(number_of_trades(trades)),
        "turnover": turnover(pos),
    }
