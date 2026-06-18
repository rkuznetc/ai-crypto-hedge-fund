from __future__ import annotations

import pandas as pd
import vectorbt as vbt

from crypto_hf.backtesting.base import BacktestResult
from crypto_hf.metrics.performance import compute_all_metrics


class VectorbtBacktester:
    """VectorBT-based long-only backtest engine."""

    def __init__(self, initial_cash: float, fee_rate: float) -> None:
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate

    def run(
        self,
        prices: pd.Series,
        positions: pd.Series,
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        """Run a long-only backtest using pre-computed position sizes (0 or 1)."""
        aligned_prices = prices.astype(float)
        aligned_positions = positions.reindex(aligned_prices.index).fillna(0.0).astype(float)

        portfolio = vbt.Portfolio.from_signals(
            close=aligned_prices,
            entries=aligned_positions.diff().fillna(aligned_positions.iloc[0]) > 0,
            exits=aligned_positions.diff().fillna(0.0) < 0,
            init_cash=self.initial_cash,
            fees=self.fee_rate,
            freq="1D",
        )

        equity_curve = portfolio.value()
        if isinstance(equity_curve, pd.DataFrame):
            equity_curve = equity_curve.iloc[:, 0]

        returns = equity_curve.pct_change().fillna(0.0)
        trades_df = _extract_trades(portfolio)
        metrics = compute_all_metrics(equity_curve, returns, trades_df, aligned_positions)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=aligned_positions,
            trades=trades_df,
            metrics=metrics,
            strategy_name=strategy_name,
        )


def _extract_trades(portfolio: vbt.Portfolio) -> pd.DataFrame | None:
    trades = portfolio.trades.records_readable
    if trades is None or len(trades) == 0:
        return None
    if isinstance(trades, pd.DataFrame):
        return trades
    return pd.DataFrame(trades)
