from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from crypto_hf.backtesting.base import BacktestResult
from crypto_hf.metrics.performance import compute_all_metrics

BINARY_POSITIONS_MSG = (
    "VectorbtBacktester supports only binary long-only positions: 0.0 or 1.0"
)


def positions_to_entries_exits(positions: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Map target positions to entry/exit signals for vectorbt.

    Expected upstream semantics:
        signal[t] -> position[t+1] -> entry/exit at close[t+1]

    ``positions`` must already be lagged relative to ``signal`` (shifted by one bar).
    Entries fire when position turns 1; exits when it turns 0. vectorbt executes
    at the close of the same bar where the entry/exit flag is True.
    """
    prev_position = positions.shift(1).fillna(0.0)
    entries = (positions == 1.0) & (prev_position == 0.0)
    exits = (positions == 0.0) & (prev_position == 1.0)
    return entries, exits


def _align_positions(
    prices: pd.Series,
    positions: pd.Series,
    strict_alignment: bool,
) -> pd.Series:
    """Align positions to the price index."""
    aligned = positions.reindex(prices.index)
    if strict_alignment:
        if aligned.isna().any():
            missing_count = int(aligned.isna().sum())
            raise ValueError(
                f"Positions missing for {missing_count} price date(s); "
                "strict_alignment=True requires a position on every bar"
            )
        return aligned.astype(float)

    return aligned.fillna(0.0).astype(float)


def _validate_binary_positions(positions: pd.Series) -> None:
    """Reject non-binary or NaN position values."""
    if positions.isna().any():
        raise ValueError(BINARY_POSITIONS_MSG)

    values = positions.to_numpy(dtype=float)
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(BINARY_POSITIONS_MSG)


class VectorbtBacktester:
    """VectorBT-based long-only backtest engine."""

    def __init__(
        self,
        initial_cash: float,
        fee_rate: float,
        annualization_factor: int = 365,
        slippage: float = 0.0,
        strict_alignment: bool = True,
    ) -> None:
        if slippage < 0:
            raise ValueError("slippage must be >= 0")
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.annualization_factor = annualization_factor
        self.slippage = slippage
        self.strict_alignment = strict_alignment

    def run(
        self,
        prices: pd.Series,
        positions: pd.Series,
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        """Run a long-only backtest using pre-computed position sizes (0 or 1)."""
        aligned_prices = prices.astype(float)
        aligned_positions = _align_positions(
            aligned_prices,
            positions,
            self.strict_alignment,
        )
        _validate_binary_positions(aligned_positions)
        entries, exits = positions_to_entries_exits(aligned_positions)

        portfolio = vbt.Portfolio.from_signals(
            close=aligned_prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.fee_rate,
            slippage=self.slippage,
            freq="1D",
        )

        equity_curve = portfolio.value()
        if isinstance(equity_curve, pd.DataFrame):
            equity_curve = equity_curve.iloc[:, 0]

        returns = equity_curve.pct_change().fillna(0.0)
        trades_df = _extract_trades(portfolio)
        metrics = compute_all_metrics(
            equity_curve,
            returns,
            trades_df,
            aligned_positions,
            periods_per_year=self.annualization_factor,
        )

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
