from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    strategy_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "returns": self.returns,
            "positions": self.positions,
        }
