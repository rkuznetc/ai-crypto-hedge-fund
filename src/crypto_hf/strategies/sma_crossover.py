from __future__ import annotations

import pandas as pd

from crypto_hf.strategies.base import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    """Long-only SMA crossover: long when fast SMA > slow SMA, cash otherwise."""

    name = "sma_crossover"

    def __init__(self, fast_window: int, slow_window: int) -> None:
        if slow_window <= fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        fast_col = f"sma_{self.fast_window}"
        slow_col = f"sma_{self.slow_window}"

        if fast_col not in out.columns or slow_col not in out.columns:
            raise KeyError(
                f"Missing SMA columns {fast_col} / {slow_col}. "
                "Run add_moving_averages before generating signals."
            )

        out["signal"] = (out[fast_col] > out[slow_col]).astype(float)
        # Position shifted by 1 bar to avoid look-ahead bias.
        out["position"] = out["signal"].shift(1).fillna(0.0)
        return out
