from __future__ import annotations

import pandas as pd

from crypto_hf.strategies.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """Long-only buy-and-hold benchmark."""

    name = "buy_and_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["signal"] = 1.0
        out["position"] = out["signal"].shift(1).fillna(0.0)
        return out
