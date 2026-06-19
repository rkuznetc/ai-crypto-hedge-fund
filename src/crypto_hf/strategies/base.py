from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Common interface for signal-generating strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with standardized `signal` and `position` columns."""
