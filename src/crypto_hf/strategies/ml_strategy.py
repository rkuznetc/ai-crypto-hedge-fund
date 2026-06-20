from __future__ import annotations

import pandas as pd

from crypto_hf.strategies.base import BaseStrategy
from crypto_hf.strategies.model_signal import signals_from_values


class EconometricSignalStrategy(BaseStrategy):
    """Long-only strategy from econometric return forecasts."""

    name = "econometric_autoreg"

    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return signals_from_values(data, self._signals)


class MLSignalStrategy(BaseStrategy):
    """Long-only strategy from ML direction probabilities."""

    def __init__(self, signals: pd.Series, name: str = "ml_logistic_regression") -> None:
        self._signals = signals
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return signals_from_values(data, self._signals)
