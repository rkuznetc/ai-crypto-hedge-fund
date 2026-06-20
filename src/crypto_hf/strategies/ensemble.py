from __future__ import annotations

import pandas as pd

from crypto_hf.strategies.base import BaseStrategy
from crypto_hf.strategies.model_signal import signals_from_values


class EnsembleMajorityVoteStrategy(BaseStrategy):
    """Long-only ensemble from precomputed component signals on the same index."""

    name = "ensemble_majority_vote"

    def __init__(
        self,
        component_signals: dict[str, pd.Series],
        min_votes: int,
    ) -> None:
        if min_votes <= 0:
            raise ValueError("min_votes must be positive")
        if not component_signals:
            raise ValueError("component_signals must not be empty")
        self._component_signals = component_signals
        self.min_votes = min_votes

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        aligned: dict[str, pd.Series] = {}
        for name, series in self._component_signals.items():
            reindexed = series.reindex(data.index)
            if reindexed.isna().any():
                missing = int(reindexed.isna().sum())
                raise ValueError(
                    f"Component signal '{name}' missing for {missing} date(s) on test index"
                )
            aligned[name] = reindexed.astype(float)

        vote_count = sum(aligned.values())
        signal = (vote_count >= self.min_votes).astype(float)
        return signals_from_values(data, signal)
