from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.strategies.base import BaseStrategy
from crypto_hf.strategies.model_signal import signals_from_values


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Long-only z-score mean reversion on close."""

    name = "stat_zscore_mean_reversion"

    def __init__(self, window: int, entry_threshold: float, exit_threshold: float) -> None:
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        rolling_mean = out["close"].rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = out["close"].rolling(window=self.window, min_periods=self.window).std()
        z_score = (out["close"] - rolling_mean) / rolling_std

        signal = pd.Series(0.0, index=out.index)
        long_state = False
        for idx in out.index:
            z = z_score.loc[idx]
            if pd.isna(z):
                signal.loc[idx] = 0.0
                long_state = False
                continue
            if not long_state and z < self.entry_threshold:
                long_state = True
            elif long_state and z > self.exit_threshold:
                long_state = False
            signal.loc[idx] = 1.0 if long_state else 0.0

        return signals_from_values(out, signal)


class MomentumBreakoutStrategy(BaseStrategy):
    """Long-only breakout above rolling high."""

    name = "stat_momentum_breakout"

    def __init__(self, window: int, momentum_threshold: float = 0.0) -> None:
        self.window = window
        self.momentum_threshold = momentum_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        rolling_high = out["close"].rolling(window=self.window, min_periods=self.window).max().shift(1)
        momentum = out["close"] / out["close"].shift(self.window) - 1.0
        breakout = out["close"] > rolling_high
        signal = ((breakout) | (momentum > self.momentum_threshold)).astype(float)
        signal = signal.where(rolling_high.notna() & momentum.notna(), 0.0)
        return signals_from_values(out, signal)


class VolatilityRegimeFilterStrategy(BaseStrategy):
    """Long-only when rolling volatility is below a train/validation-calibrated threshold."""

    name = "stat_volatility_regime_filter"

    def __init__(
        self,
        window: int,
        threshold: float,
        annualization_factor: int = 365,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.annualization_factor = annualization_factor

    @staticmethod
    def calibrate_threshold(
        data: pd.DataFrame,
        window: int,
        annualization_factor: int,
        use_quantile: bool,
        fixed_threshold: float,
        quantile: float,
    ) -> float:
        """Calibrate volatility threshold on train/validation only."""
        rolling_vol = VolatilityRegimeFilterStrategy._rolling_volatility(
            data,
            window=window,
            annualization_factor=annualization_factor,
        )
        valid = rolling_vol.dropna()
        if valid.empty:
            return fixed_threshold
        if use_quantile:
            return float(valid.quantile(quantile))
        return fixed_threshold

    @staticmethod
    def _rolling_volatility(
        data: pd.DataFrame,
        window: int,
        annualization_factor: int,
    ) -> pd.Series:
        if "returns" not in data.columns:
            returns = data["close"].pct_change()
        else:
            returns = data["returns"]
        return returns.rolling(window=window, min_periods=window).std() * np.sqrt(
            annualization_factor
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        rolling_vol = self._rolling_volatility(
            out,
            window=self.window,
            annualization_factor=self.annualization_factor,
        )
        signal = (rolling_vol < self.threshold).astype(float)
        signal = signal.where(rolling_vol.notna(), 0.0)
        return signals_from_values(out, signal)
