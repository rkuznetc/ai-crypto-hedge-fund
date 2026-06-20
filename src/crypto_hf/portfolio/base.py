from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class OptimizationError(RuntimeError):
    """Raised when a portfolio optimizer fails to produce valid weights."""


@dataclass
class PortfolioWeights:
    """Optimized static portfolio weights."""

    name: str
    weights: pd.Series
    metadata: dict[str, float | str] = field(default_factory=dict)


class BasePortfolioOptimizer(ABC):
    """Interface for static and future rolling portfolio optimizers."""

    name: str = "base_optimizer"

    @abstractmethod
    def optimize(self, returns_train: pd.DataFrame) -> PortfolioWeights:
        """Estimate weights using training-period returns only."""


def validate_portfolio_weights(
    weights: pd.Series,
    *,
    allow_short: bool,
    min_weight: float,
    max_weight: float,
    tolerance: float = 1e-6,
) -> None:
    """Validate optimizer output weights."""
    if weights.isna().any():
        raise OptimizationError("Portfolio weights contain NaN values")
    if len(weights) == 0:
        raise OptimizationError("Portfolio weights are empty")
    if not np.isclose(float(weights.sum()), 1.0, atol=tolerance):
        raise OptimizationError(f"Weights must sum to 1, got {float(weights.sum()):.6f}")
    if not allow_short and (weights < -tolerance).any():
        raise OptimizationError("Negative weights are not allowed when allow_short=False")
    if (weights > max_weight + tolerance).any():
        raise OptimizationError(f"Weight exceeds max_weight={max_weight}")
    if min_weight > 0 and (weights < min_weight - tolerance).any():
        raise OptimizationError(f"Weight below min_weight={min_weight}")


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize weights to sum to 1."""
    total = float(weights.sum())
    if total <= 0:
        raise OptimizationError("Cannot normalize non-positive weight sum")
    return weights / total
