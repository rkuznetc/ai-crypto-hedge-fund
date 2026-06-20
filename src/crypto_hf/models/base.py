from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseDirectionModel(ABC):
    """Minimal interface for single-asset direction/return models."""

    name: str = "base_model"

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit model on training features and targets."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point predictions (returns or class labels)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return positive-class probabilities when available."""
        return None
