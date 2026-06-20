from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from crypto_hf.models.base import BaseDirectionModel


class DummyAlwaysUpModel(BaseDirectionModel):
    name = "dummy_always_up"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        del X_train, y_train

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=float)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=X.index, name="signal")


class DummyAlwaysDownModel(BaseDirectionModel):
    name = "dummy_always_cash"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        del X_train, y_train

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index, name="signal")


class DummyStratifiedModel(BaseDirectionModel):
    name = "dummy_random_signal"

    def __init__(self, random_state: int = 42) -> None:
        self._model = DummyClassifier(strategy="stratified", random_state=random_state)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        signals = self.predict(X).astype(float)
        return pd.Series(signals, index=X.index, name="signal")
