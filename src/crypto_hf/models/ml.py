from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_hf.models.base import BaseDirectionModel


class LogisticRegressionDirectionModel(BaseDirectionModel):
    """StandardScaler + logistic regression for next-day direction."""

    name = "ml_logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=C, max_iter=max_iter, random_state=42),
                ),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X)[:, 1]

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        signals = (proba > self.threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")


class RandomForestDirectionModel(BaseDirectionModel):
    """Random forest classifier for next-day direction."""

    name = "ml_random_forest"

    def __init__(self, n_estimators: int = 100, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X)[:, 1]

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        signals = (proba > self.threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")
