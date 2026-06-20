from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
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
                ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=42)),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.direction_from_proba(self.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X)[:, 1]

    def direction_from_proba(self, proba: np.ndarray | pd.Series) -> np.ndarray:
        return (np.asarray(proba, dtype=float) > self.threshold).astype(int)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        signals = (proba > self.threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")


class RandomForestDirectionModel(BaseDirectionModel):
    """Random forest classifier for next-day direction."""

    name = "ml_random_forest"

    def __init__(self, n_estimators: int = 100, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.direction_from_proba(self.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def direction_from_proba(self, proba: np.ndarray | pd.Series) -> np.ndarray:
        return (np.asarray(proba, dtype=float) > self.threshold).astype(int)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        signals = (proba > self.threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")


class GradientBoostingDirectionModel(BaseDirectionModel):
    """Gradient boosting classifier for next-day direction."""

    name = "ml_gradient_boosting"

    def __init__(self, n_estimators: int = 100, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=42,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.direction_from_proba(self.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def direction_from_proba(self, proba: np.ndarray | pd.Series) -> np.ndarray:
        return (np.asarray(proba, dtype=float) > self.threshold).astype(int)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        signals = (proba > self.threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")


class RidgeReturnRegressionModel(BaseDirectionModel):
    """Ridge regression on next_return with cost-aware trading signal."""

    name = "ml_ridge_regression"

    def __init__(self, alpha: float = 1.0, cost_threshold: float = 0.0) -> None:
        self.cost_threshold = cost_threshold
        self._pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=alpha, random_state=42)),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_returns(X).to_numpy(dtype=float)

    def predict_returns(self, X: pd.DataFrame) -> pd.Series:
        values = self._pipeline.predict(X)
        return pd.Series(values, index=X.index, name="predicted_return")

    def classification_direction(self, predicted_returns: pd.Series) -> pd.Series:
        return (predicted_returns > 0.0).astype(int)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        predicted = self.predict_returns(X)
        signals = (predicted > self.cost_threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")
