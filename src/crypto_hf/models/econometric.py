from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from crypto_hf.models.base import BaseDirectionModel

TARGET_RETURN_COL = "returns"


class AutoRegReturnModel(BaseDirectionModel):
    """AR(p) model on returns with rolling one-step test forecasts."""

    name = "econometric_autoreg"

    def __init__(self, lags: int, threshold: float = 0.0) -> None:
        self.lags = lags
        self.threshold = threshold
        self._train_returns: pd.Series | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if TARGET_RETURN_COL not in X_train.columns:
            raise KeyError(f"Training features must include '{TARGET_RETURN_COL}'")
        self._train_returns = X_train[TARGET_RETURN_COL].dropna()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._train_returns is None:
            raise RuntimeError("Model must be fitted before predict()")
        if TARGET_RETURN_COL not in X.columns:
            raise KeyError(f"Features must include '{TARGET_RETURN_COL}'")

        test_returns = X[TARGET_RETURN_COL]
        history = self._train_returns.tolist()
        predictions: list[float] = []

        for observed in test_returns:
            if len(history) < self.lags + 1:
                predictions.append(0.0)
            else:
                model = AutoReg(np.asarray(history, dtype=float), lags=self.lags, old_names=False).fit()
                forecast = model.predict(start=len(history), end=len(history))
                pred_val = float(np.asarray(forecast).ravel()[0])
                predictions.append(pred_val)
            if pd.notna(observed):
                history.append(float(observed))

        return np.asarray(predictions, dtype=float)

    def predict_signals(self, X: pd.DataFrame) -> pd.Series:
        predicted_returns = self.predict(X)
        signals = (predicted_returns > self.threshold).astype(float)
        return pd.Series(signals, index=X.index, name="signal")
