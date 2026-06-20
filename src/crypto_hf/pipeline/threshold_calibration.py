from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.models.base import BaseDirectionModel
from crypto_hf.strategies.model_signal import signals_from_values


@dataclass
class ThresholdValidationResult:
    """Validation sweep result for probability threshold selection."""

    selected_threshold: float
    rows: list[dict[str, Any]]


def calibrate_probability_threshold(
    model: BaseDirectionModel,
    validation_data: pd.DataFrame,
    feature_cols: list[str],
    config_thresholds: list[float],
    backtester: VectorbtBacktester,
) -> ThresholdValidationResult:
    """Pick probability threshold maximizing validation Sharpe and record all candidates."""
    default = config_thresholds[0] if config_thresholds else 0.5
    if validation_data.empty or not hasattr(model, "predict_proba"):
        return ThresholdValidationResult(selected_threshold=default, rows=[])

    X_val = validation_data[feature_cols]
    proba = model.predict_proba(X_val)
    prices = validation_data["close"]
    rows: list[dict[str, Any]] = []
    best_threshold = default
    best_sharpe = float("-inf")

    for threshold in config_thresholds:
        signals = pd.Series((proba > threshold).astype(float), index=validation_data.index)
        positions = signals_from_values(validation_data, signals, strict_alignment=True)["position"]
        result = backtester.run(prices, positions, strategy_name="threshold_calibration")
        sharpe = float(result.metrics.get("sharpe_ratio", 0.0))
        exposure = float(positions.fillna(0.0).mean())
        rows.append(
            {
                "model": model.name,
                "threshold": float(threshold),
                "validation_total_return": float(result.metrics.get("total_return", 0.0)),
                "validation_sharpe_ratio": sharpe,
                "validation_max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                "validation_exposure_time": exposure,
                "validation_number_of_trades": float(result.metrics.get("number_of_trades", 0.0)),
                "selected": False,
            }
        )
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold

    for row in rows:
        row["selected"] = row["threshold"] == best_threshold

    return ThresholdValidationResult(selected_threshold=best_threshold, rows=rows)


def split_train_validation_test(
    df: pd.DataFrame,
    train_size: float,
    validation_size_within_train: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split."""
    if validation_size_within_train <= 0 or validation_size_within_train >= 1:
        train, test = _split_two_way(df, train_size)
        return train, test.iloc[:0].copy(), test

    train_end = int(len(df) * train_size)
    if train_end <= 0 or train_end >= len(df):
        raise ValueError("train_size produces an empty train or test split")

    train_portion = df.iloc[:train_end]
    test = df.iloc[train_end:].copy()
    val_size = int(len(train_portion) * validation_size_within_train)
    if val_size <= 0 or val_size >= len(train_portion):
        raise ValueError("validation_size_within_train produces an empty train or validation split")

    train_base = train_portion.iloc[:-val_size].copy()
    validation = train_portion.iloc[-val_size:].copy()
    return train_base, validation, test


def _split_two_way(df: pd.DataFrame, train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_size)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("train_size produces an empty train or test split")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
