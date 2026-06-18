from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_hf.data.loader import REQUIRED_COLUMNS


class DataValidationError(ValueError):
    """Raised when OHLCV data fails schema or quality checks."""


@dataclass(frozen=True)
class ValidationResult:
    """Summary of successful data validation."""

    row_count: int
    start: pd.Timestamp
    end: pd.Timestamp
    columns: tuple[str, ...]


def validate_ohlcv(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLCV DataFrame schema and basic data quality."""
    _check_required_columns(df)
    _check_sorted_index(df)
    _check_no_duplicate_timestamps(df)
    _check_positive_prices(df)
    _check_high_low(df)
    _check_volume(df)
    _check_no_missing_close(df)

    index = df.index
    return ValidationResult(
        row_count=len(df),
        start=index[0],
        end=index[-1],
        columns=tuple(df.columns),
    )


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS[1:] if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def _check_sorted_index(df: pd.DataFrame) -> None:
    if not df.index.is_monotonic_increasing:
        raise DataValidationError("Timestamps are not sorted in ascending order")


def _check_no_duplicate_timestamps(df: pd.DataFrame) -> None:
    if df.index.has_duplicates:
        raise DataValidationError("Duplicate timestamps found")


def _check_positive_prices(df: pd.DataFrame) -> None:
    for col in ("open", "high", "low", "close"):
        if (df[col] <= 0).any():
            raise DataValidationError(f"Non-positive values found in column '{col}'")


def _check_high_low(df: pd.DataFrame) -> None:
    if (df["high"] < df["low"]).any():
        raise DataValidationError("Found rows where high < low")


def _check_volume(df: pd.DataFrame) -> None:
    if (df["volume"] < 0).any():
        raise DataValidationError("Negative volume values found")


def _check_no_missing_close(df: pd.DataFrame) -> None:
    if df["close"].isna().any():
        raise DataValidationError("Missing close values found")
