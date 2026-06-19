from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class BaselineConfig(BaseModel):
    """Configuration for the baseline single-asset backtest experiment."""

    symbol: str = "BTC/USDT"
    timeframe: str = "1d"
    data_path: Path = Path("data/raw/BTC_USDT_1d.csv")
    initial_cash: float = Field(default=10_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0, ge=0)
    train_size: float = Field(default=0.7, gt=0, lt=1)
    fast_window: int = Field(default=10, gt=0)
    slow_window: int = Field(default=30, gt=0)
    volatility_window: int = Field(default=20, gt=0)
    annualization_factor: int = Field(default=365, gt=0)

    @field_validator("data_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("slow_window")
    @classmethod
    def _slow_gt_fast(cls, slow_window: int, info: Any) -> int:
        fast = info.data.get("fast_window")
        if fast is not None and slow_window <= fast:
            raise ValueError("slow_window must be greater than fast_window")
        return slow_window


class SingleAssetModelsConfig(BaselineConfig):
    """Configuration for single-asset model comparison pipeline."""

    return_lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 5])
    rolling_mean_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    rolling_vol_windows: list[int] = Field(default_factory=lambda: [10, 20])
    momentum_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    sma_ratio_windows: list[int] = Field(default_factory=lambda: [10, 20])
    rsi_window: int = Field(default=14, gt=0)
    econometric_lags: int = Field(default=5, gt=0)
    econometric_use_cost_threshold: bool = True
    ml_probability_threshold: float = Field(default=0.5, gt=0, lt=1)
    logistic_regression_c: float = Field(default=1.0, gt=0)
    logistic_regression_max_iter: int = Field(default=1000, gt=0)
    enable_random_forest: bool = True
    random_forest_n_estimators: int = Field(default=100, gt=0)


def _resolve_data_path(config: BaselineConfig, config_path: Path) -> BaselineConfig:
    if not config.data_path.is_absolute():
        project_root = config_path.parent.parent
        return config.model_copy(
            update={"data_path": (project_root / config.data_path).resolve()}
        )
    return config


def load_config(path: str | Path = "configs/baseline.yaml") -> BaselineConfig:
    """Load baseline experiment config from a YAML file."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = BaselineConfig(**raw)
    return _resolve_data_path(config, config_path)


def load_single_asset_models_config(
    path: str | Path = "configs/single_asset_models.yaml",
) -> SingleAssetModelsConfig:
    """Load single-asset models experiment config from YAML."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = SingleAssetModelsConfig(**raw)
    return _resolve_data_path(config, config_path)
