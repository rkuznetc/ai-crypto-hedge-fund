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
    enable_gradient_boosting: bool = True
    gradient_boosting_n_estimators: int = Field(default=100, gt=0)
    enable_ridge_regression: bool = True
    ridge_alpha: float = Field(default=1.0, gt=0)
    enable_dummy_baselines: bool = True
    validation_size_within_train: float = Field(default=0.2, gt=0, lt=1)
    ml_threshold_candidates: list[float] = Field(default_factory=lambda: [0.45, 0.5, 0.55, 0.6])
    zscore_window: int = Field(default=20, gt=0)
    zscore_entry_threshold: float = -1.0
    zscore_exit_threshold: float = 1.0
    breakout_window: int = Field(default=20, gt=0)
    stat_momentum_threshold: float = 0.0
    vol_regime_window: int = Field(default=20, gt=0)
    vol_regime_threshold: float = Field(default=0.5, gt=0)
    vol_regime_use_quantile: bool = True
    vol_regime_quantile: float = Field(default=0.7, gt=0, lt=1)
    enable_ensemble_majority_vote: bool = True
    ensemble_components: list[str] = Field(
        default_factory=lambda: [
            "sma_crossover",
            "econometric_autoreg",
            "ml_gradient_boosting",
            "ml_ridge_regression",
            "stat_momentum_breakout",
        ]
    )
    ensemble_min_votes: int = Field(default=3, gt=0)


class StaticPortfolioConfig(BaseModel):
    """Configuration for static multi-asset portfolio experiments."""

    symbols: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
        ],
        min_length=2,
    )
    timeframe: str = "1d"
    data_dir: Path = Path("data/raw")
    initial_cash: float = Field(default=10_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0, ge=0)
    annualization_factor: int = Field(default=365, gt=0)
    train_size: float = Field(default=0.7, gt=0, lt=1)
    portfolio_lookback_days: int = Field(default=365, gt=0)
    min_weight: float = Field(default=0.0, ge=0)
    max_weight: float = Field(default=0.35, gt=0, le=1)
    allow_short: bool = False
    risk_free_rate: float = 0.0
    covariance_method: str = "ledoit_wolf"
    expected_return_method: str = "historical_mean"
    ewm_span: int = Field(default=60, gt=0)

    @field_validator("data_dir", mode="before")
    @classmethod
    def _coerce_data_dir(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("max_weight")
    @classmethod
    def _max_weight_valid(cls, max_weight: float, info: Any) -> float:
        min_weight = info.data.get("min_weight", 0.0)
        if min_weight is not None and max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight")
        return max_weight


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


def _resolve_static_portfolio_paths(
    config: StaticPortfolioConfig,
    config_path: Path,
) -> StaticPortfolioConfig:
    if not config.data_dir.is_absolute():
        project_root = config_path.parent.parent
        return config.model_copy(
            update={"data_dir": (project_root / config.data_dir).resolve()}
        )
    return config


def load_static_portfolio_config(
    path: str | Path = "configs/static_multi_asset_portfolio.yaml",
) -> StaticPortfolioConfig:
    """Load static multi-asset portfolio config from YAML."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = StaticPortfolioConfig(**raw)
    return _resolve_static_portfolio_paths(config, config_path)
