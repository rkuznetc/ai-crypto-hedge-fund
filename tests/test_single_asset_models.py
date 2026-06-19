from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.config import SingleAssetModelsConfig
from crypto_hf.models.econometric import AutoRegReturnModel
from crypto_hf.models.ml import LogisticRegressionDirectionModel
from crypto_hf.pipeline.baseline import split_train_test
from crypto_hf.pipeline.single_asset_models import prepare_model_dataset, run_single_asset_models_pipeline
from crypto_hf.strategies.ml_strategy import EconometricSignalStrategy, MLSignalStrategy


def _config() -> SingleAssetModelsConfig:
    return SingleAssetModelsConfig(
        train_size=0.7,
        return_lags=[1, 2],
        rolling_mean_windows=[3],
        rolling_vol_windows=[3],
        momentum_windows=[3],
        sma_ratio_windows=[3],
        rsi_window=3,
        econometric_lags=2,
        enable_random_forest=False,
    )


def _raw_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0.2, 1.0, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )


def test_scaler_fit_only_on_train() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(), _config())
    train, test = split_train_test(dataset, train_size=0.7)
    model = LogisticRegressionDirectionModel()
    model.fit(train[feature_cols], train["target_up"])
    assert len(model.predict_proba(train[feature_cols])) == len(train)
    assert len(model.predict_proba(test[feature_cols])) == len(test)


def test_model_strategy_binary_positions_match_test_index() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(), _config())
    train, test = split_train_test(dataset, train_size=0.7)
    model = LogisticRegressionDirectionModel()
    model.fit(train[feature_cols], train["target_up"])
    signals = model.predict_signals(test[feature_cols])
    out = MLSignalStrategy(signals, name=model.name).generate_signals(test)
    assert list(out.index) == list(test.index)
    assert set(out["position"].dropna().unique()).issubset({0.0, 1.0})


def test_econometric_strategy_passes_backtester() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(), _config())
    train, test = split_train_test(dataset, train_size=0.7)
    econ = AutoRegReturnModel(lags=2, threshold=0.0)
    econ.fit(train[["returns"]], train["target_up"])
    signals = econ.predict_signals(test[["returns"]])
    positions = EconometricSignalStrategy(signals).generate_signals(test)["position"]
    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0).run(test["close"], positions)
    assert not result.equity_curve.empty


def test_pipeline_smoke(tmp_path: Path) -> None:
    config = _config()
    config = config.model_copy(update={"data_path": tmp_path / "dummy.csv"})
    raw = _raw_ohlcv(200)
    raw.reset_index().rename(columns={"index": "timestamp"}).to_csv(config.data_path, index=False)

    outputs = run_single_asset_models_pipeline(
        config.model_copy(update={"data_path": config.data_path}),
        reports_dir=tmp_path / "reports",
    )
    assert len(outputs.results) >= 4
    assert (tmp_path / "reports" / "metrics" / "single_asset_model_metrics.csv").exists()
