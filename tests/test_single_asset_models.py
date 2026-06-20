from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.config import SingleAssetModelsConfig
from crypto_hf.features.model_features import build_model_features, drop_rows_with_missing_features
from crypto_hf.features.targets import make_next_return_target, model_feature_columns
from crypto_hf.metrics.classification import compute_classification_metrics
from crypto_hf.models.econometric import AutoRegReturnModel
from crypto_hf.models.ml import (
    GradientBoostingDirectionModel,
    LogisticRegressionDirectionModel,
    RandomForestDirectionModel,
    RidgeReturnRegressionModel,
)
from crypto_hf.pipeline.baseline import split_train_test
from crypto_hf.pipeline.single_asset_models import prepare_model_dataset, run_single_asset_models_pipeline
from crypto_hf.pipeline.threshold_calibration import (
    calibrate_probability_threshold,
    split_train_validation_test,
)
from crypto_hf.strategies.ml_strategy import EconometricSignalStrategy, MLSignalStrategy
from crypto_hf.strategies.statistical import MomentumBreakoutStrategy, ZScoreMeanReversionStrategy


def _config(**overrides: object) -> SingleAssetModelsConfig:
    base = dict(
        train_size=0.7,
        validation_size_within_train=0.2,
        return_lags=[1, 2],
        rolling_mean_windows=[3],
        rolling_vol_windows=[3],
        momentum_windows=[3],
        sma_ratio_windows=[3],
        rsi_window=3,
        econometric_lags=2,
        enable_random_forest=False,
        enable_gradient_boosting=False,
        enable_ridge_regression=False,
        enable_dummy_baselines=False,
        enable_ensemble_majority_vote=False,
        zscore_window=3,
        breakout_window=3,
    )
    base.update(overrides)
    return SingleAssetModelsConfig(**base)


def _raw_ohlcv(n: int = 120, *, volume: np.ndarray | None = None) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0.2, 1.0, n))
    vol = volume if volume is not None else np.linspace(1000, 2000, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": vol,
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
    econ = AutoRegReturnModel(lags=2, trading_threshold=0.0)
    econ.fit(train[["returns"]], train["target_up"])
    signals = econ.predict_signals(test[["returns"]])
    positions = EconometricSignalStrategy(signals).generate_signals(test)["position"]
    result = VectorbtBacktester(initial_cash=10_000, fee_rate=0.0).run(test["close"], positions)
    assert not result.equity_curve.empty


def test_autoreg_forecast_alignment() -> None:
    idx = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.015, 0.005, 0.01, -0.005, 0.02, -0.01], index=idx)
    train = returns.iloc[:7]
    test = returns.iloc[7:11]
    target_up = (test.shift(-1) > 0).astype(int).iloc[:-1]
    test = test.iloc[: len(target_up)]
    target_up.index = test.index

    model = AutoRegReturnModel(lags=2, trading_threshold=0.0)
    model.fit(pd.DataFrame({"returns": train}), pd.Series(0, index=train.index))
    predicted = model.predict_returns(pd.DataFrame({"returns": test}))
    direction = model.classification_direction(predicted)

    assert len(predicted) == len(test)
    assert compute_classification_metrics(target_up, direction, predicted)["roc_auc"] >= 0.0

    alt_future = test.copy()
    alt_future.iloc[1:] = 999.0
    predicted_alt = model.predict_returns(pd.DataFrame({"returns": alt_future}))
    assert predicted.iloc[0] == pytest.approx(predicted_alt.iloc[0])

    alt_current = test.copy()
    alt_current.iloc[0] = -999.0
    predicted_alt_current = model.predict_returns(pd.DataFrame({"returns": alt_current}))
    assert predicted.iloc[0] != pytest.approx(predicted_alt_current.iloc[0])


def test_ml_threshold_consistency() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(160), _config())
    train, test = split_train_test(dataset, train_size=0.7)
    threshold = 0.6
    model = LogisticRegressionDirectionModel(threshold=threshold)
    model.fit(train[feature_cols], train["target_up"])
    proba = model.predict_proba(test[feature_cols])
    pred = model.predict(test[feature_cols])
    signals = model.predict_signals(test[feature_cols])
    expected = (proba > threshold).astype(int)
    np.testing.assert_array_equal(pred, expected)
    np.testing.assert_array_equal(signals.to_numpy(), expected.astype(float))


def test_autoreg_classification_has_roc_auc() -> None:
    dataset, _ = prepare_model_dataset(_raw_ohlcv(), _config())
    train_base, validation, test = split_train_validation_test(dataset, 0.7, 0.2)
    pre_test = pd.concat([train_base, validation])
    model = AutoRegReturnModel(lags=2, trading_threshold=0.001)
    model.fit(pre_test[["returns"]], pre_test["target_up"])
    predicted = model.predict_returns(test[["returns"]])
    direction = model.classification_direction(predicted)
    metrics = compute_classification_metrics(test["target_up"], direction, predicted)
    assert "roc_auc" in metrics
    trading_signals = model.predict_signals(test[["returns"]])
    assert (trading_signals == (predicted > 0.001).astype(float)).all()


def test_volume_change_inf_dropped_from_dataset() -> None:
    raw = _raw_ohlcv(40, volume=np.array([0.0, 100.0, 200.0] + [150.0] * 37))
    featured = build_model_features(
        raw,
        return_lags=[1],
        rolling_mean_windows=[3],
        rolling_vol_windows=[3],
        momentum_windows=[3],
        sma_ratio_windows=[3],
        rsi_window=3,
    )
    featured = make_next_return_target(featured)
    feature_cols = model_feature_columns(featured)
    dataset = drop_rows_with_missing_features(featured, feature_cols)
    assert np.isfinite(dataset[feature_cols].to_numpy()).all()


def test_ridge_regression_fit_on_train_only_and_binary_signals() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(160), _config())
    train_base, validation, test = split_train_validation_test(dataset, 0.7, 0.2)
    pre_test = pd.concat([train_base, validation])
    model = RidgeReturnRegressionModel(cost_threshold=0.001)
    model.fit(pre_test[feature_cols], pre_test["next_return"])
    predicted = model.predict_returns(test[feature_cols])
    signals = model.predict_signals(test[feature_cols])
    assert len(predicted) == len(test)
    assert set(signals.unique()).issubset({0.0, 1.0})


def test_gradient_boosting_smoke_and_binary_signals() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(160), _config())
    train, test = split_train_test(dataset, train_size=0.7)
    model = GradientBoostingDirectionModel(n_estimators=10, threshold=0.55)
    model.fit(train[feature_cols], train["target_up"])
    signals = model.predict_signals(test[feature_cols])
    assert len(model.predict_proba(test[feature_cols])) == len(test)
    assert set(signals.unique()).issubset({0.0, 1.0})


def test_statistical_strategies_binary_shifted_positions() -> None:
    dataset, _ = prepare_model_dataset(_raw_ohlcv(), _config(zscore_window=3, breakout_window=3))
    test = dataset.iloc[-20:]
    for strategy in [
        ZScoreMeanReversionStrategy(window=3, entry_threshold=-1.0, exit_threshold=1.0),
        MomentumBreakoutStrategy(window=3, momentum_threshold=0.0),
    ]:
        out = strategy.generate_signals(test)
        assert set(out["signal"].unique()).issubset({0.0, 1.0})
        expected = out["signal"].shift(1).fillna(0.0)
        pd.testing.assert_series_equal(out["position"], expected, check_names=False)


def test_gradient_boosting_threshold_calibration() -> None:
    dataset, feature_cols = prepare_model_dataset(_raw_ohlcv(160), _config())
    train_base, validation, test = split_train_validation_test(dataset, 0.7, 0.2)
    pre_test = pd.concat([train_base, validation])
    candidates = [0.45, 0.5, 0.55]
    model = GradientBoostingDirectionModel(n_estimators=10, threshold=0.5)
    model.fit(train_base[feature_cols], train_base["target_up"])
    backtester = VectorbtBacktester(initial_cash=10_000, fee_rate=0.001, slippage=0.0)
    calibration = calibrate_probability_threshold(
        model,
        validation,
        feature_cols,
        candidates,
        backtester,
    )
    assert calibration.selected_threshold in candidates
    assert len(calibration.rows) == len(candidates)
    assert sum(row["selected"] for row in calibration.rows) == 1

    model.threshold = calibration.selected_threshold
    model.fit(pre_test[feature_cols], pre_test["target_up"])
    proba = model.predict_proba(test[feature_cols])
    pred = model.predict(test[feature_cols])
    signals = model.predict_signals(test[feature_cols])
    expected = (proba > calibration.selected_threshold).astype(int)
    np.testing.assert_array_equal(pred, expected)
    np.testing.assert_array_equal(signals.to_numpy(), expected.astype(float))


def test_pipeline_smoke(tmp_path: Path) -> None:
    config = _config(
        enable_random_forest=True,
        enable_gradient_boosting=True,
        enable_ridge_regression=True,
        enable_dummy_baselines=True,
        enable_ensemble_majority_vote=True,
    )
    config = config.model_copy(update={"data_path": tmp_path / "dummy.csv"})
    raw = _raw_ohlcv(200)
    raw.reset_index().rename(columns={"index": "timestamp"}).to_csv(config.data_path, index=False)

    outputs = run_single_asset_models_pipeline(
        config.model_copy(update={"data_path": config.data_path}),
        reports_dir=tmp_path / "reports",
    )
    expected_strategies = {
        "buy_and_hold",
        "sma_crossover",
        "econometric_autoreg",
        "ml_logistic_regression",
        "ml_random_forest",
        "ml_gradient_boosting",
        "ml_ridge_regression",
        "stat_zscore_mean_reversion",
        "stat_momentum_breakout",
        "stat_volatility_regime_filter",
        "ensemble_majority_vote",
        "dummy_always_long",
        "dummy_always_cash",
        "dummy_random_signal",
    }
    assert expected_strategies.issubset(set(outputs.results))
    metrics_dir = tmp_path / "reports" / "metrics"
    assert (metrics_dir / "single_asset_model_metrics.csv").exists()
    assert (metrics_dir / "single_asset_model_classification.csv").exists()
    assert (metrics_dir / "single_asset_signal_diagnostics.csv").exists()
    assert (metrics_dir / "single_asset_selected_thresholds.csv").exists()
    assert (metrics_dir / "single_asset_threshold_validation.csv").exists()
    assert "roc_auc" in outputs.classification_metrics.columns
    assert outputs.classification_metrics.loc["econometric_autoreg", "roc_auc"] >= 0.0

    diagnostics = outputs.signal_diagnostics
    assert "net_total_return" in diagnostics.columns
    assert "gross_total_return" in diagnostics.columns
    assert "estimated_cost_impact" in diagnostics.columns
    assert "total_return" not in diagnostics.columns

    for model_name in ["ml_logistic_regression", "ml_random_forest", "ml_gradient_boosting"]:
        model_rows = outputs.threshold_validation[
            outputs.threshold_validation["model"] == model_name
        ]
        assert len(model_rows) >= 2
        assert model_rows["selected"].sum() == 1
        selected = float(model_rows.loc[model_rows["selected"], "threshold"].iloc[0])
        assert selected == float(
            outputs.selected_thresholds.loc[
                outputs.selected_thresholds["model"] == model_name,
                "selected_threshold",
            ].iloc[0]
        )

    figures_dir = tmp_path / "reports" / "figures"
    assert (figures_dir / "single_asset_equity_benchmarks_statistical.png").exists()
    assert (figures_dir / "single_asset_equity_models.png").exists()
    assert (figures_dir / "single_asset_total_return_ranking.png").exists()
