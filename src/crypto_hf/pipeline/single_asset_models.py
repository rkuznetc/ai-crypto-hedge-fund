from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from crypto_hf.backtesting.base import BacktestResult
from crypto_hf.backtesting.vectorbt_engine import VectorbtBacktester
from crypto_hf.config import SingleAssetModelsConfig
from crypto_hf.data.loader import load_ohlcv_csv
from crypto_hf.data.validation import ValidationResult, validate_ohlcv
from crypto_hf.features.model_features import (
    build_model_features,
    drop_rows_with_missing_features,
)
from crypto_hf.features.targets import (
    drop_unknown_target_rows,
    make_next_return_target,
    model_feature_columns,
)
from crypto_hf.metrics.classification import compute_classification_metrics
from crypto_hf.models.econometric import AutoRegReturnModel
from crypto_hf.models.ml import LogisticRegressionDirectionModel, RandomForestDirectionModel
from crypto_hf.pipeline.baseline import build_features, split_train_test
from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy
from crypto_hf.strategies.ml_strategy import EconometricSignalStrategy, MLSignalStrategy
from crypto_hf.strategies.sma_crossover import SMACrossoverStrategy
from crypto_hf.visualization.plots import (
    export_metrics_table,
    plot_drawdown_comparison,
    plot_equity_curve,
    plot_metrics_table,
    plot_model_prediction_diagnostics,
)


@dataclass
class SingleAssetModelsOutputs:
    """Artifacts from the single-asset model comparison pipeline."""

    config: SingleAssetModelsConfig
    validation: ValidationResult
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    results: dict[str, BacktestResult]
    trading_metrics: pd.DataFrame
    classification_metrics: pd.DataFrame
    model_predictions: dict[str, pd.DataFrame] = field(default_factory=dict)


def prepare_model_dataset(
    raw: pd.DataFrame,
    config: SingleAssetModelsConfig,
) -> pd.DataFrame:
    """Build technical + model features and targets, dropping incomplete rows."""
    technical = build_features(raw, config)
    model_ready = build_model_features(
        technical,
        return_lags=config.return_lags,
        rolling_mean_windows=config.rolling_mean_windows,
        rolling_vol_windows=config.rolling_vol_windows,
        momentum_windows=config.momentum_windows,
        sma_ratio_windows=config.sma_ratio_windows,
        rsi_window=config.rsi_window,
        annualization_factor=config.annualization_factor,
    )
    model_ready = make_next_return_target(model_ready)
    model_ready = drop_unknown_target_rows(model_ready)
    feature_cols = model_feature_columns(model_ready)
    return drop_rows_with_missing_features(model_ready, feature_cols), feature_cols


def _econometric_threshold(config: SingleAssetModelsConfig) -> float:
    if config.econometric_use_cost_threshold:
        return config.fee_rate + config.slippage
    return 0.0


def _make_backtester(config: SingleAssetModelsConfig) -> VectorbtBacktester:
    return VectorbtBacktester(
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        annualization_factor=config.annualization_factor,
        slippage=config.slippage,
    )


def run_single_asset_models_pipeline(
    config: SingleAssetModelsConfig,
    reports_dir: Path = Path("reports"),
) -> SingleAssetModelsOutputs:
    """Run full baseline + econometric + ML comparison on the test period."""
    raw = load_ohlcv_csv(config.data_path)
    validation = validate_ohlcv(raw, timeframe=config.timeframe)
    dataset, feature_cols = prepare_model_dataset(raw, config)
    train, test = split_train_test(dataset, config.train_size)

    X_train = train[feature_cols]
    y_train = train["target_up"]
    X_test = test[feature_cols]
    y_test = test["target_up"]

    econ_threshold = _econometric_threshold(config)
    econ_model = AutoRegReturnModel(lags=config.econometric_lags, threshold=econ_threshold)
    econ_model.fit(train[["returns"]], y_train)
    econ_signals = econ_model.predict_signals(test[["returns"]])

    logit = LogisticRegressionDirectionModel(
        C=config.logistic_regression_c,
        max_iter=config.logistic_regression_max_iter,
        threshold=config.ml_probability_threshold,
    )
    logit.fit(X_train, y_train)
    logit_signals = logit.predict_signals(X_test)
    logit_pred = pd.Series(logit.predict(X_test), index=test.index, name="prediction")
    logit_proba = pd.Series(logit.predict_proba(X_test), index=test.index, name="probability")

    rf_signals: pd.Series | None = None
    rf_model: RandomForestDirectionModel | None = None
    if config.enable_random_forest:
        rf_model = RandomForestDirectionModel(
            n_estimators=config.random_forest_n_estimators,
            threshold=config.ml_probability_threshold,
        )
        rf_model.fit(X_train, y_train)
        rf_signals = rf_model.predict_signals(X_test)

    engine = _make_backtester(config)
    prices = test["close"]
    results: dict[str, BacktestResult] = {}

    bh = BuyAndHoldStrategy()
    results[bh.name] = engine.run(prices, bh.generate_signals(test)["position"], strategy_name=bh.name)

    sma = SMACrossoverStrategy(config.fast_window, config.slow_window)
    results[sma.name] = engine.run(prices, sma.generate_signals(test)["position"], strategy_name=sma.name)

    econ_strategy = EconometricSignalStrategy(econ_signals)
    results[econ_strategy.name] = engine.run(
        prices,
        econ_strategy.generate_signals(test)["position"],
        strategy_name=econ_strategy.name,
    )

    ml_strategy = MLSignalStrategy(logit_signals, name=logit.name)
    results[ml_strategy.name] = engine.run(
        prices,
        ml_strategy.generate_signals(test)["position"],
        strategy_name=ml_strategy.name,
    )

    if rf_model is not None and rf_signals is not None:
        rf_strategy = MLSignalStrategy(rf_signals, name=rf_model.name)
        results[rf_strategy.name] = engine.run(
            prices,
            rf_strategy.generate_signals(test)["position"],
            strategy_name=rf_strategy.name,
        )

    trading_metrics = export_metrics_table({name: r.metrics for name, r in results.items()})

    econ_pred_direction = (econ_signals > 0).astype(int)
    classification_rows = {
        econ_strategy.name: compute_classification_metrics(y_test, econ_pred_direction),
        ml_strategy.name: compute_classification_metrics(y_test, logit_pred, logit_proba),
    }
    if rf_model is not None and rf_signals is not None:
        rf_pred = pd.Series(rf_model.predict(X_test), index=test.index)
        rf_proba = pd.Series(rf_model.predict_proba(X_test), index=test.index)
        classification_rows[rf_model.name] = compute_classification_metrics(y_test, rf_pred, rf_proba)

    classification_metrics = pd.DataFrame(classification_rows).T
    classification_metrics.index.name = "model"

    model_predictions = {
        ml_strategy.name: pd.DataFrame(
            {"actual": y_test, "prediction": logit_pred, "probability": logit_proba}
        ),
        econ_strategy.name: pd.DataFrame(
            {"actual": y_test, "prediction": econ_pred_direction, "predicted_return": econ_model.predict(test[["returns"]])}
        ),
    }

    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    export_metrics_table(
        {name: r.metrics for name, r in results.items()},
        metrics_dir / "single_asset_model_metrics.csv",
    )
    classification_metrics.to_csv(metrics_dir / "single_asset_model_classification.csv")

    for model_name, preds in model_predictions.items():
        preds.to_csv(metrics_dir / f"predictions_{model_name}.csv")

    equity_curves = {name: r.equity_curve for name, r in results.items()}
    plot_equity_curve(
        equity_curves,
        title="Single-Asset Model Equity Curves (test)",
        save_path=figures_dir / "single_asset_model_equity_curves.png",
    )
    plot_drawdown_comparison(
        equity_curves,
        title="Single-Asset Model Drawdown Comparison (test)",
        save_path=figures_dir / "single_asset_model_drawdowns.png",
    )
    plot_metrics_table(
        {name: r.metrics for name, r in results.items()},
        title="Single-Asset Model Trading Metrics",
        save_path=figures_dir / "single_asset_model_metrics_table.png",
    )
    plot_model_prediction_diagnostics(
        y_test,
        logit_pred,
        title="Logistic Regression Direction Predictions (test)",
        save_path=figures_dir / "ml_logistic_prediction_diagnostics.png",
    )

    _close_all_figures()

    return SingleAssetModelsOutputs(
        config=config,
        validation=validation,
        train=train,
        test=test,
        feature_columns=feature_cols,
        results=results,
        trading_metrics=trading_metrics,
        classification_metrics=classification_metrics,
        model_predictions=model_predictions,
    )


def _close_all_figures() -> None:
    import matplotlib.pyplot as plt

    for fig_num in plt.get_fignums():
        plt.close(fig_num)
