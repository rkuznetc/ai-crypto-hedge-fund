from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from crypto_hf.metrics.signal_diagnostics import compute_signal_diagnostics
from crypto_hf.models.base import BaseDirectionModel
from crypto_hf.models.dummy import (
    DummyAlwaysDownModel,
    DummyAlwaysUpModel,
    DummyStratifiedModel,
)
from crypto_hf.models.econometric import AutoRegReturnModel
from crypto_hf.models.ml import (
    GradientBoostingDirectionModel,
    LogisticRegressionDirectionModel,
    RandomForestDirectionModel,
    RidgeReturnRegressionModel,
)
from crypto_hf.pipeline.baseline import build_features
from crypto_hf.pipeline.threshold_calibration import (
    calibrate_probability_threshold,
    split_train_validation_test,
)
from crypto_hf.strategies.buy_and_hold import BuyAndHoldStrategy
from crypto_hf.strategies.ensemble import EnsembleMajorityVoteStrategy
from crypto_hf.strategies.ml_strategy import EconometricSignalStrategy, MLSignalStrategy
from crypto_hf.strategies.sma_crossover import SMACrossoverStrategy
from crypto_hf.strategies.statistical import (
    MomentumBreakoutStrategy,
    VolatilityRegimeFilterStrategy,
    ZScoreMeanReversionStrategy,
)
from crypto_hf.visualization.plots import (
    export_metrics_table,
    plot_drawdown_comparison,
    plot_equity_curve,
    plot_grouped_equity_and_drawdowns,
    plot_metric_ranking,
    plot_metrics_table,
    plot_model_prediction_diagnostics,
)


@dataclass
class SingleAssetModelsOutputs:
    """Artifacts from the single-asset model comparison pipeline."""

    config: SingleAssetModelsConfig
    validation: ValidationResult
    train: pd.DataFrame
    validation_split: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    results: dict[str, BacktestResult]
    trading_metrics: pd.DataFrame
    classification_metrics: pd.DataFrame
    signal_diagnostics: pd.DataFrame
    selected_thresholds: pd.DataFrame
    threshold_validation: pd.DataFrame
    model_predictions: dict[str, pd.DataFrame] = field(default_factory=dict)


def prepare_model_dataset(
    raw: pd.DataFrame,
    config: SingleAssetModelsConfig,
) -> tuple[pd.DataFrame, list[str]]:
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


def _econometric_trading_threshold(config: SingleAssetModelsConfig) -> float:
    if config.econometric_use_cost_threshold:
        return config.fee_rate + config.slippage
    return 0.0


def _cost_threshold(config: SingleAssetModelsConfig) -> float:
    return config.fee_rate + config.slippage


def _make_backtester(config: SingleAssetModelsConfig) -> VectorbtBacktester:
    return VectorbtBacktester(
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        annualization_factor=config.annualization_factor,
        slippage=config.slippage,
    )


def _make_gross_backtester(config: SingleAssetModelsConfig) -> VectorbtBacktester:
    return VectorbtBacktester(
        initial_cash=config.initial_cash,
        fee_rate=0.0,
        annualization_factor=config.annualization_factor,
        slippage=0.0,
    )


def _return_diagnostics(
    net_result: BacktestResult,
    gross_engine: VectorbtBacktester,
    prices: pd.Series,
    positions: pd.Series,
) -> dict[str, float]:
    net = float(net_result.metrics.get("total_return", 0.0))
    gross_result = gross_engine.run(
        prices,
        positions,
        strategy_name=f"{net_result.strategy_name}_gross",
    )
    gross = float(gross_result.metrics.get("total_return", 0.0))
    return {
        "net_total_return": net,
        "gross_total_return": gross,
        "estimated_cost_impact": gross - net,
    }


def _build_prediction_frame(
    test: pd.DataFrame,
    y_test: pd.Series,
    prediction_direction: pd.Series,
    signal_for_trading: pd.Series,
    probability: pd.Series | None = None,
    predicted_return: pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "actual_target_up": y_test,
            "actual_next_return": test["next_return"],
            "prediction_direction": prediction_direction,
            "signal_for_trading": signal_for_trading,
        },
        index=test.index,
    )
    if probability is not None:
        frame["probability"] = probability
    if predicted_return is not None:
        frame["predicted_return"] = predicted_return
    return frame


def _apply_ml_threshold_calibration(
    model: BaseDirectionModel,
    train_base: pd.DataFrame,
    validation: pd.DataFrame,
    pre_test: pd.DataFrame,
    feature_cols: list[str],
    config: SingleAssetModelsConfig,
    backtester: VectorbtBacktester,
) -> tuple[float, list[dict[str, Any]]]:
    model.fit(train_base[feature_cols], train_base["target_up"])
    if validation.empty:
        return config.ml_probability_threshold, []

    calibration = calibrate_probability_threshold(
        model,
        validation,
        feature_cols,
        config.ml_threshold_candidates,
        backtester,
    )
    model.threshold = calibration.selected_threshold
    model.fit(pre_test[feature_cols], pre_test["target_up"])
    return calibration.selected_threshold, calibration.rows


def _save_grouped_and_ranking_plots(
    results: dict[str, BacktestResult],
    figures_dir: Path,
) -> None:
    equity_curves = {name: r.equity_curve for name, r in results.items()}
    metrics = {name: r.metrics for name, r in results.items()}

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

    benchmarks_statistical = [
        name
        for name in [
            "buy_and_hold",
            "sma_crossover",
            "stat_zscore_mean_reversion",
            "stat_momentum_breakout",
            "stat_volatility_regime_filter",
            "dummy_always_cash",
        ]
        if name in equity_curves
    ]
    plot_grouped_equity_and_drawdowns(
        equity_curves,
        benchmarks_statistical,
        equity_title="Benchmarks and Statistical Signals (test)",
        drawdown_title="Drawdowns: Benchmarks and Statistical (test)",
        equity_save_path=figures_dir / "single_asset_equity_benchmarks_statistical.png",
        drawdown_save_path=figures_dir / "single_asset_drawdowns_benchmarks_statistical.png",
    )

    model_names = [
        name
        for name in [
            "econometric_autoreg",
            "ml_logistic_regression",
            "ml_random_forest",
            "ml_gradient_boosting",
            "ml_ridge_regression",
            "sma_crossover",
        ]
        if name in equity_curves
    ]
    plot_grouped_equity_and_drawdowns(
        equity_curves,
        model_names,
        equity_title="Econometric and ML Models (test)",
        drawdown_title="Drawdowns: Econometric and ML (test)",
        equity_save_path=figures_dir / "single_asset_equity_models.png",
        drawdown_save_path=figures_dir / "single_asset_drawdowns_models.png",
    )

    dummy_names = [
        name
        for name in [
            "dummy_always_long",
            "dummy_always_cash",
            "dummy_random_signal",
            "buy_and_hold",
        ]
        if name in equity_curves
    ]
    plot_grouped_equity_and_drawdowns(
        equity_curves,
        dummy_names,
        equity_title="Dummy Baselines (test)",
        drawdown_title="Drawdowns: Dummy Baselines (test)",
        equity_save_path=figures_dir / "single_asset_equity_dummy_baselines.png",
        drawdown_save_path=figures_dir / "single_asset_drawdowns_dummy_baselines.png",
    )

    plot_metric_ranking(
        metrics,
        metric="total_return",
        title="Total Return Ranking (test)",
        save_path=figures_dir / "single_asset_total_return_ranking.png",
    )
    plot_metric_ranking(
        metrics,
        metric="sharpe_ratio",
        title="Sharpe Ratio Ranking (test)",
        save_path=figures_dir / "single_asset_sharpe_ranking.png",
    )
    plot_metric_ranking(
        metrics,
        metric="max_drawdown",
        title="Max Drawdown Ranking (test)",
        save_path=figures_dir / "single_asset_max_drawdown_ranking.png",
    )


def run_single_asset_models_pipeline(
    config: SingleAssetModelsConfig,
    reports_dir: Path = Path("reports"),
) -> SingleAssetModelsOutputs:
    """Run full baseline + econometric + ML + statistical comparison on the test period."""
    raw = load_ohlcv_csv(config.data_path)
    validation = validate_ohlcv(raw, timeframe=config.timeframe)
    dataset, feature_cols = prepare_model_dataset(raw, config)
    train_base, validation_split, test = split_train_validation_test(
        dataset,
        config.train_size,
        config.validation_size_within_train,
    )
    pre_test = pd.concat([train_base, validation_split])

    X_train_base = train_base[feature_cols]
    y_train_base = train_base["target_up"]
    X_test = test[feature_cols]
    y_test = test["target_up"]

    engine = _make_backtester(config)
    gross_engine = _make_gross_backtester(config)
    prices = test["close"]
    results: dict[str, BacktestResult] = {}
    signal_frames: dict[str, pd.DataFrame] = {}
    classification_rows: dict[str, dict[str, float]] = {}
    model_predictions: dict[str, pd.DataFrame] = {}
    diagnostics_rows: dict[str, dict[str, float]] = {}
    selected_threshold_rows: list[dict[str, float | str]] = []
    threshold_validation_rows: list[dict[str, Any]] = []

    econ_trading_threshold = _econometric_trading_threshold(config)
    cost_threshold = _cost_threshold(config)

    bh = BuyAndHoldStrategy()
    bh_signals = bh.generate_signals(test)
    results[bh.name] = engine.run(prices, bh_signals["position"], strategy_name=bh.name)
    signal_frames[bh.name] = bh_signals

    sma = SMACrossoverStrategy(config.fast_window, config.slow_window)
    sma_signals = sma.generate_signals(test)
    results[sma.name] = engine.run(prices, sma_signals["position"], strategy_name=sma.name)
    signal_frames[sma.name] = sma_signals

    econ_model = AutoRegReturnModel(lags=config.econometric_lags, trading_threshold=econ_trading_threshold)
    econ_model.fit(pre_test[["returns"]], pre_test["target_up"])
    econ_predicted_returns = econ_model.predict_returns(test[["returns"]])
    econ_prediction_direction = econ_model.classification_direction(econ_predicted_returns)
    econ_signals = econ_model.predict_signals(test[["returns"]])
    econ_strategy = EconometricSignalStrategy(econ_signals)
    econ_signal_frame = econ_strategy.generate_signals(test)
    results[econ_strategy.name] = engine.run(
        prices,
        econ_signal_frame["position"],
        strategy_name=econ_strategy.name,
    )
    signal_frames[econ_strategy.name] = econ_signal_frame
    classification_rows[econ_strategy.name] = compute_classification_metrics(
        y_test,
        econ_prediction_direction,
        econ_predicted_returns,
    )
    model_predictions[econ_strategy.name] = _build_prediction_frame(
        test,
        y_test,
        econ_prediction_direction,
        econ_signals,
        predicted_return=econ_predicted_returns,
    )
    selected_threshold_rows.append(
        {"model": econ_strategy.name, "selected_threshold": econ_trading_threshold}
    )

    logit = LogisticRegressionDirectionModel(
        C=config.logistic_regression_c,
        max_iter=config.logistic_regression_max_iter,
        threshold=config.ml_probability_threshold,
    )
    logit_threshold, logit_validation_rows = _apply_ml_threshold_calibration(
        logit,
        train_base,
        validation_split,
        pre_test,
        feature_cols,
        config,
        engine,
    )
    threshold_validation_rows.extend(logit_validation_rows)
    logit_proba = pd.Series(logit.predict_proba(X_test), index=test.index, name="probability")
    logit_pred = pd.Series(logit.predict(X_test), index=test.index, name="prediction_direction")
    logit_signals = logit.predict_signals(X_test)
    logit_strategy = MLSignalStrategy(logit_signals, name=logit.name)
    logit_signal_frame = logit_strategy.generate_signals(test)
    results[logit_strategy.name] = engine.run(
        prices,
        logit_signal_frame["position"],
        strategy_name=logit_strategy.name,
    )
    signal_frames[logit_strategy.name] = logit_signal_frame
    classification_rows[logit_strategy.name] = compute_classification_metrics(
        y_test,
        logit_pred,
        logit_proba,
    )
    model_predictions[logit_strategy.name] = _build_prediction_frame(
        test,
        y_test,
        logit_pred,
        logit_signals,
        probability=logit_proba,
    )
    selected_threshold_rows.append({"model": logit.name, "selected_threshold": logit_threshold})

    if config.enable_random_forest:
        rf = RandomForestDirectionModel(
            n_estimators=config.random_forest_n_estimators,
            threshold=config.ml_probability_threshold,
        )
        rf_threshold, rf_validation_rows = _apply_ml_threshold_calibration(
            rf,
            train_base,
            validation_split,
            pre_test,
            feature_cols,
            config,
            engine,
        )
        threshold_validation_rows.extend(rf_validation_rows)
        rf_proba = pd.Series(rf.predict_proba(X_test), index=test.index)
        rf_pred = pd.Series(rf.predict(X_test), index=test.index)
        rf_signals = rf.predict_signals(X_test)
        rf_strategy = MLSignalStrategy(rf_signals, name=rf.name)
        rf_signal_frame = rf_strategy.generate_signals(test)
        results[rf_strategy.name] = engine.run(
            prices,
            rf_signal_frame["position"],
            strategy_name=rf_strategy.name,
        )
        signal_frames[rf_strategy.name] = rf_signal_frame
        classification_rows[rf_strategy.name] = compute_classification_metrics(
            y_test,
            rf_pred,
            rf_proba,
        )
        model_predictions[rf_strategy.name] = _build_prediction_frame(
            test,
            y_test,
            rf_pred,
            rf_signals,
            probability=rf_proba,
        )
        selected_threshold_rows.append({"model": rf.name, "selected_threshold": rf_threshold})

    if config.enable_gradient_boosting:
        gbc = GradientBoostingDirectionModel(
            n_estimators=config.gradient_boosting_n_estimators,
            threshold=config.ml_probability_threshold,
        )
        gbc_threshold, gbc_validation_rows = _apply_ml_threshold_calibration(
            gbc,
            train_base,
            validation_split,
            pre_test,
            feature_cols,
            config,
            engine,
        )
        threshold_validation_rows.extend(gbc_validation_rows)
        gbc_proba = pd.Series(gbc.predict_proba(X_test), index=test.index)
        gbc_pred = pd.Series(gbc.predict(X_test), index=test.index)
        gbc_signals = gbc.predict_signals(X_test)
        gbc_strategy = MLSignalStrategy(gbc_signals, name=gbc.name)
        gbc_signal_frame = gbc_strategy.generate_signals(test)
        results[gbc_strategy.name] = engine.run(
            prices,
            gbc_signal_frame["position"],
            strategy_name=gbc_strategy.name,
        )
        signal_frames[gbc_strategy.name] = gbc_signal_frame
        classification_rows[gbc_strategy.name] = compute_classification_metrics(
            y_test,
            gbc_pred,
            gbc_proba,
        )
        model_predictions[gbc_strategy.name] = _build_prediction_frame(
            test,
            y_test,
            gbc_pred,
            gbc_signals,
            probability=gbc_proba,
        )
        selected_threshold_rows.append({"model": gbc.name, "selected_threshold": gbc_threshold})

    if config.enable_ridge_regression:
        ridge = RidgeReturnRegressionModel(alpha=config.ridge_alpha, cost_threshold=cost_threshold)
        ridge.fit(pre_test[feature_cols], pre_test["next_return"])
        ridge_predicted_returns = ridge.predict_returns(X_test)
        ridge_pred_direction = ridge.classification_direction(ridge_predicted_returns)
        ridge_signals = ridge.predict_signals(X_test)
        ridge_strategy = MLSignalStrategy(ridge_signals, name=ridge.name)
        ridge_signal_frame = ridge_strategy.generate_signals(test)
        results[ridge_strategy.name] = engine.run(
            prices,
            ridge_signal_frame["position"],
            strategy_name=ridge_strategy.name,
        )
        signal_frames[ridge_strategy.name] = ridge_signal_frame
        classification_rows[ridge_strategy.name] = compute_classification_metrics(
            y_test,
            ridge_pred_direction,
            ridge_predicted_returns,
        )
        model_predictions[ridge_strategy.name] = _build_prediction_frame(
            test,
            y_test,
            ridge_pred_direction,
            ridge_signals,
            predicted_return=ridge_predicted_returns,
        )
        selected_threshold_rows.append({"model": ridge.name, "selected_threshold": cost_threshold})

    if config.enable_dummy_baselines:
        dummy_models: list[tuple[BaseDirectionModel, str | None]] = [
            (DummyAlwaysUpModel(), "dummy_always_long"),
            (DummyAlwaysDownModel(), "dummy_always_cash"),
            (DummyStratifiedModel(random_state=42), "dummy_random_signal"),
        ]
        for dummy_model, trading_name in dummy_models:
            dummy_model.fit(X_train_base, y_train_base)
            dummy_pred = pd.Series(dummy_model.predict(X_test), index=test.index)
            dummy_proba = pd.Series(dummy_model.predict_proba(X_test), index=test.index)
            classification_rows[dummy_model.name] = compute_classification_metrics(
                y_test,
                dummy_pred,
                dummy_proba,
            )
            dummy_signals = dummy_model.predict_signals(X_test)
            model_predictions[dummy_model.name] = _build_prediction_frame(
                test,
                y_test,
                dummy_pred,
                dummy_signals,
                probability=dummy_proba,
            )
            if trading_name is not None:
                dummy_strategy = MLSignalStrategy(dummy_signals, name=trading_name)
                dummy_signal_frame = dummy_strategy.generate_signals(test)
                results[trading_name] = engine.run(
                    prices,
                    dummy_signal_frame["position"],
                    strategy_name=trading_name,
                )
                signal_frames[trading_name] = dummy_signal_frame

    zscore = ZScoreMeanReversionStrategy(
        window=config.zscore_window,
        entry_threshold=config.zscore_entry_threshold,
        exit_threshold=config.zscore_exit_threshold,
    )
    zscore_signals = zscore.generate_signals(test)
    results[zscore.name] = engine.run(
        prices,
        zscore_signals["position"],
        strategy_name=zscore.name,
    )
    signal_frames[zscore.name] = zscore_signals

    breakout = MomentumBreakoutStrategy(
        window=config.breakout_window,
        momentum_threshold=config.stat_momentum_threshold,
    )
    breakout_signals = breakout.generate_signals(test)
    results[breakout.name] = engine.run(
        prices,
        breakout_signals["position"],
        strategy_name=breakout.name,
    )
    signal_frames[breakout.name] = breakout_signals

    vol_threshold = VolatilityRegimeFilterStrategy.calibrate_threshold(
        pre_test,
        window=config.vol_regime_window,
        annualization_factor=config.annualization_factor,
        use_quantile=config.vol_regime_use_quantile,
        fixed_threshold=config.vol_regime_threshold,
        quantile=config.vol_regime_quantile,
    )
    vol_strategy = VolatilityRegimeFilterStrategy(
        window=config.vol_regime_window,
        threshold=vol_threshold,
        annualization_factor=config.annualization_factor,
    )
    vol_signals = vol_strategy.generate_signals(test)
    results[vol_strategy.name] = engine.run(
        prices,
        vol_signals["position"],
        strategy_name=vol_strategy.name,
    )
    signal_frames[vol_strategy.name] = vol_signals
    selected_threshold_rows.append(
        {"model": vol_strategy.name, "selected_threshold": vol_threshold}
    )

    if config.enable_ensemble_majority_vote:
        component_signals = {
            name: signal_frames[name]["signal"]
            for name in config.ensemble_components
            if name in signal_frames
        }
        missing = [name for name in config.ensemble_components if name not in signal_frames]
        if missing:
            raise ValueError(f"Ensemble components missing generated signals: {missing}")
        ensemble = EnsembleMajorityVoteStrategy(
            component_signals=component_signals,
            min_votes=config.ensemble_min_votes,
        )
        ensemble_signals = ensemble.generate_signals(test)
        results[ensemble.name] = engine.run(
            prices,
            ensemble_signals["position"],
            strategy_name=ensemble.name,
        )
        signal_frames[ensemble.name] = ensemble_signals

    threshold_lookup = {
        str(row["model"]): float(row["selected_threshold"]) for row in selected_threshold_rows
    }

    for strategy_name, signal_frame in signal_frames.items():
        diagnostics = compute_signal_diagnostics(
            signal_frame["signal"],
            signal_frame["position"],
            selected_threshold=threshold_lookup.get(strategy_name),
        )
        result = results[strategy_name]
        diagnostics.update(
            _return_diagnostics(result, gross_engine, prices, signal_frame["position"])
        )
        diagnostics_rows[strategy_name] = diagnostics

    trading_metrics = export_metrics_table({name: r.metrics for name, r in results.items()})
    classification_metrics = pd.DataFrame(classification_rows).T
    classification_metrics.index.name = "model"
    signal_diagnostics = pd.DataFrame(diagnostics_rows).T
    signal_diagnostics.index.name = "strategy"
    selected_thresholds = pd.DataFrame(selected_threshold_rows)
    threshold_validation = pd.DataFrame(threshold_validation_rows)

    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    export_metrics_table(
        {name: r.metrics for name, r in results.items()},
        metrics_dir / "single_asset_model_metrics.csv",
    )
    classification_metrics.to_csv(metrics_dir / "single_asset_model_classification.csv")
    signal_diagnostics.to_csv(metrics_dir / "single_asset_signal_diagnostics.csv")
    selected_thresholds.to_csv(metrics_dir / "single_asset_selected_thresholds.csv", index=False)
    if not threshold_validation.empty:
        threshold_validation.to_csv(
            metrics_dir / "single_asset_threshold_validation.csv",
            index=False,
        )

    for model_name, preds in model_predictions.items():
        preds.to_csv(metrics_dir / f"predictions_{model_name}.csv")

    _save_grouped_and_ranking_plots(results, figures_dir)
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
        train=train_base,
        validation_split=validation_split,
        test=test,
        feature_columns=feature_cols,
        results=results,
        trading_metrics=trading_metrics,
        classification_metrics=classification_metrics,
        signal_diagnostics=signal_diagnostics,
        selected_thresholds=selected_thresholds,
        threshold_validation=threshold_validation,
        model_predictions=model_predictions,
    )


def _close_all_figures() -> None:
    import matplotlib.pyplot as plt

    for fig_num in plt.get_fignums():
        plt.close(fig_num)
