from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.metrics.performance import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from crypto_hf.portfolio.backtesting import PortfolioBacktestResult
from crypto_hf.portfolio.optimizers import compute_inverse_volatility_weights

_WEIGHT_COLUMN_MAP = {
    "equal_weight": "test_weight_equal",
    "inverse_volatility": "test_weight_inverse_vol",
    "min_variance": "test_weight_min_variance",
    "max_sharpe": "test_weight_max_sharpe",
    "hrp": "test_weight_hrp",
}


def _weight_column_name(method: str) -> str | None:
    return _WEIGHT_COLUMN_MAP.get(method)


def build_asset_risk_report(
    symbols: list[str],
    train_window: pd.DataFrame,
    test_returns: pd.DataFrame,
    close_prices: pd.DataFrame,
    weights: pd.DataFrame,
    optimizer_config,
) -> pd.DataFrame:
    """Build per-asset train/test risk diagnostics and inverse-vol weights."""
    annualization = optimizer_config.annualization_factor
    n_train = len(train_window)
    n_test = len(test_returns)
    data_start = close_prices.index.min()
    data_end = close_prices.index.max()

    inv_raw, inv_final, inv_meta = compute_inverse_volatility_weights(train_window, optimizer_config)
    equal_weight = 1.0 / len(symbols)

    rows: list[dict[str, float | str]] = []
    for symbol in symbols:
        train_asset = train_window[symbol]
        test_asset = test_returns[symbol]
        train_vol = float(train_asset.std() * np.sqrt(annualization))
        train_ret = float(train_asset.mean() * annualization)
        train_sharpe = train_ret / train_vol if train_vol > 0 else 0.0

        test_equity = (1.0 + test_asset).cumprod()
        test_total = total_return(test_equity)
        test_vol = annualized_volatility(test_asset, annualization)
        test_dd = max_drawdown(test_equity)

        rows.append(
            {
                "symbol": symbol,
                "train_annualized_return": train_ret,
                "train_annualized_volatility": train_vol,
                "train_sharpe_ratio": train_sharpe,
                "test_total_return": test_total,
                "test_annualized_volatility": test_vol,
                "test_max_drawdown": test_dd,
                "inverse_vol_raw_weight": float(inv_raw[symbol]),
                "inverse_vol_final_weight": float(inv_final[symbol]),
                "equal_weight": equal_weight,
                "data_start": str(data_start),
                "data_end": str(data_end),
                "n_train_rows": float(n_train),
                "n_test_rows": float(n_test),
                "inverse_vol_fallback_applied": float(
                    inv_meta.get("inverse_vol_fallback_applied", False)
                ),
                "inverse_vol_fallback_reason": str(inv_meta.get("inverse_vol_fallback_reason", "")),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def build_asset_test_performance_report(
    symbols: list[str],
    test_returns: pd.DataFrame,
    weights: pd.DataFrame,
    annualization_factor: int,
) -> pd.DataFrame:
    """Build per-asset standalone test buy-and-hold metrics and optimizer weights."""
    rows: list[dict[str, float | str]] = []
    for symbol in symbols:
        asset_returns = test_returns[symbol]
        equity = (1.0 + asset_returns).cumprod()
        row: dict[str, float | str] = {
            "symbol": symbol,
            "test_total_return": total_return(equity),
            "test_cagr": cagr(equity, annualization_factor),
            "test_annualized_volatility": annualized_volatility(asset_returns, annualization_factor),
            "test_sharpe_ratio": sharpe_ratio(asset_returns, periods_per_year=annualization_factor),
            "test_max_drawdown": max_drawdown(equity),
        }
        for method, method_weights in weights.iterrows():
            col = _weight_column_name(method)
            if col:
                row[col] = float(method_weights[symbol])
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol")


def build_train_test_metrics_report(
    results: dict[str, PortfolioBacktestResult],
    train_results: dict[str, PortfolioBacktestResult],
) -> pd.DataFrame:
    """Combine train and test metrics with decay columns."""
    rows: dict[str, dict[str, float]] = {}
    for name, test_result in results.items():
        train_result = train_results[name]
        row = {
            "train_total_return": train_result.metrics.get("total_return", 0.0),
            "train_cagr": train_result.metrics.get("cagr", 0.0),
            "train_annualized_volatility": train_result.metrics.get("annualized_volatility", 0.0),
            "train_sharpe_ratio": train_result.metrics.get("sharpe_ratio", 0.0),
            "train_max_drawdown": train_result.metrics.get("max_drawdown", 0.0),
            "test_total_return": test_result.metrics.get("total_return", 0.0),
            "test_cagr": test_result.metrics.get("cagr", 0.0),
            "test_annualized_volatility": test_result.metrics.get("annualized_volatility", 0.0),
            "test_sharpe_ratio": test_result.metrics.get("sharpe_ratio", 0.0),
            "test_max_drawdown": test_result.metrics.get("max_drawdown", 0.0),
        }
        row["sharpe_decay"] = row["test_sharpe_ratio"] - row["train_sharpe_ratio"]
        row["return_decay"] = row["test_total_return"] - row["train_total_return"]
        row["drawdown_change"] = row["test_max_drawdown"] - row["train_max_drawdown"]
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")
