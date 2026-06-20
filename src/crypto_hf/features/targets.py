from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.features.technical import add_returns

TARGET_COLUMNS = ("next_return", "target_up")


def make_next_return_target(data: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add regression and classification targets for next-bar return."""
    out = data.copy()
    if "returns" not in out.columns:
        out = add_returns(out, price_col=price_col)
    out["next_return"] = out[price_col].pct_change().shift(-1)
    out["target_up"] = (out["next_return"] > 0).astype(int)
    return out


def drop_unknown_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the future return target is unknown."""
    return df.loc[df["next_return"].notna()].copy()


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names, excluding OHLCV, targets, and diagnostics."""
    excluded = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "returns",
        "log_returns",
        "drawdown",
        "next_return",
        "target_up",
        "signal",
        "position",
    }
    excluded.update(col for col in df.columns if col.startswith("sma_"))
    excluded.update(col for col in df.columns if col.startswith("volatility_"))
    return [col for col in df.columns if col not in excluded]
