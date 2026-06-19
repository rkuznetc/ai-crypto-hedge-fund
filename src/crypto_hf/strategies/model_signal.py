from __future__ import annotations

import pandas as pd


def signals_from_values(
    data: pd.DataFrame,
    signal_values: pd.Series,
) -> pd.DataFrame:
    """Build standardized signal/position columns from precomputed signals."""
    out = data.copy()
    aligned = signal_values.reindex(data.index).fillna(0.0).astype(float)
    out["signal"] = aligned
    out["position"] = out["signal"].shift(1).fillna(0.0)
    return out
