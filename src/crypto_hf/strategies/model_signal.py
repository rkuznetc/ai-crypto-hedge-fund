from __future__ import annotations

import pandas as pd

BINARY_SIGNAL_MSG = "Signals must be binary long-only values: 0.0 or 1.0"


def _validate_binary_signals(signals: pd.Series) -> None:
    if signals.isna().any():
        raise ValueError(BINARY_SIGNAL_MSG)
    if not signals.isin([0.0, 1.0]).all():
        raise ValueError(BINARY_SIGNAL_MSG)


def signals_from_values(
    data: pd.DataFrame,
    signal_values: pd.Series,
    strict_alignment: bool = True,
) -> pd.DataFrame:
    """Build standardized signal/position columns from precomputed signals."""
    out = data.copy()
    aligned = signal_values.reindex(data.index)
    if strict_alignment:
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(
                f"Signals missing for {missing} date(s); strict_alignment=True requires "
                "a signal on every bar"
            )
    else:
        aligned = aligned.fillna(0.0)

    aligned = aligned.astype(float)
    _validate_binary_signals(aligned)
    out["signal"] = aligned
    out["position"] = out["signal"].shift(1).fillna(0.0)
    return out
