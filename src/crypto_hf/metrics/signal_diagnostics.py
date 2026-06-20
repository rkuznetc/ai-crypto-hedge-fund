from __future__ import annotations

import pandas as pd


def compute_signal_diagnostics(
    signals: pd.Series,
    positions: pd.Series,
    selected_threshold: float | None = None,
) -> dict[str, float]:
    """Compute signal-level diagnostics for a strategy."""
    aligned_signals = signals.reindex(positions.index).fillna(0.0)
    aligned_positions = positions.fillna(0.0)
    signal_changes = aligned_signals.diff().fillna(aligned_signals.iloc[0]).abs()

    holding_periods: list[int] = []
    current = 0
    for pos in aligned_positions:
        if pos == 1.0:
            current += 1
        elif current > 0:
            holding_periods.append(current)
            current = 0
    if current > 0:
        holding_periods.append(current)

    avg_holding = float(sum(holding_periods) / len(holding_periods)) if holding_periods else 0.0

    diagnostics = {
        "positive_signal_rate": float(aligned_signals.mean()),
        "exposure_time": float(aligned_positions.mean()),
        "number_of_long_days": float((aligned_positions == 1.0).sum()),
        "number_of_cash_days": float((aligned_positions == 0.0).sum()),
        "number_of_signal_changes": float(signal_changes.sum()),
        "average_holding_period": avg_holding,
    }
    if selected_threshold is not None:
        diagnostics["selected_threshold"] = float(selected_threshold)
    return diagnostics
