from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.portfolio.base import OptimizationError, normalize_weights


def project_bounded_simplex(
    raw_weights: pd.Series,
    min_weight: float,
    max_weight: float,
    *,
    allow_short: bool = False,
) -> pd.Series:
    """Project non-negative weights onto the simplex with box constraints."""
    if raw_weights.isna().any():
        raise OptimizationError("Cannot project weights containing NaN")
    if not allow_short and (raw_weights < 0).any():
        raise OptimizationError("Negative raw weights are not allowed when allow_short=False")

    n_assets = len(raw_weights)
    if max_weight * n_assets < 1.0 - 1e-9:
        raise OptimizationError(
            f"max_weight={max_weight} is infeasible for {n_assets} assets (need max_weight * N >= 1)"
        )

    weights = normalize_weights(raw_weights.clip(lower=0.0))
    lower = 0.0 if not allow_short else min_weight

    for _ in range(n_assets * 20):
        if float(weights.max()) <= max_weight + 1e-9 and float(weights.min()) >= lower - 1e-9:
            return weights

        over = weights > max_weight
        if over.any():
            weights.loc[over] = max_weight
            free = ~over
            remaining = 1.0 - float(weights.loc[over].sum())
            if remaining <= 1e-12:
                raise OptimizationError("max_weight projection left no free weight mass")
            if not free.any():
                raise OptimizationError("All weights hit max_weight; constraints infeasible")
            free_sum = float(weights.loc[free].sum())
            if free_sum <= 0:
                raise OptimizationError("No positive free weights remain for redistribution")
            weights.loc[free] = weights.loc[free] / free_sum * remaining
            continue

        under = weights < lower
        if under.any():
            weights.loc[under] = lower
            free = ~under
            remaining = 1.0 - float(weights.loc[under].sum())
            if remaining <= 1e-12 or not free.any():
                raise OptimizationError("min_weight projection failed")
            free_sum = float(weights.loc[free].sum())
            weights.loc[free] = weights.loc[free] / free_sum * remaining

    raise OptimizationError("Failed to project weights onto bounded simplex")
