from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_hf.portfolio.base import OptimizationError
from crypto_hf.portfolio.weight_projection import project_bounded_simplex


def test_project_bounded_simplex_respects_max_weight() -> None:
    raw = pd.Series({"a": 0.7, "b": 0.2, "c": 0.1})
    weights = project_bounded_simplex(raw, min_weight=0.0, max_weight=0.5)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights <= 0.5 + 1e-9).all()
    assert weights["a"] == pytest.approx(0.5)


def test_project_bounded_simplex_infeasible_max_weight() -> None:
    raw = pd.Series({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    with pytest.raises(OptimizationError, match="infeasible"):
        project_bounded_simplex(raw, min_weight=0.0, max_weight=0.2)
