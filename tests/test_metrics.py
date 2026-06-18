from __future__ import annotations

import pandas as pd
import pytest

from crypto_hf.metrics.performance import max_drawdown, sharpe_ratio, total_return


def test_total_return() -> None:
    equity = pd.Series([100.0, 110.0, 121.0])
    assert total_return(equity) == pytest.approx(0.21)


def test_max_drawdown_known_curve() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 95.0])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_sharpe_zero_volatility() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert sharpe_ratio(returns) == 0.0
