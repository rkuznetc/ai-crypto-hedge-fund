from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_hf.pipeline.baseline import split_train_test
from crypto_hf.portfolio.estimators import (
    historical_mean_returns,
    ledoit_wolf_covariance,
    sample_covariance,
)


def _returns(n: int = 120, assets: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    data = rng.normal(0.001, 0.02, size=(n, assets))
    cols = [f"ASSET{i}" for i in range(assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_historical_mean_returns_shape() -> None:
    returns = _returns()
    mu = historical_mean_returns(returns, annualization_factor=365)
    assert len(mu) == 3


def test_sample_covariance_shape_and_labels() -> None:
    returns = _returns()
    cov = sample_covariance(returns, annualization_factor=365)
    assert cov.shape == (3, 3)
    assert list(cov.index) == list(returns.columns)
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)


def test_ledoit_wolf_on_small_sample() -> None:
    returns = _returns(n=40)
    cov = ledoit_wolf_covariance(returns, annualization_factor=365)
    assert cov.shape == (3, 3)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigenvalues >= -1e-8).all()


def test_estimators_use_train_only() -> None:
    returns = _returns()
    train, test = split_train_test(returns, train_size=0.7)
    mu_train = historical_mean_returns(train, annualization_factor=365)
    cov_train = sample_covariance(train, annualization_factor=365)
    assert len(mu_train) == len(train.columns)
    assert cov_train.shape[0] == len(train.columns)
    assert len(test) > 0
