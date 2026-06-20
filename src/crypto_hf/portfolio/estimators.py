from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def historical_mean_returns(
    returns: pd.DataFrame,
    annualization_factor: int = 365,
) -> pd.Series:
    """Annualized historical mean returns."""
    return returns.mean() * annualization_factor


def exponentially_weighted_mean_returns(
    returns: pd.DataFrame,
    span: int,
    annualization_factor: int = 365,
) -> pd.Series:
    """Annualized exponentially weighted mean returns."""
    ewm_mean = returns.ewm(span=span, adjust=False).mean().iloc[-1]
    return ewm_mean * annualization_factor


def sample_covariance(
    returns: pd.DataFrame,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    cov = returns.cov() * annualization_factor
    return _symmetrize_covariance(cov)


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrinkage covariance matrix."""
    clean = returns.dropna(how="any")
    if clean.empty:
        raise ValueError("Cannot estimate covariance from empty returns")
    model = LedoitWolf().fit(clean.to_numpy())
    cov = pd.DataFrame(
        model.covariance_ * annualization_factor,
        index=returns.columns,
        columns=returns.columns,
    )
    return _symmetrize_covariance(cov)


def _symmetrize_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Symmetrize covariance and clip tiny negative eigenvalues."""
    sym = (cov + cov.T) / 2.0
    values, vectors = np.linalg.eigh(sym.to_numpy())
    values = np.clip(values, 0.0, None)
    rebuilt = vectors @ np.diag(values) @ vectors.T
    return pd.DataFrame(rebuilt, index=cov.index, columns=cov.columns)
