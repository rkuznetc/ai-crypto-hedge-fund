from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_portfolio_weights(
    weights: pd.DataFrame,
    title: str = "Portfolio Weights",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot portfolio weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(weights))))
    im = ax.imshow(weights.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(weights.columns)))
    ax.set_xticklabels(weights.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(weights.index)))
    ax.set_yticklabels(weights.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Return Correlation",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_return_scatter(
    metrics: pd.DataFrame,
    title: str = "Risk-Return Scatter",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot annualized volatility vs CAGR for portfolio methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = metrics["annualized_volatility"]
    y = metrics["cagr"]
    ax.scatter(x, y, s=60)
    offsets = [
        (6, 6),
        (-6, 6),
        (6, -6),
        (-6, -6),
        (10, 0),
        (0, 10),
        (-10, 0),
        (0, -10),
    ]
    for i, (name, row) in enumerate(metrics.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            name,
            (row["annualized_volatility"], row["cagr"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            ha="center",
        )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("CAGR")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_portfolio_metric_rankings(
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot horizontal ranking for one portfolio metric."""
    # ascending=True places the largest (best) value at the top of horizontal bars.
    series = metrics[metric].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(series))))
    ax.barh(series.index, series.values)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
