from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_with_sma(
    data: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    title: str = "Price with SMA",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot close price with fast and slow SMA overlays."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data["close"], label="Close", linewidth=1.2)
    ax.plot(data.index, data[f"sma_{fast_window}"], label=f"SMA {fast_window}", alpha=0.8)
    ax.plot(data.index, data[f"sma_{slow_window}"], label=f"SMA {slow_window}", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_equity_curve(
    equity_curves: dict[str, pd.Series],
    title: str = "Equity Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one or more equity curves."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot drawdown series derived from an equity curve."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def export_metrics_table(
    metrics_by_strategy: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and optionally save a metrics comparison table."""
    table = pd.DataFrame(metrics_by_strategy).T
    table.index.name = "strategy"
    if save_path:
        table.to_csv(save_path)
    return table


def plot_metrics_table(
    metrics_by_strategy: dict[str, dict[str, float]],
    title: str = "Performance Metrics",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Render metrics as a matplotlib table figure."""
    table = export_metrics_table(metrics_by_strategy)
    display = table.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(12, 2 + 0.4 * len(display)))
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(
        cellText=display.values,
        rowLabels=display.index.tolist(),
        colLabels=display.columns.tolist(),
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
