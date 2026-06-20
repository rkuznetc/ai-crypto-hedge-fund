#!/usr/bin/env python3
"""Run static multi-asset portfolio pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_hf.config import load_static_portfolio_config
from crypto_hf.pipeline.static_multi_asset_portfolio import run_static_multi_asset_portfolio_pipeline


def main() -> None:
    config = load_static_portfolio_config(
        PROJECT_ROOT / "configs" / "static_multi_asset_portfolio.yaml"
    )
    outputs = run_static_multi_asset_portfolio_pipeline(
        config,
        reports_dir=PROJECT_ROOT / "reports",
    )

    print(f"Aligned rows: {outputs.alignment_rows}")
    print(f"Train: {len(outputs.train_returns)} | Test: {len(outputs.test_returns)}")
    print("\nWeights:")
    print(outputs.weights.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMetrics (test):")
    print(outputs.metrics.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nReports saved to reports/metrics/ and reports/figures/")


if __name__ == "__main__":
    main()
