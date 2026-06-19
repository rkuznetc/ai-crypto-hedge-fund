#!/usr/bin/env python3
"""Run single-asset model comparison pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_hf.config import load_single_asset_models_config
from crypto_hf.pipeline.single_asset_models import run_single_asset_models_pipeline


def main() -> None:
    config = load_single_asset_models_config(PROJECT_ROOT / "configs" / "single_asset_models.yaml")
    outputs = run_single_asset_models_pipeline(config, reports_dir=PROJECT_ROOT / "reports")

    print(f"Validated {outputs.validation.row_count} rows")
    print(f"Train: {len(outputs.train)} | Test: {len(outputs.test)}")
    print(f"Features: {len(outputs.feature_columns)}")
    print("\nTrading metrics (test):")
    print(outputs.trading_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nClassification metrics (test):")
    print(outputs.classification_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nReports saved to reports/metrics/ and reports/figures/")


if __name__ == "__main__":
    main()
