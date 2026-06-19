#!/usr/bin/env python3
"""Run the baseline single-asset backtest pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_hf.config import load_config
from crypto_hf.pipeline.baseline import run_baseline_pipeline


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "baseline.yaml")
    outputs = run_baseline_pipeline(config, reports_dir=PROJECT_ROOT / "reports")

    print(f"Validated {outputs.validation.row_count} rows "
          f"({outputs.validation.start.date()} → {outputs.validation.end.date()})")
    print(f"Train: {len(outputs.train)} bars | Test: {len(outputs.test)} bars")
    print("\nMetrics (test set):")
    print(outputs.metrics_table.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nReports saved to reports/metrics/ and reports/figures/")


if __name__ == "__main__":
    main()
