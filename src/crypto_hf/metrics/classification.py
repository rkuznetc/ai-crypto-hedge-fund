from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute diagnostic classification metrics for model predictions."""
    y_true_arr = y_true.astype(int).to_numpy()
    y_pred_arr = y_pred.astype(int).to_numpy()

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        if len(np.unique(y_true_arr)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba))
        else:
            metrics["roc_auc"] = 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    metrics["confusion_tn"] = float(cm[0, 0])
    metrics["confusion_fp"] = float(cm[0, 1])
    metrics["confusion_fn"] = float(cm[1, 0])
    metrics["confusion_tp"] = float(cm[1, 1])
    return metrics
