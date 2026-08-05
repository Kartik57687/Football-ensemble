"""
Evaluation metrics -- Section V.

The paper reports Accuracy, Precision, Recall and F1-score:

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)

It does not state the averaging scheme for the headline Table VI figures, but
quotes a macro-averaged F1 of 0.8540 in the confusion-matrix discussion. Both
macro and weighted averages are therefore computed and reported side by side.
See docs/unspecified_details.md #7.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .. import config


def compute_metrics(y_true, y_pred, average: str | None = None) -> dict:
    """Accuracy plus precision/recall/F1 under the requested averaging."""
    average = average or config.PRIMARY_AVERAGE
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "average": average,
    }


def compute_all_metrics(y_true, y_pred) -> dict:
    """Metrics under every averaging scheme in config.METRIC_AVERAGES."""
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in config.METRIC_AVERAGES:
        out[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        out[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        out[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    return out


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Per-class precision/recall/F1 -- the paper notes most works omit these."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(config.N_CLASSES)),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Confusion matrix in config.CLASS_NAMES order (Fig. 5)."""
    return confusion_matrix(y_true, y_pred, labels=list(range(config.N_CLASSES)))


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Assemble a Table VI-style comparison across models.

    `results` maps model name -> metrics dict from compute_all_metrics.
    """
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision (macro)": m["precision_macro"],
                "Recall (macro)": m["recall_macro"],
                "F1 (macro)": m["f1_macro"],
                "Precision (weighted)": m["precision_weighted"],
                "Recall (weighted)": m["recall_weighted"],
                "F1 (weighted)": m["f1_weighted"],
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def save_json(obj, path: Path | str) -> None:
    """Persist a metrics dict, converting NumPy scalars to native types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)


def format_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    """Render a metrics frame as percentages, matching the paper's tables."""
    return (df * 100).round(2)
