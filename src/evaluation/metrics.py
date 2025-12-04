"""
Evaluation metrics for fake-news classification.
"""

from typing import Dict

import numpy as np
from sklearn import metrics


def classification_report(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute standard metrics; include AUC if probabilities provided."""
    results = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        results["roc_auc"] = metrics.roc_auc_score(y_true, y_prob)
    return results


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix."""
    return metrics.confusion_matrix(y_true, y_pred)
