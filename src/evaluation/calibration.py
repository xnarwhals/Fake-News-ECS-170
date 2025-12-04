"""
Calibration utilities to quantify model confidence alignment.
"""

import numpy as np
from sklearn.calibration import calibration_curve


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary classification.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    total = np.sum(bin_counts)
    ece = 0.0
    for i in range(len(prob_true)):
        weight = bin_counts[i] / total if total else 0
        ece += weight * abs(prob_pred[i] - prob_true[i])
    return float(ece)
