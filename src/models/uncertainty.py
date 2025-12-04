"""
Uncertainty estimation helpers.
"""

import numpy as np
from scipy.stats import entropy


def predictive_entropy(probabilities: np.ndarray) -> float:
    """Compute predictive entropy for binary class probabilities."""
    probs = np.clip(probabilities, 1e-8, 1 - 1e-8)
    return float(entropy([probs, 1 - probs], base=2))


def confidence(probabilities: np.ndarray) -> float:
    """Return max-class confidence."""
    return float(np.max(probabilities))
