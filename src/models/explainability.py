"""
Explainability helpers for highlighting influential tokens.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def _unwrap_pipeline(model):
    """
    Return the underlying Pipeline if present (handles CalibratedClassifierCV wrapping a pipeline).
    """
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, CalibratedClassifierCV) and hasattr(model, "base_estimator"):
        base = model.base_estimator
        if isinstance(base, Pipeline):
            return base
    return None


def top_tfidf_features(model, text: str, top_k: int = 10) -> list[tuple[str, float]]:
    """
    Return top-k tokens by TF-IDF weight for a fitted TF-IDF + classifier pipeline.
    Safely handles calibrated wrappers.
    """
    pipeline = _unwrap_pipeline(model)
    if pipeline is None or "tfidf" not in pipeline.named_steps:
        return []
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    response = vectorizer.transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = tfidf_sorting[:top_k]
    return [(feature_array[i], float(response[0, i])) for i in top_n]
