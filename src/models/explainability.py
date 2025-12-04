"""
Explainability helpers for highlighting influential tokens.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def top_tfidf_features(model: Pipeline, text: str, top_k: int = 10) -> list[tuple[str, float]]:
    """
    Return top-k tokens by TF-IDF weight for a fitted TF-IDF + classifier pipeline.
    """
    if "tfidf" not in model.named_steps:
        return []
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    response = vectorizer.transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = tfidf_sorting[:top_k]
    return [(feature_array[i], float(response[0, i])) for i in top_n]
