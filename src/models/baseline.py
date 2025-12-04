"""
Baseline TF-IDF + Logistic Regression classifier for fake-news detection.
"""

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_model(max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)) -> Pipeline:
    """
    Create a scikit-learn pipeline with TF-IDF vectorizer and Logistic Regression classifier.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    clf = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_baseline(texts: Iterable[str], labels: Iterable[int]) -> Pipeline:
    """Fit the baseline pipeline and return the trained model."""
    model = build_baseline_model()
    model.fit(list(texts), list(labels))
    return model
