"""
Pipeline entrypoint for URL/text prediction: fetch -> clean -> model -> threshold.
"""

from pathlib import Path
from typing import Optional

import joblib
import langdetect

from src.data.url_scrapper import fetch_article
from src.data.text_cleaning import clean_text
from src.models.explainability import top_tfidf_features


class Predictor:
    def __init__(self, model_path: str = "models/baseline.joblib", min_tokens: int = 30, threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.min_tokens = min_tokens
        self.threshold = threshold
        if not self.model_path.exists():
            raise RuntimeError(f"Model not found at {self.model_path}. Train and save it first.")
        self.model = joblib.load(self.model_path)

    def detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except Exception:
            return "unknown"

    def predict_text(self, text: str):
        cleaned = clean_text(text)
        probs = self.model.predict_proba([cleaned])[0]
        classes = list(self.model.classes_)
        prob_real = float(probs[classes.index(1)]) if 1 in classes else 0.0
        pred_label = 1 if prob_real >= self.threshold else 0
        tokens = cleaned.split()
        warnings = []
        if len(tokens) < self.min_tokens:
            warnings.append("Text is very short; prediction may be unreliable.")
        lang = self.detect_language(text)
        if lang != "en":
            warnings.append(f"Detected language '{lang}'; model trained on English.")
        top_tokens = top_tfidf_features(self.model, cleaned, top_k=8)
        return {
            "pred_label": pred_label,
            "prob_real": prob_real,
            "top_tokens": top_tokens,
            "warnings": warnings,
            "language": lang,
        }

    def predict_url(self, url: str):
        article_text = fetch_article(url)
        return self.predict_text(article_text)
