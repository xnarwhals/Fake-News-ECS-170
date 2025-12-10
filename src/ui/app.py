"""Streamlit UI for fake-news credibility scoring."""

import sys
import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import joblib
import langdetect
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Avoid pulling TensorFlow/vision deps in hosted envs
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Ensure project root is on the path when run by Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.data.text_cleaning import clean_text
from src.data.url_scrapper import fetch_article
from src.models.explainability import top_tfidf_features
from src.models.uncertainty import confidence, predictive_entropy

USE_TRANSFORMER = True  # Prefer transformer for sharper predictions and higher confidence.
USE_CALIBRATED_BASELINE = False  # Prefer uncalibrated baseline unless explicitly enabled.
USE_ENSEMBLE = True  # Combine transformer + baseline probabilities for robustness.
TRANSFORMER_WEIGHT = 0.6
BASELINE_WEIGHT = 0.4
MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
CAL_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_calibrated.joblib"
MIN_TOKENS = 50
TRANSFORMER_DIR = PROJECT_ROOT / "models" / "distilbert"
TRUSTED_DOMAINS = {
    "bbc.co.uk",
    "bbc.com",
    "apnews.com",
    "reuters.com",
    "nytimes.com",
    "washingtonpost.com",
    "npr.org",
    "theguardian.com",
    "wsj.com",
}
DOMAIN_TRUST_BONUS = 0.3  # Boost for trusted hosts (clipped to [0,1])
TRUST_FLOOR = 0.6  # Minimum real prob for trusted hosts after bias.


@st.cache_resource
def load_model():
    """
    Load a pre-trained baseline from disk (prefers calibrated model if enabled).
    """
    if USE_CALIBRATED_BASELINE and CAL_MODEL_PATH.exists():
        return joblib.load(CAL_MODEL_PATH)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if CAL_MODEL_PATH.exists():
        return joblib.load(CAL_MODEL_PATH)
    return None


@st.cache_resource
def load_transformer():
    """
    Load a fine-tuned transformer if available; otherwise return None.
    """
    if TRANSFORMER_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
        model.eval()
        return tokenizer, model
    return None


def _apply_domain_bias(prob_real: float, host: str | None) -> float:
    """
    Boost probability for trusted domains; clamp to [0,1].
    """
    if not host:
        return prob_real
    normalized = host.lower().lstrip("www.")
    boosted = prob_real
    if normalized in TRUSTED_DOMAINS:
        boosted = min(1.0, prob_real + DOMAIN_TRUST_BONUS)
        boosted = max(boosted, TRUST_FLOOR)
    return boosted


def predict(text: str, host: str | None = None):
    baseline_model = load_model()
    transformer = load_transformer() if USE_TRANSFORMER else None

    baseline_prob = None
    top_tokens: list[tuple[str, float]] = []
    if baseline_model is not None:
        cleaned = clean_text(text)
        probs = baseline_model.predict_proba([cleaned])[0]
        classes = list(baseline_model.classes_)
        baseline_prob = float(probs[classes.index(1)]) if 1 in classes else 0.0
        top_tokens = top_tfidf_features(baseline_model, cleaned, top_k=8)

    transformer_prob = None
    if transformer:
        tokenizer, model = transformer
        cleaned_raw = text.strip()
        inputs = tokenizer(
            cleaned_raw,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label2id = {k.lower(): v for k, v in getattr(model.config, "label2id", {}).items()}
        real_idx = label2id.get("real", label2id.get("true", 1))
        real_idx = real_idx if real_idx is not None and real_idx < len(probs) else 1
        transformer_prob = float(probs[real_idx]) if len(probs) > real_idx else 0.0

    # Ensemble if enabled and both are present; else fall back to whichever exists.
    prob_real = 0.0
    if USE_ENSEMBLE and transformer_prob is not None and baseline_prob is not None:
        total_w = TRANSFORMER_WEIGHT + BASELINE_WEIGHT
        prob_real = (
            TRANSFORMER_WEIGHT * transformer_prob + BASELINE_WEIGHT * baseline_prob
        ) / total_w
    elif transformer_prob is not None:
        prob_real = transformer_prob
    elif baseline_prob is not None:
        prob_real = baseline_prob
    else:
        raise RuntimeError("No trained model found. Please add models/baseline.joblib or transformer weights.")

    prob_real = _apply_domain_bias(prob_real, host)
    pred_label = 1 if prob_real >= 0.5 else 0
    return pred_label, prob_real, top_tokens


def main():
    st.title("Fake News Credibility Checker (Demo)")
    st.write(
        "Paste a headline/article or enter a URL. The app uses a trained TF-IDF baseline (or DistilBERT if provided)."
    )

    input_mode = st.radio("Input mode", ["Text", "URL"], horizontal=True)
    text_input = st.text_area("Headline / Article", height=180) if input_mode == "Text" else ""
    url_input = st.text_input("URL") if input_mode == "URL" else ""

    threshold = st.slider("Decision threshold for 'Real' (prob >= threshold => Real)", 0.1, 0.9, 0.4, 0.05)

    host = None
    if st.button("Analyze"):
        if input_mode == "URL":
            if not url_input:
                st.warning("Please provide a URL.")
                return
            with st.spinner("Fetching article..."):
                try:
                    text_input = fetch_article(url_input)
                    host = urlparse(url_input).hostname
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to fetch URL: {exc}")
                    return
        if not text_input:
            st.warning("Please provide text.")
            return

        token_count = len(text_input.split())
        st.info(f"Fetched {token_count} tokens from the input/URL.")
        st.text_area("Preview (first 600 chars)", text_input[:600], height=200, key="preview_area")
        if token_count < MIN_TOKENS:
            st.error(
                f"The fetched text is too short ({token_count} tokens). Please provide a fuller article or a different URL."
            )
            st.info("Out-of-distribution: text length below minimum for training data.")
            return

        try:
            lang = langdetect.detect(text_input)
        except Exception:
            lang = "unknown"
        if lang != "en":
            st.warning(f"Detected language: {lang}. Model was trained on English; results may be unreliable.")

        pred_label, prob_real, top_tokens = predict(text_input, host=host)
        pred_label = 1 if prob_real >= threshold else 0
        st.subheader("Result")
        if host:
            st.caption(f"Host: {host}")
        st.write(f"Prediction: {'Real / Credible' if pred_label == 1 else 'Fake / Unreliable'}")
        st.write(f"Confidence (real prob): {prob_real:.2f}")
        st.write(f"Predictive entropy: {predictive_entropy(prob_real):.3f} (lower = more certain)")

        if top_tokens:
            st.subheader("Top Influential Tokens (TF-IDF)")
            for token, weight in top_tokens:
                st.write(f"- {token}: {weight:.4f}")

    st.divider()
    st.markdown(
        """
        **About**: This UI matches the project proposal—text cleaning, TF-IDF baseline, explainability, and uncertainty.
        Swap in the fine-tuned transformer + calibrated probabilities for your final app.
        """
    )


if __name__ == "__main__":
    main()
