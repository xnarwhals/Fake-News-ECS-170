"""
Streamlit UI for fake-news credibility scoring.
"""

import sys
from pathlib import Path

import streamlit as st
import joblib
import langdetect
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is on the path when run by Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.data.text_cleaning import clean_text
from src.data.url_scrapper import fetch_article
from src.data.load_datasets import load_true_fake_dataset
from src.models.baseline import train_baseline
from src.models.explainability import top_tfidf_features
from src.models.uncertainty import confidence, predictive_entropy

MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
MIN_TOKENS = 30
TRANSFORMER_DIR = PROJECT_ROOT / "models" / "distilbert"


@st.cache_resource
def load_model():
    """
    Load a pre-trained baseline from disk. No training on-the-fly to keep hosted app fast and deterministic.
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

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


def predict(text: str):
    # Prefer transformer if present
    transformer = load_transformer()
    if transformer:
        tokenizer, model = transformer
        cleaned = clean_text(text)
        inputs = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        prob_real = float(probs[1]) if probs.shape[0] > 1 else 0.0
        pred_label = 1 if prob_real >= 0.5 else 0
        top_tokens = []  # TODO: add transformer explainability if desired
        return pred_label, prob_real, top_tokens

    model = load_model()
    if model is None:
        raise RuntimeError("No trained model found. Please add models/baseline.joblib.")
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    # Ensure we map probabilities to the correct class index
    classes = list(model.classes_)
    prob_real = float(probs[classes.index(1)]) if 1 in classes else 0.0
    pred_label = int(model.predict([cleaned])[0])
    top_tokens = top_tfidf_features(model, cleaned, top_k=8)
    return pred_label, prob_real, top_tokens


def main():
    st.title("Fake News Credibility Checker (Demo)")
    st.write(
        "Paste a headline/article or enter a URL. The app uses a trained TF-IDF baseline (or DistilBERT if provided)."
    )

    input_mode = st.radio("Input mode", ["Text", "URL"], horizontal=True)
    text_input = st.text_area("Headline / Article", height=180) if input_mode == "Text" else ""
    url_input = st.text_input("URL") if input_mode == "URL" else ""

    threshold = st.slider("Decision threshold for 'Real' (prob >= threshold => Real)", 0.1, 0.9, 0.5, 0.05)

    if st.button("Analyze"):
        model = load_model()
        if model is None:
            st.error("No trained model found. Please add models/baseline.joblib (run scripts/train_baseline.py locally).")
            return

        if input_mode == "URL":
            if not url_input:
                st.warning("Please provide a URL.")
                return
            with st.spinner("Fetching article..."):
                try:
                    text_input = fetch_article(url_input)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to fetch URL: {exc}")
                    return
        if not text_input:
            st.warning("Please provide text.")
            return

        if len(text_input.split()) < MIN_TOKENS:
            st.warning("The fetched text is very short; predictions may be unreliable. Please provide a fuller article.")

        try:
            lang = langdetect.detect(text_input)
        except Exception:
            lang = "unknown"
        if lang != "en":
            st.warning(f"Detected language: {lang}. Model was trained on English; results may be unreliable.")

        pred_label, prob_real, top_tokens = predict(text_input)
        pred_label = 1 if prob_real >= threshold else 0
        st.subheader("Result")
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
