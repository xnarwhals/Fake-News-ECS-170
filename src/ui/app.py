"""
Streamlit UI for fake-news credibility scoring.
"""

import sys
from pathlib import Path

import streamlit as st
import joblib

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


@st.cache_resource
def load_model():
    """
    Load a pre-trained baseline if available; otherwise train from True/Fake dataset; fallback to tiny demo.
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    df = load_true_fake_dataset()
    if not df.empty:
        model = train_baseline(df["text"], df["label"])
        return model

    demo_texts = [
        "Breaking: government announces new verified policy to support science funding",
        "Exclusive: celebrity endorses miracle cure that scientists call fake",
        "Report: elections commission releases official and audited results",
        "Shocking proof that the moon landing was staged according to anonymous source",
    ]
    demo_labels = [1, 0, 1, 0]
    return train_baseline(demo_texts, demo_labels)


def predict(text: str):
    model = load_model()
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
        "Paste a headline/article or enter a URL. This demo uses a small TF-IDF baseline; swap with your trained model for the final submission."
    )

    input_mode = st.radio("Input mode", ["Text", "URL"], horizontal=True)
    text_input = st.text_area("Headline / Article", height=180) if input_mode == "Text" else ""
    url_input = st.text_input("URL") if input_mode == "URL" else ""

    if st.button("Analyze"):
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

        pred_label, prob_real, top_tokens = predict(text_input)
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
