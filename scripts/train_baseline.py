pip install -r requirements.txt"""
Train a TF-IDF + Logistic Regression baseline on the Kaggle True/Fake dataset and save it for the Streamlit app.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure repo root on path when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.data.load_datasets import load_true_fake_dataset
from src.models.baseline import train_baseline
from src.evaluation.metrics import classification_report


def main():
    df = load_true_fake_dataset()
    if df.empty:
        raise SystemExit("No data found. Place True.csv and Fake.csv under data/raw/.")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    model = train_baseline(train_df["text"], train_df["label"])
    preds = model.predict(val_df["text"])
    probs = model.predict_proba(val_df["text"])[:, list(model.classes_).index(1)]

    report = classification_report(val_df["label"], preds, probs)
    print("Validation metrics:", report)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "baseline.joblib"
    joblib.dump(model, out_path)
    print(f"Saved baseline model to {out_path}")


if __name__ == "__main__":
    main()
