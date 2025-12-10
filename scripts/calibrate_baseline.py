"""
Calibrate a trained baseline model using Platt scaling (logistic) on a validation split.
Saves the calibrated model to models/baseline_calibrated.joblib for the Streamlit app.
"""

import sys
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
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

MIN_TOKENS_TRAIN = 50


def main():
    df = load_true_fake_dataset()
    if df.empty:
        raise SystemExit("No data found. Place True.csv and Fake.csv under data/raw/.")

    df["length"] = df["text"].str.split().apply(len)
    df = df[df["length"] >= MIN_TOKENS_TRAIN]
    if df.empty:
        raise SystemExit(f"No samples meet minimum length of {MIN_TOKENS_TRAIN} tokens.")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    # Split val into calibration and evaluation to avoid reusing the same data.
    calib_df, eval_df = train_test_split(val_df, test_size=0.5, stratify=val_df["label"], random_state=123)

    base_model = train_baseline(train_df["text"], train_df["label"])

    # Fit calibrator on a held-out split; base_model is already fitted.
    calibrator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
    calibrator.fit(calib_df["text"], calib_df["label"])

    preds = calibrator.predict(eval_df["text"])
    prob_real_idx = list(calibrator.classes_).index(1)
    probs = calibrator.predict_proba(eval_df["text"])[:, prob_real_idx]
    report = classification_report(eval_df["label"], preds, probs)
    print("Validation metrics (calibrated):", report)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "baseline_calibrated.joblib"
    joblib.dump(calibrator, out_path)
    print(f"Saved calibrated baseline model to {out_path}")


if __name__ == "__main__":
    main()
