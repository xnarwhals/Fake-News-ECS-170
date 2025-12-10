"""
Generate slide-ready evaluation plots comparing the baseline TF-IDF model and the transformer model.

Outputs:
- reports/metrics_summary.csv
- reports/figures/confusion_matrices.png
- reports/figures/roc_pr_curves.png
- reports/figures/calibration_curve.png
- reports/figures/confidence_histograms.png
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")  # Ensure headless plotting works in restricted environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure repo root on path when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.data.load_datasets import load_true_fake_dataset
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.metrics import classification_report

plt.style.use("ggplot")

LABEL_NAMES = ["fake", "real"]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def stratified_sample(df: pd.DataFrame, n: int, *, random_state: int = 42) -> pd.DataFrame:
    """
    Take a stratified sample of n rows from df while preserving label balance.
    """
    if n is None or n <= 0 or n >= len(df):
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    counts = df["label"].value_counts()
    total = counts.sum()
    frames = []
    for label, count in counts.items():
        target = max(1, int(round(n * count / total)))
        sampled = df[df["label"] == label].sample(n=min(target, count), random_state=random_state)
        frames.append(sampled)
    sampled_df = pd.concat(frames).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df


def load_models():
    """
    Load the saved baseline (joblib) and transformer (HF format) models.
    """
    baseline_path = REPO_ROOT / "models" / "baseline.joblib"
    transformer_dir = REPO_ROOT / "models" / "distilbert"
    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline model at {baseline_path}. Train it before plotting.")
    if not transformer_dir.exists():
        raise SystemExit(f"Missing transformer directory at {transformer_dir}. Train it before plotting.")

    baseline_model = joblib.load(baseline_path)
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
    return baseline_model, transformer_model, tokenizer


def predict_baseline_probs(model, texts: Iterable[str]) -> np.ndarray:
    """Predict P(real) for the baseline model."""
    classes = list(model.classes_)
    idx_real = classes.index(1) if 1 in classes else 0
    probs = model.predict_proba(list(texts))[:, idx_real]
    return probs


def predict_transformer_probs(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Predict P(real) for the transformer model."""
    model.to(device)
    model.eval()
    probs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)


def plot_confusion_matrices(y_true, base_preds, transformer_preds, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for preds, title, ax in (
        (base_preds, "Baseline TF-IDF", axes[0]),
        (transformer_preds, "Transformer", axes[1]),
    ):
        cm = metrics.confusion_matrix(y_true, preds)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_roc_pr_curves(y_true, base_probs, transformer_probs, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # ROC
    for probs, label in ((base_probs, "Baseline"), (transformer_probs, "Transformer")):
        fpr, tpr, _ = metrics.roc_curve(y_true, probs)
        auc = metrics.roc_auc_score(y_true, probs)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    # Precision-Recall
    for probs, label in ((base_probs, "Baseline"), (transformer_probs, "Transformer")):
        precision, recall, _ = metrics.precision_recall_curve(y_true, probs)
        ap = metrics.average_precision_score(y_true, probs)
        axes[1].plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_calibration_curves(y_true, base_probs, transformer_probs, out_path: Path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 5))
    for probs, label in ((base_probs, "Baseline"), (transformer_probs, "Transformer")):
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        ece = expected_calibration_error(y_true, probs, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker="o", label=f"{label} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_confidence_histograms(y_true, base_probs, transformer_probs, out_path: Path):
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for probs, title, ax in (
        (base_probs, "Baseline TF-IDF", axes[0]),
        (transformer_probs, "Transformer", axes[1]),
    ):
        ax.hist(probs[y_true == 0], bins=bins, alpha=0.6, label="Fake", color="#d62728")
        ax.hist(probs[y_true == 1], bins=bins, alpha=0.6, label="Real", color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel("Predicted P(real)")
        ax.legend()
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(sample_size: int, test_size: float, batch_size: int, max_length: int):
    np.random.seed(42)
    torch.manual_seed(42)

    df = load_true_fake_dataset()
    if df.empty:
        raise SystemExit("No data found. Place True.csv and Fake.csv under data/raw/.")

    df = stratified_sample(df, sample_size, random_state=42)
    if df.empty:
        raise SystemExit("Sampling resulted in empty dataframe.")

    _, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    y_true = test_df["label"].to_numpy()
    texts = list(test_df["text"])

    baseline_model, transformer_model, tokenizer = load_models()
    base_probs = predict_baseline_probs(baseline_model, texts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_probs = predict_transformer_probs(
        transformer_model,
        tokenizer,
        texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    base_preds = (base_probs >= 0.5).astype(int)
    transformer_preds = (transformer_probs >= 0.5).astype(int)

    metrics_rows = [
        {"model": "baseline", **classification_report(y_true, base_preds, base_probs)},
        {"model": "transformer", **classification_report(y_true, transformer_preds, transformer_probs)},
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = FIGURES_DIR.parent
    reports_dir.mkdir(exist_ok=True)
    metrics_path = reports_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved metrics to", metrics_path)
    print(metrics_df)

    plot_confusion_matrices(y_true, base_preds, transformer_preds, FIGURES_DIR / "confusion_matrices.png")
    plot_roc_pr_curves(y_true, base_probs, transformer_probs, FIGURES_DIR / "roc_pr_curves.png")
    plot_calibration_curves(y_true, base_probs, transformer_probs, FIGURES_DIR / "calibration_curve.png")
    plot_confidence_histograms(y_true, base_probs, transformer_probs, FIGURES_DIR / "confidence_histograms.png")
    print("Saved figures to", FIGURES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison plots for baseline vs transformer models.")
    parser.add_argument("--sample-size", type=int, default=4000, help="Stratified sample size for faster evaluation.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to hold out for evaluation.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for transformer inference.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length for transformer tokenizer.")
    args = parser.parse_args()
    main(args.sample_size, args.test_size, args.batch_size, args.max_length)
