"""
Train a DistilBERT classifier on the True/Fake dataset and save model+tokenizer for the UI.
"""

import sys
import os
from pathlib import Path

# Disable TF/vision integrations before importing transformers to avoid missing deps.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.data.load_datasets import load_true_fake_dataset
from src.models.transformer import NewsDataset, TransformerConfig
from src.evaluation.metrics import classification_report


def main():
    df = load_true_fake_dataset()
    if df.empty:
        raise SystemExit("No data found. Place True.csv and Fake.csv under data/raw/.")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    texts_train, labels_train = list(train_df["text"]), list(train_df["label"])
    texts_val, labels_val = list(val_df["text"]), list(val_df["label"])

    cfg = TransformerConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    train_dataset = NewsDataset(texts_train, labels_train, tokenizer, max_length=cfg.max_length)
    val_dataset = NewsDataset(texts_val, labels_val, tokenizer, max_length=cfg.max_length)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Simple eval: use predicted labels for metrics
    val_logits = trainer.predict(val_dataset).predictions
    val_pred_labels = val_logits.argmax(axis=-1)
    report = classification_report(labels_val, val_pred_labels)
    print("Validation metrics:", report)

    save_dir = Path(cfg.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    trainer.model.save_pretrained(save_dir)
    print(f"Saved transformer model to {save_dir}")


if __name__ == "__main__":
    main()
