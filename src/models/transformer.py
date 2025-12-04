"""
Transformer fine-tuning utilities (e.g., DistilBERT) for fake-news detection.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    lr: float = 2e-5
    epochs: int = 2
    batch_size: int = 16
    weight_decay: float = 0.01
    output_dir: str = "models/distilbert"


def build_trainer(texts, labels, config: Optional[TransformerConfig] = None) -> Trainer:
    """
    Build a HuggingFace Trainer for sequence classification.
    """
    cfg = config or TransformerConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    dataset = NewsDataset(texts, labels, tokenizer, max_length=cfg.max_length)
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        evaluation_strategy="no",
        save_strategy="no",
    )
    return Trainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)


def train_transformer(texts, labels, config: Optional[TransformerConfig] = None) -> AutoModelForSequenceClassification:
    """Train and return a fine-tuned transformer model."""
    trainer = build_trainer(texts, labels, config)
    trainer.train()
    return trainer.model
