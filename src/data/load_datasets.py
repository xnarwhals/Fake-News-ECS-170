"""
Dataset loading helpers for FakeNewsNet/LIAR/Kaggle-style CSVs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.standardize import standardize_columns, standardize_labels
from src.data.text_cleaning import clean_text


def load_standardized_csv(path: str = "data/raw/standardized.csv") -> pd.DataFrame:
    """
    Load the prestandardized CSV shipped in the repo; returns empty DataFrame if missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=["text", "label", "source"])
    df = pd.read_csv(csv_path)
    expected = {"text", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    return df


def load_generic_csv(
    path: str,
    text_col: str,
    label_col: str,
    source_col: Optional[str] = None,
    positive_labels: Optional[list[str]] = None,
    negative_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV with arbitrary column names and remap to text/label/source.
    """
    df = pd.read_csv(path)
    df = standardize_columns(df, text_col=text_col, label_col=label_col, source_col=source_col)
    df = standardize_labels(df, label_col="label", positive_labels=positive_labels, negative_labels=negative_labels)
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    return df


def load_true_fake_dataset(
    true_path: str = "data/raw/True.csv",
    fake_path: str = "data/raw/Fake.csv",
) -> pd.DataFrame:
    """
    Load the Kaggle fake/real news CSV pair, combine title+text, attach labels.
    Labels: 1 = real/credible (True.csv), 0 = fake (Fake.csv).
    """
    true_csv = Path(true_path)
    fake_csv = Path(fake_path)
    if not true_csv.exists() or not fake_csv.exists():
        return pd.DataFrame(columns=["text", "label", "source"])

    true_df = pd.read_csv(true_csv)
    fake_df = pd.read_csv(fake_csv)

    # Combine title + text if both exist; else fallback to text
    def _combine(df: pd.DataFrame) -> pd.Series:
        if "title" in df.columns and "text" in df.columns:
            return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
        if "text" in df.columns:
            return df["text"].fillna("")
        raise ValueError("Expected 'text' column in dataset.")

    true_df = true_df.copy()
    fake_df = fake_df.copy()

    true_df["text"] = _combine(true_df)
    fake_df["text"] = _combine(fake_df)

    true_df["label"] = 1
    fake_df["label"] = 0
    true_df["source"] = "true_csv"
    fake_df["source"] = "fake_csv"

    df = pd.concat([true_df[["text", "label", "source"]], fake_df[["text", "label", "source"]]], ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df
