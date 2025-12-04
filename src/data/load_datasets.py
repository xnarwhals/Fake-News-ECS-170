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
