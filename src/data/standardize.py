"""
Dataset standardization utilities to align multiple fake-news sources.
"""

from typing import Iterable, Mapping, Optional

import pandas as pd


def standardize_labels(
    df: pd.DataFrame,
    label_col: str,
    positive_labels: Optional[Iterable[str]] = None,
    negative_labels: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Normalize heterogeneous label strings into {0,1} where 1 = real/credible and 0 = fake.
    """
    pos = {lbl.lower() for lbl in (positive_labels or {"real", "true", "reliable", "1"})}
    neg = {lbl.lower() for lbl in (negative_labels or {"fake", "false", "unreliable", "0"})}

    def _map_label(raw: str) -> int:
        lowered = str(raw).lower()
        if lowered in pos:
            return 1
        if lowered in neg:
            return 0
        raise ValueError(f"Unmapped label: {raw}")

    df = df.copy()
    df["label"] = df[label_col].map(_map_label)
    return df


def standardize_columns(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    source_col: Optional[str] = None,
    extra_meta: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Rename columns to a unified schema: text, label, source, plus optional metadata mappings.
    """
    df = df.copy()
    rename_map = {text_col: "text", label_col: "label"}
    if source_col:
        rename_map[source_col] = "source"
    if extra_meta:
        rename_map.update(extra_meta)
    df = df.rename(columns=rename_map)
    keep_cols = [col for col in ["text", "label", "source"] if col in df.columns]
    keep_cols.extend(extra_meta.values() if extra_meta else [])
    return df[keep_cols]
