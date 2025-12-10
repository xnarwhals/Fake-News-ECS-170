"""
Dataset diagnostics: class balance, text length stats, and sample rows for True/Fake CSVs.
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure repo root on path when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.data.load_datasets import load_true_fake_dataset


def main():
    df = load_true_fake_dataset()
    if df.empty:
        raise SystemExit("No data found. Place True.csv and Fake.csv under data/raw/.")

    df["length"] = df["text"].str.split().apply(len)
    counts = df["label"].value_counts()
    print("Class balance (label: count):")
    print(counts.to_string())
    print(f"\nTotal examples: {len(df)}")
    print(f"Avg length: {df['length'].mean():.1f} tokens; Median length: {df['length'].median():.1f}")
    print(f"Min length: {df['length'].min()} tokens; Max length: {df['length'].max()} tokens")

    print("\nSample real (label=1):")
    print(df[df["label"] == 1].head(2)[["text"]].to_string(index=False, header=False)[:500])

    print("\nSample fake (label=0):")
    print(df[df["label"] == 0].head(2)[["text"]].to_string(index=False, header=False)[:500])


if __name__ == "__main__":
    main()
