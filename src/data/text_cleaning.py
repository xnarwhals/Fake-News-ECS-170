"""
Lightweight text cleaning helpers for fake-news detection.
"""

import re
import string
from typing import Iterable

# Minimal stopword list to avoid extra dependencies; expand during training.
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "that",
    "this",
    "with",
    "as",
}


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines and trim ends."""
    return re.sub(r"\s+", " ", text).strip()


def strip_punctuation(text: str) -> str:
    """Remove punctuation characters while preserving spaces."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Drop common stopwords from a token sequence."""
    return [tok for tok in tokens if tok not in STOPWORDS]


def clean_text(text: str) -> str:
    """
    Basic cleaning pipeline: lowercase, strip punctuation, remove stopwords, normalize whitespace.
    This keeps tokens simple for TF-IDF baselines; feel free to extend for stemming/lemmatization.
    """
    lowered = text.lower()
    no_punct = strip_punctuation(lowered)
    tokens = no_punct.split()
    filtered = remove_stopwords(tokens)
    return normalize_whitespace(" ".join(filtered))
