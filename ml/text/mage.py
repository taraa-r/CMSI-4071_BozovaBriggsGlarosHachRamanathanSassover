"""Shared loading and cleaning for the MAGE dataset (see DATASETS.md)."""

import re

import pandas as pd
from datasets import load_dataset

MIN_CHARS = 200  # SRS 5.5.3, same limits as the extension
MAX_CHARS = 5000
SPLITS = ["train", "validation", "test"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_raw() -> dict[str, pd.DataFrame]:
    """All MAGE splits, whitespace-normalized, with labels flipped to 0 = human, 1 = AI."""
    mage = load_dataset("yaful/MAGE")
    splits = {}
    for name in SPLITS:
        df = mage[name].to_pandas()
        df["text"] = df["text"].map(normalize)
        df["label"] = 1 - df["label"]  # MAGE uses 1 = human
        df["chars"] = df["text"].str.len()
        splits[name] = df
    return splits


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 200-5000 character texts and drop exact duplicates."""
    in_range = df[(df["chars"] >= MIN_CHARS) & (df["chars"] <= MAX_CHARS)]
    return in_range.drop_duplicates(subset="text")
