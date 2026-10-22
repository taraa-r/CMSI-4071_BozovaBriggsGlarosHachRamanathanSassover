"""Build the cleaned, balanced text datasets used for training and evaluation.

Run from the ml folder:
    .venv\\Scripts\\python text\\prepare_data.py      (Windows)
    .venv/bin/python text/prepare_data.py             (Mac)

Writes to ml/data/text/ (not stored in git — everyone regenerates it; the seed makes it identical):
    train.jsonl       balanced sample: TRAIN_PER_CLASS human + TRAIN_PER_CLASS AI
    validation.jsonl  balanced sample, for choosing the threshold
    test.jsonl        ALL cleaned test texts — the final exam, only used once at the end
    manifest.json     counts, seed, and settings (SRS 5.3.4.11 evaluation record)
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from mage import MAX_CHARS, MIN_CHARS, clean, load_raw

SEED = 42
TRAIN_PER_CLASS = 10_000
VALIDATION_PER_CLASS = 2_500
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "text"
COLUMNS = ["text", "label", "src"]


def balanced_sample(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    parts = [df[df["label"] == label].sample(n=per_class, random_state=SEED) for label in (0, 1)]
    # Shuffle so human and AI rows are mixed.
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def save(df: pd.DataFrame, name: str) -> dict:
    df[COLUMNS].to_json(OUT_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    counts = df["label"].value_counts()
    print(f"{name:<11} {len(df):>7,} rows  (human {counts.get(0, 0):,} | AI {counts.get(1, 0):,})")
    return {"rows": len(df), "human": int(counts.get(0, 0)), "ai": int(counts.get(1, 0))}


def main():
    raw = load_raw()
    train, validation, test = (clean(raw[name]) for name in ["train", "validation", "test"])

    # Remove anything that also appears in a later split, so no split leaks into another.
    validation = validation[~validation["text"].isin(set(test["text"]))]
    train = train[~train["text"].isin(set(validation["text"]) | set(test["text"]))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "yaful/MAGE",
        "created": date.today().isoformat(),
        "seed": SEED,
        "labels": {"0": "human", "1": "AI"},
        "char_range": [MIN_CHARS, MAX_CHARS],
        "splits": {
            "train": save(balanced_sample(train, TRAIN_PER_CLASS), "train"),
            "validation": save(balanced_sample(validation, VALIDATION_PER_CLASS), "validation"),
            "test": save(test, "test"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
