"""Look at the MAGE dataset before training: size, label balance, text lengths, duplicates.

Run from the ml folder:
    .venv\\Scripts\\python text\\explore_mage.py      (Windows)
    .venv/bin/python text/explore_mage.py             (Mac)

The first run downloads MAGE from Hugging Face (a few hundred MB) and caches it.
"""

import pandas as pd

from mage import MAX_CHARS, MIN_CHARS, SPLITS, clean, load_raw


def describe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    in_range = df[(df["chars"] >= MIN_CHARS) & (df["chars"] <= MAX_CHARS)]
    kept = clean(df)

    print(f"\n=== {name} ===")
    print(f"Rows:                     {len(df):>8,}")
    print(f"200-5000 characters:      {len(in_range):>8,}  ({len(in_range) / len(df):.0%})")
    print(f"  too short (<{MIN_CHARS}):       {(df['chars'] < MIN_CHARS).sum():>8,}")
    print(f"  too long (>{MAX_CHARS}):       {(df['chars'] > MAX_CHARS).sum():>8,}")
    print(f"Duplicates removed:       {len(in_range) - len(kept):>8,}")
    print(f"Kept:                     {len(kept):>8,}")

    counts = kept["label"].value_counts()
    human, ai = counts.get(0, 0), counts.get(1, 0)
    print(f"  human (0):              {human:>8,}  ({human / len(kept):.0%})")
    print(f"  AI (1):                 {ai:>8,}  ({ai / len(kept):.0%})")
    print(f"Median length (chars):    human {kept[kept.label == 0].chars.median():,.0f}"
          f" | AI {kept[kept.label == 1].chars.median():,.0f}")
    return kept


def main():
    raw = load_raw()
    splits = {name: describe(name, raw[name]) for name in SPLITS}

    # The same text in two splits would leak test answers into training.
    train_texts = set(splits["train"]["text"])
    print("\n=== Overlap between splits (should be 0) ===")
    for name in ["validation", "test"]:
        overlap = splits[name]["text"].isin(train_texts).sum()
        print(f"train & {name}: {overlap:,}")
    overlap = splits["validation"]["text"].isin(set(splits["test"]["text"])).sum()
    print(f"validation & test: {overlap:,}")

    print("\n=== One example of each (train) ===")
    for label, word in [(0, "HUMAN"), (1, "AI")]:
        row = splits["train"][splits["train"]["label"] == label].iloc[0]
        print(f"[{word}, source: {row['src']}] {row['text'][:300]}...\n")


if __name__ == "__main__":
    main()
