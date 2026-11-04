"""Build the cleaned, balanced text datasets used for training and evaluation.

Run from the ml folder:
    .venv\\Scripts\\python text\\prepare_data.py      (Windows)
    .venv/bin/python text/prepare_data.py             (Mac)

Options:
    --train-per-class N   up to N human AND N AI texts in train.jsonl (default 40,000;
                          the first run, Oct 26, used 10,000)
    --balance label       old behavior: only balance human vs AI overall. The default,
                          "domain", also balances within each writing type (yelp, xsum, ...)

Why balance by writing type: with "label", 33% of human training texts were Yelp reviews
but only 5% were news summaries (xsum), while AI texts were spread evenly. The model learned
"news summary = AI" and flagged 80% of human xsum texts. See ERROR_ANALYSIS.md.

Writes to ml/data/text/ (not stored in git — everyone regenerates it; the seed makes it identical):
    train.jsonl       balanced sample: the same number of human and AI texts in every writing type
    validation.jsonl  balanced sample, for choosing the threshold (same for every N)
    test.jsonl        ALL cleaned test texts — the final exam, only used once at the end
    manifest.json     counts, seed, and settings (SRS 5.3.4.11 evaluation record)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from mage import MAX_CHARS, MIN_CHARS, clean, domain, load_raw

SEED = 42
DEFAULT_TRAIN_PER_CLASS = 40_000
VALIDATION_PER_CLASS = 2_500
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "text"
COLUMNS = ["text", "label", "src"]


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    """Mix human and AI rows (and writing types) together."""
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """per_class human + per_class AI texts, ignoring writing type."""
    return shuffle(pd.concat([df[df["label"] == label].sample(n=per_class, random_state=SEED)
                              for label in (0, 1)]))


def domain_balanced_sample(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Equal human and AI texts within every writing type, up to per_class of each in total.

    Each writing type gets an equal share. A type with fewer human texts than its share
    (e.g. hswag) uses all of them and the same number of AI texts, so totals can come out
    under per_class.
    """
    df = df.assign(domain=df["src"].map(domain))
    domains = sorted(df["domain"].unique())
    share = per_class // len(domains)
    parts = []
    print(f"{'type':<9} {'per class':>9}")
    for name in domains:
        rows = df[df["domain"] == name]
        n = min(share, *(int((rows["label"] == label).sum()) for label in (0, 1)))
        print(f"{name:<9} {n:>9,}")
        parts += [rows[rows["label"] == label].sample(n=n, random_state=SEED) for label in (0, 1)]
    return shuffle(pd.concat(parts))


def save(df: pd.DataFrame, name: str) -> dict:
    df[COLUMNS].to_json(OUT_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    counts = df["label"].value_counts()
    print(f"{name:<11} {len(df):>7,} rows  (human {counts.get(0, 0):,} | AI {counts.get(1, 0):,})")
    return {"rows": len(df), "human": int(counts.get(0, 0)), "ai": int(counts.get(1, 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_TRAIN_PER_CLASS)
    parser.add_argument("--balance", choices=["domain", "label"], default="domain")
    args = parser.parse_args()

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
        "train_balance": args.balance,
        "splits": {
            "train": save((domain_balanced_sample if args.balance == "domain" else balanced_sample)(
                train, args.train_per_class), "train"),
            "validation": save(balanced_sample(validation, VALIDATION_PER_CLASS), "validation"),
            "test": save(test, "test"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
