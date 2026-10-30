"""Look at what a trained text model gets wrong on the validation set.

Answers three questions:
  1. Which kinds of human writing get flagged as AI? (grouped by MAGE source, e.g. yelp, xsum)
  2. How confident is the model when it is wrong?
  3. Did validation loss go up between epochs? (only if trainer_state.json was kept)

Run from the ml folder, after prepare_data.py:
    python text/analyze_errors.py --model-dir /content/drive/MyDrive/RealEye/models/distilroberta-base-20261026-2310

Prints the tables and saves every validation score to <model dir>/validation_scores.csv,
so you can sort and read the mistakes yourself (e.g. in Google Sheets).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mage import domain

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "text"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Folder with the saved model (from train_text.py)")
    parser.add_argument("--data", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.999],
                        help="Cutoffs to compare (score >= cutoff means 'AI')")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512, help="Use the same value as in training")
    parser.add_argument("--limit", type=int, default=None, help="Only score this many rows (for quick tests)")
    return parser.parse_args()


def score(texts, model_dir, batch_size, max_length):
    """The model's AI score (softmax probability of label 1) for each text."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    print(f"Scoring {len(texts):,} texts on {device}...")

    scores = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(**batch).logits.float()
        scores += torch.softmax(logits, dim=-1)[:, 1].tolist()
        if (start // batch_size) % 20 == 0:
            print(f"  {start + len(batch['input_ids']):,} / {len(texts):,}")
    return scores


def print_by_domain(df, thresholds):
    human = df[df["label"] == 0]
    ai = df[df["label"] == 1]
    rows = []
    for name in sorted(df["domain"].unique()):
        h, a = human[human["domain"] == name], ai[ai["domain"] == name]
        row = {"source": name, "human texts": len(h)}
        for t in thresholds:
            row[f"humans flagged @{t}"] = f"{(h['score'] >= t).mean():.0%}" if len(h) else "-"
        row["AI texts"] = len(a)
        row[f"AI missed @{thresholds[0]}"] = f"{(a['score'] < thresholds[0]).mean():.0%}" if len(a) else "-"
        rows.append(row)
    table = pd.DataFrame(rows)
    flagged = human[human["score"] >= thresholds[0]]
    share = flagged["domain"].value_counts(normalize=True)
    table["share of all false positives"] = table["source"].map(lambda s: f"{share.get(s, 0):.0%}")
    print(table.to_string(index=False))


def print_overconfidence(df, cutoff):
    wrong = df[(df["score"] >= cutoff) != (df["label"] == 1)]
    confidence = wrong["score"].where(wrong["label"] == 0, 1 - wrong["score"])  # confidence in the wrong answer
    print(f"Wrong answers at cutoff {cutoff}: {len(wrong):,} of {len(df):,}")
    for level in (0.9, 0.99, 0.999):
        print(f"  ...made with over {level:.1%} confidence: {(confidence > level).mean():.0%}")


def print_loss_history(model_dir):
    metrics = Path(model_dir) / "metrics.json"
    states = sorted(Path(model_dir).rglob("trainer_state.json"))
    if metrics.exists() and "log_history" in json.loads(metrics.read_text()):
        history = json.loads(metrics.read_text())["log_history"]
    elif states:
        history = json.loads(states[-1].read_text())["log_history"]
    else:
        print("No loss history saved (models trained before Oct 29 didn't keep it).")
        print("Look for the per-epoch 'eval_loss' lines in the Colab output from the training run instead.")
        return
    for entry in history:
        if "eval_loss" in entry:
            print(f"  epoch {entry['epoch']:.0f}: validation loss {entry['eval_loss']:.4f}"
                  + (f", F1 {entry['eval_f1']:.3f}" if "eval_f1" in entry else ""))
    print("If validation loss went up from epoch 1 to 2, the second epoch made the model more overconfident.")


def main():
    args = parse_args()
    df = pd.read_json(args.data, lines=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)
    df["domain"] = df["src"].map(domain)
    df["score"] = score(df["text"].tolist(), args.model_dir, args.batch_size, args.max_length)

    print("\n=== 1. Which writing gets flagged? (by source) ===")
    print_by_domain(df, args.thresholds)

    print("\n=== 10 human texts the model thought were most AI-like ===")
    top = df[df["label"] == 0].sort_values("score", ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"  {row['score']:.5f}  [{row['src']}]  {row['text'][:150]}...")

    print("\n=== 2. How confident is it when wrong? ===")
    print_overconfidence(df, args.thresholds[0])

    print("\n=== 3. Validation loss per epoch ===")
    print_loss_history(args.model_dir)

    out = Path(args.model_dir) / "validation_scores.csv"
    df.sort_values("score", ascending=False)[["score", "label", "domain", "src", "text"]].to_csv(out, index=False)
    print(f"\nSaved all scores to {out}  (label 0 = human, 1 = AI)")


if __name__ == "__main__":
    main()
