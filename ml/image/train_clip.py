"""Train the small classifier on top of frozen CLIP features: real (0) or AI-generated (1).

Needs <data-dir>/features/ from extract_features.py. Uses only train + validation, never test.

The classifier is one linear layer (512 numbers in, one score out), the "linear probe" from
Ojha et al. 2023, "Towards Universal Fake Image Detectors". Each round it sees one random
version of every train image (clean or blur/JPEG-augmented). It tries a few weight-decay
settings (how strongly the weights are kept small) and keeps the setting and round with the
lowest validation loss. The AI cutoff is chosen later, in calibration; the 0.5 cutoff below
is only for a first look.

Run from the ml folder (seconds to minutes, CPU is fine):
    .venv/bin/python image/train_clip.py [--data-dir PATH]

Saves to ml/models/image/clip-linear-<date-time>/ (or --out-dir):
    head.pt                the trained layer (weight, bias) and feature scaling (mean, std)
    config.json            CLIP version, feature and preprocessing settings, chosen settings
    metrics.json           validation results, overall and broken down (see below)
    validation_scores.csv  every validation image's score, for calibration
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from check_shortcuts import auc
from preprocess import SETTINGS as PREPROCESS

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data" / "image"
MODELS_DIR = ML_DIR / "models" / "image"
SEED = 42
WEIGHT_DECAYS = [0.0, 0.01, 0.1, 1.0]  # AdamW: shrinks weights by lr x this each step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def load(features_dir: Path, split: str):
    x = np.load(features_dir / f"{split}.npy").astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)  # CLIP features are compared by direction
    return torch.from_numpy(x), pd.read_parquet(features_dir / f"{split}.parquet")


def scaler(x_train: torch.Tensor) -> dict:
    """Per-feature mean and spread of the clean train images. After L2 normalization each of
    the 512 numbers is tiny (~0.04), which makes the layer learn very slowly; standardizing
    them (subtract mean, divide by spread) fixes that. Saved with the layer for the server."""
    clean = x_train[:, 0]
    return {"mean": clean.mean(dim=0), "std": clean.std(dim=0) + 1e-6}


def train(x_train, y_train, x_val, y_val, weight_decay, args):
    """Return (best validation loss, best epoch, best layer state, loss per epoch)."""
    torch.manual_seed(SEED)
    layer = torch.nn.Linear(x_train.shape[-1], 1)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    n, views = x_train.shape[0], x_train.shape[1]
    best, history = (float("inf"), 0, None), []
    for epoch in range(1, args.epochs + 1):
        view = torch.randint(views, (n,), generator=generator)  # one random version per image
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(layer(x_train[idx, view[idx]]).squeeze(1), y_train[idx])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = loss_fn(layer(x_val).squeeze(1), y_val).item()
        history.append(round(val_loss, 5))
        if val_loss < best[0]:
            best = (val_loss, epoch, {k: v.clone() for k, v in layer.state_dict().items()})
    return (*best, history)


def rates(df: pd.DataFrame) -> dict:
    """Counts and rates at the 0.5 first-look cutoff."""
    real, ai = df[df.label == 0], df[df.label == 1]
    tp, fn = int((ai.score >= 0.5).sum()), int((ai.score < 0.5).sum())
    fp, tn = int((real.score >= 0.5).sum()), int((real.score < 0.5).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "accuracy": round((tp + tn) / len(df), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "recall": round(recall, 4), "precision": round(precision, 4),
        "auc": round(auc(df.score, df.label), 4) if len(real) and len(ai) else None,
        "confusion": {"true_ai": tp, "missed_ai": fn, "false_alarm": fp, "true_real": tn},
    }


def breakdown(df: pd.DataFrame) -> dict:
    """Where the errors are: real sources and styles (false alarms), generators (misses)."""
    real, ai = df[df.label == 0], df[df.label == 1]
    by_generator = ai.groupby("model").score.apply(lambda s: round(float((s >= 0.5).mean()), 3))
    return {
        "false_positive_rate_by_real_source": real.groupby("model").score.apply(
            lambda s: round(float((s >= 0.5).mean()), 3)).to_dict(),
        "false_positive_rate_by_style": real.groupby("style").score.apply(
            lambda s: round(float((s >= 0.5).mean()), 3)).to_dict(),
        "recall_by_style": ai.groupby("style").score.apply(lambda s: round(float((s >= 0.5).mean()), 3)).to_dict(),
        "recall_by_generator": by_generator.sort_values().to_dict(),
        # Does the score just follow image detail? Near 0 is good (DATASETS.md, file-size gap).
        "score_vs_file_size_spearman": {
            "real": round(float(real[["score", "kb"]].corr(method="spearman").iloc[0, 1]), 3),
            "ai": round(float(ai[["score", "kb"]].corr(method="spearman").iloc[0, 1]), 3),
        },
    }


def main():
    args = parse_args()
    features_dir = args.data_dir / "features"
    x_train, train_info = load(features_dir, "train")
    x_val, val_info = load(features_dir, "validation")
    x_val = x_val[:, 0]  # validation has only the clean version
    y_train = torch.tensor(train_info.label.values, dtype=torch.float32)
    y_val = torch.tensor(val_info.label.values, dtype=torch.float32)
    scale = scaler(x_train)
    x_train = (x_train - scale["mean"]) / scale["std"]
    x_val = (x_val - scale["mean"]) / scale["std"]
    print(f"Train {len(y_train):,} images x {x_train.shape[1]} versions | validation {len(y_val):,}")

    runs = {}
    for wd in WEIGHT_DECAYS:
        loss, epoch, state, history = train(x_train, y_train, x_val, y_val, wd, args)
        runs[wd] = {"loss": loss, "epoch": epoch, "state": state, "history": history}
        print(f"  weight decay {wd:<7} best validation loss {loss:.4f} at round {epoch}")
    wd = min(runs, key=lambda k: runs[k]["loss"])
    best = runs[wd]
    print(f"Chosen: weight decay {wd}, round {best['epoch']}")
    if best["epoch"] == args.epochs:
        print(f"  Note: still improving at the last round; try again with --epochs {args.epochs * 2}.")

    layer = torch.nn.Linear(x_train.shape[-1], 1)
    layer.load_state_dict(best["state"])
    with torch.no_grad():
        logits = layer(x_val).squeeze(1).numpy()
    scores = val_info.assign(logit=logits, score=1 / (1 + np.exp(-logits)))

    overall = rates(scores)
    print(f"\nValidation (first-look cutoff 0.5): F1 {overall['f1']:.3f} | accuracy {overall['accuracy']:.3f} | "
          f"false-positive rate {overall['false_positive_rate']:.3f} | AUC {overall['auc']:.3f}")
    details = breakdown(scores)
    print("False-positive rate by real source:", details["false_positive_rate_by_real_source"])
    print("Score vs. file size (Spearman):", details["score_vs_file_size_spearman"])

    settings = json.loads((features_dir / "settings.json").read_text())
    out_dir = args.out_dir or MODELS_DIR / f"clip-linear-{datetime.now():%Y%m%d-%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({**best["state"], "mean": scale["mean"], "std": scale["std"]}, out_dir / "head.pt")
    manifest_path = args.data_dir / "manifest.json"
    (out_dir / "config.json").write_text(json.dumps({
        **settings, "head": "linear 512 -> 1 on L2-normalized, then standardized (head.pt mean/std) features; score = sigmoid(logit)",
        "label_meaning": {"0": "real", "1": "AI"}, "preprocess": PREPROCESS,
        "chosen": {"weight_decay": wd, "epoch": best["epoch"]},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "weight_decays_tried": WEIGHT_DECAYS, "seed": SEED},
        "dataset_revision": json.loads(manifest_path.read_text()).get("dataset_revision") if manifest_path.exists() else None,
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "validation": overall, **details,
        "validation_loss_by_weight_decay": {str(k): v["history"] for k, v in runs.items()},
        "train_images": len(y_train), "validation_images": len(y_val),
    }, indent=2))
    scores[["row_id", "label", "model", "style", "kb", "logit", "score"]].to_csv(out_dir / "validation_scores.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
