"""Train the small classifier on top of frozen CLIP features: real (0) or AI-generated (1).

Needs <data-dir>/features/ from extract_features.py. Uses only train + validation, never test.

The default classifier is one linear layer (512 numbers in, one score out), the "linear probe"
from Ojha et al. 2023, "Towards Universal Fake Image Detectors". With --head mlp it is two
layers instead (512 -> --hidden -> 1, with dropout), which can learn combinations of CLIP
features that a single straight boundary can't (ERROR_ANALYSIS.md). Each round it sees one random
version of every train image (clean or blur/JPEG-augmented). It tries a few weight-decay
settings (how strongly the weights are kept small) and keeps the setting and round with the
lowest validation loss. The AI cutoff is chosen later, in calibration; the 0.5 cutoff below
is only for a first look.

Run from the ml folder (seconds to minutes, CPU is fine):
    .venv/bin/python image/train_clip.py [--head mlp] [--data-dir PATH] [--compare-to OLD_MODEL_DIR]

--compare-to scores an earlier model (its head.pt, linear or mlp) on the same validation images,
so the two can be compared fairly even if the earlier one was validated on a different set of images.

Saves to ml/models/image/clip-<linear|mlp>-<date-time>/ (or --out-dir):
    head.pt                the trained head and feature scaling (mean, std). A linear head is stored
                           as weight, bias (as before); an mlp head adds "head", "hidden", "dropout"
                           and its layers under "state" (see load_head)
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
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear",
                        help="linear: one layer (default); mlp: two layers with dropout")
    parser.add_argument("--hidden", type=int, default=256, help="mlp only: size of the middle layer")
    parser.add_argument("--dropout", type=float, default=0.2, help="mlp only: share of the middle layer hidden each step")
    parser.add_argument("--compare-to", type=Path, default=None, help="earlier model folder to compare with")
    parser.add_argument("--target-fpr", type=float, default=0.10, help="false-positive rate for the comparison cutoff")
    return parser.parse_args()


def check_features_current(data_dir: Path, features_dir: Path) -> None:
    """Stop if the features were built for a different set of kept images (label_styles.py rerun)."""
    styles_path = data_dir / "styles.parquet"
    if not styles_path.exists():
        return
    styles = pd.read_parquet(styles_path, columns=["split", "row_id", "keep"])
    for split in ("train", "validation"):
        kept = set(styles[(styles.split == split) & styles.keep].row_id)
        have = set(pd.read_parquet(features_dir / f"{split}.parquet", columns=["row_id"]).row_id)
        if kept != have:
            raise SystemExit(f"The {split} features don't match the images kept in styles.parquet "
                             f"({len(have):,} vs. {len(kept):,}). Rerun extract_features.py first.")


def at_fpr(df: pd.DataFrame, target: float) -> dict:
    """Cutoff that flags `target` of the real images, and how the model does there."""
    real, ai = df[df.label == 0], df[df.label == 1]
    cutoff = float(np.quantile(real.score, 1 - target))
    tp, fp = int((ai.score >= cutoff).sum()), int((real.score >= cutoff).sum())
    recall, precision = tp / len(ai), tp / max(tp + fp, 1)
    return {
        "cutoff": round(cutoff, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "false_positive_rate_by_real_source": real.groupby("model").score.apply(
            lambda s: round(float((s >= cutoff).mean()), 3)).to_dict(),
        "recall_by_style": ai.groupby("style").score.apply(lambda s: round(float((s >= cutoff).mean()), 3)).to_dict(),
    }


def make_head(kind: str, dim: int, hidden: int = 256, dropout: float = 0.2) -> torch.nn.Module:
    """The classifier on top of CLIP: one straight boundary (linear) or two layers that can bend (mlp)."""
    if kind == "linear":
        return torch.nn.Linear(dim, 1)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, hidden),  # 512 CLIP numbers -> `hidden` mini-detectors
        torch.nn.GELU(),               # the bend: each mini-detector switches on only for its pattern
        torch.nn.Dropout(dropout),     # training only: randomly hides some, so none is relied on alone
        torch.nn.Linear(hidden, 1),    # mini-detectors -> one AI score
    )


def load_head(path: Path, dim: int) -> tuple[torch.nn.Module, dict]:
    """A saved head.pt (linear or mlp) in eval mode, plus its feature scaling (mean, std)."""
    saved = torch.load(path)
    kind = saved.get("head", "linear")  # files saved before --head existed are linear
    if kind == "linear":
        head = make_head("linear", dim)
        head.load_state_dict({"weight": saved["weight"], "bias": saved["bias"]})
    else:
        head = make_head(kind, dim, saved["hidden"], saved["dropout"])
        head.load_state_dict(saved["state"])
    return head.eval(), {"mean": saved["mean"], "std": saved["std"]}


def score_with(model_dir: Path, x_val_raw: torch.Tensor) -> np.ndarray:
    """Scores from a saved head.pt (with its own feature scaling) on L2-normalized features."""
    head, scale = load_head(model_dir / "head.pt", x_val_raw.shape[-1])
    with torch.no_grad():
        logits = head((x_val_raw - scale["mean"]) / scale["std"]).squeeze(1).numpy()
    return 1 / (1 + np.exp(-logits))


def print_comparison(rows: dict, target: float) -> None:
    print(f"\nSame validation images; first-look cutoff 0.5, then the cutoff that flags {target:.0%} of real images:")
    for name, (overall, fixed) in rows.items():
        sources = " | ".join(f"{s} FPR {v:.3f}" for s, v in fixed["false_positive_rate_by_real_source"].items())
        print(f"  {name:<9} @0.5: F1 {overall['f1']:.3f}, FPR {overall['false_positive_rate']:.3f}, "
              f"AUC {overall['auc']:.3f} || @{fixed['cutoff']:.3f}: recall {fixed['recall']:.3f}, "
              f"F1 {fixed['f1']:.3f} | {sources}")
        print(f"  {'':<9} recall by style @{fixed['cutoff']:.3f}: {fixed['recall_by_style']}")


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
    """Return (best validation loss, best epoch, best head state, loss per epoch)."""
    torch.manual_seed(SEED)
    layer = make_head(args.head, x_train.shape[-1], args.hidden, args.dropout)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    n, views = x_train.shape[0], x_train.shape[1]
    best, history = (float("inf"), 0, None), []
    for epoch in range(1, args.epochs + 1):
        view = torch.randint(views, (n,), generator=generator)  # one random version per image
        order = torch.randperm(n, generator=generator)
        layer.train()  # dropout on while learning
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(layer(x_train[idx, view[idx]]).squeeze(1), y_train[idx])
            loss.backward()
            optimizer.step()
        layer.eval()  # dropout off, so validation sees the whole model
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
    check_features_current(args.data_dir, features_dir)
    x_train, train_info = load(features_dir, "train")
    x_val, val_info = load(features_dir, "validation")
    x_val = x_val[:, 0]  # validation has only the clean version
    x_val_raw = x_val.clone()
    y_train = torch.tensor(train_info.label.values, dtype=torch.float32)
    y_val = torch.tensor(val_info.label.values, dtype=torch.float32)
    scale = scaler(x_train)
    x_train = (x_train - scale["mean"]) / scale["std"]
    x_val = (x_val - scale["mean"]) / scale["std"]
    head_desc = (f"linear {x_train.shape[-1]} -> 1" if args.head == "linear" else
                 f"mlp {x_train.shape[-1]} -> {args.hidden} (GELU, dropout {args.dropout}) -> 1")
    print(f"Train {len(y_train):,} images x {x_train.shape[1]} versions | validation {len(y_val):,} | head: {head_desc}")

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

    layer = make_head(args.head, x_train.shape[-1], args.hidden, args.dropout)
    layer.load_state_dict(best["state"])
    layer.eval()
    with torch.no_grad():
        logits = layer(x_val).squeeze(1).numpy()
    scores = val_info.assign(logit=logits, score=1 / (1 + np.exp(-logits)))

    overall = rates(scores)
    print(f"\nValidation (first-look cutoff 0.5): F1 {overall['f1']:.3f} | accuracy {overall['accuracy']:.3f} | "
          f"false-positive rate {overall['false_positive_rate']:.3f} | AUC {overall['auc']:.3f}")
    details = breakdown(scores)
    print("False-positive rate by real source:", details["false_positive_rate_by_real_source"])
    print("Score vs. file size (Spearman):", details["score_vs_file_size_spearman"])
    details[f"at_fpr_{args.target_fpr}"] = at_fpr(scores, args.target_fpr)
    if args.compare_to:
        old = val_info.assign(score=score_with(args.compare_to, x_val_raw))
        details["comparison"] = {"model": str(args.compare_to), "validation": rates(old),
                                 f"at_fpr_{args.target_fpr}": at_fpr(old, args.target_fpr)}
        print_comparison({"earlier": (rates(old), at_fpr(old, args.target_fpr)),
                          "new": (overall, details[f"at_fpr_{args.target_fpr}"])}, args.target_fpr)

    settings = json.loads((features_dir / "settings.json").read_text())
    out_dir = args.out_dir or MODELS_DIR / f"clip-{args.head}-{datetime.now():%Y%m%d-%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scaling = {"mean": scale["mean"], "std": scale["std"]}
    if args.head == "linear":
        saved = {**best["state"], **scaling}  # same format as before, so older readers still work
    else:
        saved = {"head": "mlp", "hidden": args.hidden, "dropout": args.dropout, "state": best["state"], **scaling}
    torch.save(saved, out_dir / "head.pt")
    manifest_path = args.data_dir / "manifest.json"
    (out_dir / "config.json").write_text(json.dumps({
        **settings, "head": f"{head_desc} on L2-normalized, then standardized (head.pt mean/std) features; score = sigmoid(logit)",
        "head_kind": args.head,
        "label_meaning": {"0": "real", "1": "AI"}, "preprocess": PREPROCESS,
        "chosen": {"weight_decay": wd, "epoch": best["epoch"]},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "weight_decays_tried": WEIGHT_DECAYS, "seed": SEED,
                     **({"hidden": args.hidden, "dropout": args.dropout} if args.head == "mlp" else {})},
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
