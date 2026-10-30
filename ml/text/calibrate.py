"""Calibrate a trained text model's scores and choose its AI cutoff, using the validation set.

What it does (see ERROR_ANALYSIS.md for the reasoning):
  1. Scores every validation text and splits them into two halves, A and B
     (same mix of human/AI and writing types in each).
  2. Temperature scaling: finds one number T so that scores mean what they say
     ("80% AI" is right about 80% of the time). Fitted on half A.
  3. Chooses the cutoff on half B: the lowest one that keeps false positives
     (human texts called AI) at or below --target-fpr.
  4. Checks each step on the half it did NOT use, so no number is graded on its own answers:
     calibration error is measured on B; F1 and false-positive rate at the cutoff on A.
  5. Saves calibration.json (T and cutoff, for the server) and reliability.png (chart) to the model folder.

Dividing by T never changes which text scores higher than which, so it can't change F1 or the
false-positive rate. It changes the numbers: the cutoff lands somewhere stable instead of
0.94 vs 0.947, and the percentage shown in the extension is honest.

Run from the ml folder, after prepare_data.py:
    python text/calibrate.py --model-dir /content/drive/MyDrive/RealEye/models/<model folder>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_errors import margins
from mage import domain

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "text"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Folder with the saved model (from train_text.py)")
    parser.add_argument("--data", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument("--target-fpr", type=float, default=0.05,
                        help="Highest share of human texts allowed to be called AI (SRS limit is 0.10; "
                             "aiming lower leaves room for the test set to come out worse)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512, help="Use the same value as in training")
    parser.add_argument("--limit", type=int, default=None, help="Only use this many rows (for quick tests)")
    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(m, y, t):
    """Average penalty (cross-entropy) when margins m are divided by temperature t. Lower is better."""
    p = np.clip(sigmoid(m / t), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(m, y):
    """The T that makes scores match reality best (lowest loss), searched from 0.05 to 20."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    best = grid[np.argmin([loss(m, y, t) for t in grid])]
    fine = np.linspace(best * 0.97, best * 1.03, 200)  # zoom in around the best
    return float(fine[np.argmin([loss(m, y, t) for t in fine])])


def calibration_bins(p, y, n_bins=10):
    """Group texts by AI score (0-10%, 10-20%, ...). For each group: average score vs share actually AI."""
    which = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [(int(b), float(p[which == b].mean()), float(y[which == b].mean()), int((which == b).sum()))
            for b in range(n_bins) if (which == b).any()]


def ece(p, y, n_bins=10):
    """Expected calibration error: average gap between the score and how often it's AI, weighted by group size.
    0 = perfectly honest scores. 0.15 = scores are off by 15 percentage points on average."""
    return sum(abs(avg_p - frac_ai) * n for _, avg_p, frac_ai, n in calibration_bins(p, y, n_bins)) / len(p)


def choose_cutoff(p, y, target_fpr):
    """Lowest cutoff that calls at most target_fpr of human texts AI (score >= cutoff means AI)."""
    human = np.sort(p[y == 0])[::-1]  # highest first
    allowed = int(np.floor(target_fpr * len(human)))  # how many humans may be flagged
    if allowed >= len(human):
        return 0.0
    return float(np.nextafter(human[allowed], 1.0))  # just above the first human that must not be flagged


def metrics(p, y, cutoff):
    pred = p >= cutoff
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y),
    }


def save_chart(path, before, after, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping reliability.png)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1, label="Perfectly honest scores")
    for bins, label, color in [(before, "Before calibration", "#d1495b"), (after, "After calibration", "#2e86ab")]:
        ax.plot([b[1] for b in bins], [b[2] for b in bins], marker="o", color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Model's AI score (average in each group)",
           ylabel="Share of those texts that are actually AI", title=title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_json(args.data, lines=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=SEED)
    df["domain"] = df["src"].map(domain)
    df["margin"] = margins(df["text"].tolist(), args.model_dir, args.batch_size, args.max_length)

    # Two halves with the same mix of human/AI and writing types.
    half_a = df.groupby(["label", "domain"], group_keys=False).sample(frac=0.5, random_state=SEED)
    half_b = df.drop(half_a.index)
    m_a, y_a = half_a["margin"].to_numpy(), half_a["label"].to_numpy()
    m_b, y_b = half_b["margin"].to_numpy(), half_b["label"].to_numpy()
    print(f"\nHalf A: {len(half_a):,} texts (fit T, then check the cutoff)")
    print(f"Half B: {len(half_b):,} texts (choose the cutoff, check calibration)")

    # 1. Temperature, fitted on A.
    t = fit_temperature(m_a, y_a)
    raw_b, cal_b = sigmoid(m_b), sigmoid(m_b / t)
    ece_before, ece_after = ece(raw_b, y_b), ece(cal_b, y_b)
    print(f"\n=== 1. Temperature: T = {t:.3f} ===")
    print("T > 1 means the raw scores were too sure of themselves; T < 1 means they were too timid.")
    if not 0.06 < t < 19:
        print("WARNING: T is at the edge of the search range (0.05-20). Check the model scored the texts properly.")
    print(f"Calibration error (ECE, on half B): {ece_before:.3f} before -> {ece_after:.3f} after")
    print(f"Loss (on half B): {loss(m_b, y_b, 1):.4f} before -> {loss(m_b, y_b, t):.4f} after")
    print("\nScore group    before: share AI (texts)    after: share AI (texts)")
    before_bins = {b: (frac, n) for b, _, frac, n in calibration_bins(raw_b, y_b)}
    after_bins = {b: (frac, n) for b, _, frac, n in calibration_bins(cal_b, y_b)}
    for b in range(10):
        cells = [f"{before_bins[b][0]:>6.0%} ({before_bins[b][1]:>4})" if b in before_bins else f"{'-':>13}",
                 f"{after_bins[b][0]:>6.0%} ({after_bins[b][1]:>4})" if b in after_bins else f"{'-':>13}"]
        print(f"  {b * 10:>3}-{b * 10 + 10:<3}%   {cells[0]:>22}    {cells[1]:>22}")

    # 2. Cutoff, chosen on B (calibrated scores).
    cutoff = choose_cutoff(cal_b, y_b, args.target_fpr)
    raw_cutoff = float(sigmoid(np.log(cutoff / (1 - cutoff)) * t)) if 0 < cutoff < 1 else cutoff
    print(f"\n=== 2. Cutoff for <= {args.target_fpr:.0%} false positives (chosen on half B) ===")
    print(f"Calibrated cutoff: {cutoff:.4f}   (same as raw score {raw_cutoff:.4f} before calibration)")

    # 3. Check the cutoff on A, which played no part in choosing it.
    cal_a = sigmoid(m_a / t)
    check = metrics(cal_a, y_a, cutoff)
    print("\n=== 3. Check on half A (not used to choose the cutoff) ===")
    print(f"F1 {check['f1']:.3f} | false-positive rate {check['false_positive_rate']:.1%} | "
          f"recall {check['recall']:.1%} | precision {check['precision']:.1%} | accuracy {check['accuracy']:.1%}")
    print("Targets: F1 >= 0.75 and false-positive rate <= 10% (SRS 5.4.8, 5.4.20)")

    print("\nHumans flagged as AI on half A, by writing type:")
    flagged = (cal_a >= cutoff) & (y_a == 0)
    by_type = pd.DataFrame({"domain": half_a["domain"].to_numpy(), "human": y_a == 0, "flagged": flagged})
    for name, g in by_type[by_type["human"]].groupby("domain"):
        print(f"  {name:<8} {g['flagged'].mean():>5.0%}  ({g['flagged'].sum()} of {len(g)})")

    print("\nOther targets, for comparison (cutoff chosen on B, checked on A):")
    for target in (0.10, 0.07, 0.05, 0.03):
        c = choose_cutoff(cal_b, y_b, target)
        r = metrics(cal_a, y_a, c)
        print(f"  target {target:>4.0%}: cutoff {c:.4f} -> F1 {r['f1']:.3f}, "
              f"false positives {r['false_positive_rate']:.1%}, recall {r['recall']:.1%}")

    # 4. Save.
    model_dir = Path(args.model_dir)
    record = {
        "model": model_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "temperature": t,
        "cutoff": cutoff,
        "target_false_positive_rate": args.target_fpr,
        "how_to_use": "margin = logit[1] - logit[0]; ai_score = 1 / (1 + exp(-margin / temperature)); "
                      "label AI if ai_score >= cutoff",
        "raw_score_equivalent_of_cutoff": raw_cutoff,
        "validation_rows": len(df),
        "split": "validation halved by label and writing type (seed 42): T fit on A, cutoff chosen on B",
        "ece_on_half_b": {"before": ece_before, "after": ece_after},
        "metrics_on_half_a_at_cutoff": check,
    }
    (model_dir / "calibration.json").write_text(json.dumps(record, indent=2))
    save_chart(model_dir / "reliability.png", calibration_bins(raw_b, y_b), calibration_bins(cal_b, y_b),
               f"Reliability on validation half B (T = {t:.2f})")
    print(f"\nSaved {model_dir / 'calibration.json'} and reliability.png")
    print("Next: one final run on the test set with this T and cutoff.")


if __name__ == "__main__":
    main()
