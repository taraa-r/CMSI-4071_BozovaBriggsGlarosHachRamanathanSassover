"""Calibrate a trained image model's scores and choose its AI cutoff, using the validation set.

Same approach as text/calibrate.py, but it compares two methods and picks one with a rule set in
advance (ERROR_ANALYSIS.md, "Calibration"):
  1. Reads the saved validation logits (validation_scores.csv, from train_clip.py) and splits them
     into two halves, A and B, with the same mix of real/AI and of real sources and generators.
  2. Fits two methods on half A:
       temperature scaling  estimate = sigmoid(logit / T)       one number: stretches or squeezes
       Platt scaling        estimate = sigmoid(a * logit + b)   two numbers: stretch and shift
  3. Grades both, and the raw scores, on half B (which they never saw): ECE, log loss, Brier score.
     Decision rule: use temperature unless Platt is clearly better on half B, meaning its ECE is
     lower by more than 0.01 AND its log loss is lower. The simpler method wins ties.
  4. Chooses the cutoff on half B: the lowest one that flags at most --target-fpr of real images.
  5. Checks the cutoff on half A, which played no part in choosing it: F1, recall, false-positive
     rate, and the same broken down by real source (LAION, Pexels) and AI style.
  6. Saves calibration.json (for the server) and reliability.png to the model folder.

Both methods keep the order of the scores (an image that scored higher still scores higher), so the
choice doesn't change which images get flagged at a given false-positive rate. It changes what the
"Model estimate" shown in the extension means.

calibration.json always stores the result as Platt numbers a and b (temperature is a = 1/T, b = 0),
so the server's PlattCalibration reads either one the same way.

Only validation is used; the test set is never read. Needs no GPU and doesn't rerun CLIP.

Run from the ml folder:
    python image/calibrate.py --model-dir /content/drive/MyDrive/RealEye/models/image/<model folder>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ECE_MARGIN = 0.01  # Platt must beat temperature's ECE by more than this (and have lower loss) to be used


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True, help="folder with validation_scores.csv")
    parser.add_argument("--target-fpr", type=float, default=0.08,
                        help="highest share of real images allowed to be called AI (decided Nov 12: 0.08; "
                             "the SRS limit is 0.10)")
    parser.add_argument("--method", choices=["auto", "temperature", "platt"], default="auto",
                        help="auto applies the decision rule above; the others force a method")
    return parser.parse_args()


# --- the math (same as text/calibrate.py) --------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def nll(p, y):
    """Log loss: average penalty for the probability given to the right answer. Lower is better."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    """Average squared gap between the estimate and the truth (0 or 1). Lower is better."""
    return float(np.mean((p - y) ** 2))


def fit_temperature(m, y):
    """The T that makes sigmoid(m / T) match reality best (lowest loss), searched from 0.05 to 20."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    best = grid[np.argmin([nll(sigmoid(m / t), y) for t in grid])]
    fine = np.linspace(best * 0.97, best * 1.03, 200)
    return float(fine[np.argmin([nll(sigmoid(m / t), y) for t in fine])])


def fit_platt(m, y):
    """The a and b that make sigmoid(a * m + b) match reality best: one-feature logistic regression (Newton)."""
    x = np.column_stack([m, np.ones_like(m)])
    w = np.array([1.0, 0.0])
    for _ in range(100):
        p = sigmoid(x @ w)
        gradient = x.T @ (p - y) / len(y)
        hessian = (x.T * (p * (1 - p))) @ x / len(y) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def calibration_bins(p, y, n_bins=10):
    """Group images by estimate (0-10%, 10-20%, ...). For each group: average estimate vs share actually AI."""
    which = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [(int(b), float(p[which == b].mean()), float(y[which == b].mean()), int((which == b).sum()))
            for b in range(n_bins) if (which == b).any()]


def ece(p, y, n_bins=10):
    """Expected calibration error: average gap between the estimate and how often it's AI (0 = honest)."""
    return sum(abs(avg_p - frac_ai) * n for _, avg_p, frac_ai, n in calibration_bins(p, y, n_bins)) / len(p)


def choose_cutoff(p, y, target_fpr):
    """Lowest cutoff that calls at most target_fpr of real images AI (estimate >= cutoff means AI)."""
    real = np.sort(p[y == 0])[::-1]
    allowed = int(np.floor(target_fpr * len(real)))
    if allowed >= len(real):
        return 0.0
    return float(np.nextafter(real[allowed], 1.0))


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


# --- output ------------------------------------------------------------------------------------------

def print_bins_table(columns, y):
    tables = [{b: (frac, n) for b, _, frac, n in calibration_bins(p, y)} for p in columns.values()]
    print(f"\n{'Estimate':<12}" + "".join(f"{name + ': AI (images)':>28}" for name in columns))
    for b in range(10):
        cells = [f"{t[b][0]:>6.0%} ({t[b][1]:>4})" if b in t else "-" for t in tables]
        print(f"  {b * 10:>3}-{b * 10 + 10:<3}%  " + "".join(f"{c:>28}" for c in cells))


def save_chart(path, curves, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping reliability.png)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1, label="Perfectly honest estimates")
    for (label, bins), color in zip(curves.items(), ["#d1495b", "#edae49", "#2e86ab"]):
        ax.plot([b[1] for b in bins], [b[2] for b in bins], marker="o", color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Model estimate (average in each group)",
           ylabel="Share of those images that are actually AI", title=title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    path = args.model_dir / "validation_scores.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Train the model with train_clip.py first.")
    df = pd.read_csv(path)  # row_id, label (1 = AI), model (real source or generator), style, kb, logit, score
    df["label"] = df["label"].astype(int)

    # Two halves with the same mix of real/AI and of sources/generators ("model" is the real source
    # for real images and the generator for AI images).
    half_a = df.groupby(["label", "model"], group_keys=False).sample(frac=0.5, random_state=SEED)
    half_b = df.drop(half_a.index)
    m_a, y_a = half_a["logit"].to_numpy(), half_a["label"].to_numpy()
    m_b, y_b = half_b["logit"].to_numpy(), half_b["label"].to_numpy()
    print(f"Validation: {len(df):,} images ({(df.label == 0).sum():,} real, {(df.label == 1).sum():,} AI)")
    print(f"Half A: {len(half_a):,} (fit calibration, then check the cutoff)")
    print(f"Half B: {len(half_b):,} (grade calibration, choose the cutoff)")

    # 1. Fit both methods on A, grade on B.
    t = fit_temperature(m_a, y_a)
    a_platt, b_platt = fit_platt(m_a, y_a)
    est_b = {"Raw": sigmoid(m_b), "Temperature": sigmoid(m_b / t), "Platt": sigmoid(a_platt * m_b + b_platt)}
    scores = {name: {"ece": ece(p, y_b), "log_loss": nll(p, y_b), "brier": brier(p, y_b)}
              for name, p in est_b.items()}

    print("\n=== 1. Calibration (fitted on half A, graded on half B) ===")
    print(f"Temperature scaling: T = {t:.3f}  (T > 1 squeezes over-confident scores toward 50%)")
    print(f"Platt scaling:       a = {a_platt:.3f} (stretch), b = {b_platt:.3f} (shift; < 0 toward real, > 0 toward AI)")
    if not 0 < a_platt < 20:
        print("WARNING: Platt's a is outside the expected range (0-20). Check validation_scores.csv.")
    print(f"\n{'':<14}{'ECE':>8}{'Log loss':>11}{'Brier':>9}")
    for name, s in scores.items():
        print(f"{name:<14}{s['ece']:>8.3f}{s['log_loss']:>11.4f}{s['brier']:>9.4f}")
    print("ECE = average gap between the estimate and how often it's AI (0 = perfectly honest). Lower is better for all three.")
    print_bins_table(est_b, y_b)

    temp, platt = scores["Temperature"], scores["Platt"]
    platt_clearly_better = (temp["ece"] - platt["ece"] > ECE_MARGIN) and (platt["log_loss"] < temp["log_loss"])
    if args.method == "auto":
        method = "platt" if platt_clearly_better else "temperature"
        why = (f"Platt's ECE is lower by {temp['ece'] - platt['ece']:.3f} (> {ECE_MARGIN}) and its log loss is lower"
               if platt_clearly_better else
               f"Platt is not clearly better (ECE difference {temp['ece'] - platt['ece']:+.3f}, needs > {ECE_MARGIN}, "
               f"and lower log loss), so the simpler method is used")
    else:
        method, why = args.method, "chosen with --method"
    a, b = (a_platt, b_platt) if method == "platt" else (1 / t, 0.0)
    print(f"\nChosen: {method} ({why}).")
    print(f"Saved as Platt numbers: a = {a:.4f}, b = {b:.4f}")

    # 2. Cutoff, chosen on B with the chosen method.
    p_b, p_a = sigmoid(a * m_b + b), sigmoid(a * m_a + b)
    cutoff = choose_cutoff(p_b, y_b, args.target_fpr)
    raw_cutoff = float(sigmoid((logit(cutoff) - b) / a)) if 0 < cutoff < 1 else cutoff
    print(f"\n=== 2. Cutoff for <= {args.target_fpr:.0%} false positives (chosen on half B) ===")
    print(f"Calibrated cutoff: {cutoff:.4f}   (same as raw score {raw_cutoff:.4f} before calibration)")

    # 3. Check on A.
    check = metrics(p_a, y_a, cutoff)
    print("\n=== 3. Check on half A (not used to choose the cutoff) ===")
    print(f"F1 {check['f1']:.3f} | false-positive rate {check['false_positive_rate']:.1%} | "
          f"recall {check['recall']:.1%} | precision {check['precision']:.1%} | accuracy {check['accuracy']:.1%}")
    print("Targets: F1 >= 0.75 and false-positive rate <= 10% (SRS 5.4.9, 5.4.21)")

    flagged = p_a >= cutoff
    real_a = half_a.assign(flagged=flagged, estimate=p_a)[y_a == 0]
    ai_a = half_a.assign(flagged=flagged, estimate=p_a)[y_a == 1]
    print("\nReal images flagged as AI on half A, by source (and their average estimate):")
    by_source = {}
    for name, g in real_a.groupby("model"):
        by_source[name] = {"false_positive_rate": float(g.flagged.mean()), "average_estimate": float(g.estimate.mean())}
        print(f"  {name:<8} {g.flagged.mean():>6.1%}  ({int(g.flagged.sum())} of {len(g)})   average estimate {g.estimate.mean():.2f}")
    print("AI images caught on half A, by style:")
    by_style = {}
    for name, g in ai_a.groupby("style"):
        by_style[name] = float(g.flagged.mean())
        print(f"  {name:<13} {g.flagged.mean():>6.1%}  ({int(g.flagged.sum())} of {len(g)})")

    print("\nOther targets, for comparison (cutoff chosen on B, checked on A):")
    others = {}
    for target in (0.10, 0.08, 0.05):
        c = choose_cutoff(p_b, y_b, target)
        r = metrics(p_a, y_a, c)
        others[f"{target:.2f}"] = {"cutoff": c, **r}
        print(f"  target {target:>4.0%}: cutoff {c:.4f} -> F1 {r['f1']:.3f}, "
              f"false positives {r['false_positive_rate']:.1%}, recall {r['recall']:.1%}")

    # 4. Save.
    record = {
        "model": args.model_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "platt",
        "fitted_method": method,
        "a": a,
        "b": b,
        "cutoff": cutoff,
        "target_false_positive_rate": args.target_fpr,
        "how_to_use": "logit = head output; estimate = 1 / (1 + exp(-(a * logit + b))); label AI if estimate >= cutoff",
        "note": ("temperature scaling, stored as Platt with a = 1/T and b = 0" if method == "temperature"
                 else "Platt scaling"),
        "decision_rule": f"temperature unless Platt's ECE on half B is lower by more than {ECE_MARGIN} and its log loss is lower",
        "decision": why,
        "temperature": {"T": t},
        "platt": {"a": a_platt, "b": b_platt},
        "raw_score_equivalent_of_cutoff": raw_cutoff,
        "validation_rows": len(df),
        "split": "validation halved by label and source/generator (seed 42): calibration fit on A, cutoff chosen on B",
        "on_half_b": scores,
        "metrics_on_half_a_at_cutoff": check,
        "half_a_false_positives_by_source": by_source,
        "half_a_recall_by_style": by_style,
        "other_targets_on_half_a": others,
    }
    (args.model_dir / "calibration.json").write_text(json.dumps(record, indent=2))
    save_chart(args.model_dir / "reliability.png",
               {"Raw scores": calibration_bins(est_b["Raw"], y_b),
                f"Temperature (T = {t:.2f})": calibration_bins(est_b["Temperature"], y_b),
                f"Platt (a = {a_platt:.2f}, b = {b_platt:.2f})": calibration_bins(est_b["Platt"], y_b)},
               "Image model: reliability on validation half B")
    print(f"\nSaved {args.model_dir / 'calibration.json'} and reliability.png")
    print("Next: the one-time test set evaluation with this calibration and cutoff.")


if __name__ == "__main__":
    main()
