"""Final exam: score the held-out MAGE test set ONCE with the saved calibration and cutoff.

Nothing is fitted or tuned here. It uses exactly what calibrate.py saved in calibration.json,
so these are the honest numbers for the report (SRS 5.4.8: F1 >= 0.75, SRS 5.4.20: false-positive
rate <= 10%). Run it only when the model, calibration and cutoff are final: if you change
anything after seeing these results, the results stop being a fair test.

What it does:
  1. Leak check: counts test texts that also appear (exactly, or with the same first 200
     characters) in train.jsonl or validation.jsonl.
  2. Scores every test text: ai_score = 1 / (1 + exp(-(a * margin + b))), AI if >= cutoff.
  3. Prints F1, false-positive rate, recall, precision and accuracy with pass/fail on the targets,
     plus F1 on a balanced 50/50 sample (comparable to validation), results by writing type,
     and overconfidence checks: loss, calibration error (ECE) and how sure it was when wrong.
  4. Saves test_results.json, test_scores.csv and reliability_test.png to the model folder.

Refuses to run a second time (test_results.json exists) unless you pass --force.

Run from the ml folder, after prepare_data.py and calibrate.py:
    python text/evaluate_test.py --model-dir /content/drive/MyDrive/RealEye/models/<model folder>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_errors import margins
from calibrate import calibration_bins, ece, metrics, nll, print_bins_table, save_chart, sigmoid
from mage import domain

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "text"
SEED = 42
PREFIX_CHARS = 200  # texts sharing their first 200 characters count as near-copies
TARGET_F1, TARGET_FPR = 0.75, 0.10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Folder with the saved model and calibration.json")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Folder with train/validation/test .jsonl")
    parser.add_argument("--force", action="store_true", help="Run again even though test results exist")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512, help="Use the same value as in training")
    return parser.parse_args()


def calibrated_scores(m, cal):
    """Apply the saved calibration to raw margins."""
    if cal.get("method") == "platt":
        return sigmoid(cal["a"] * m + cal["b"])
    return sigmoid(m / cal["temperature"])  # older calibration.json files used temperature scaling


def leak_check(test, data_dir):
    """Which test texts also appear in train or validation (exactly, or with the same opening)."""
    seen = pd.concat([pd.read_json(Path(data_dir) / f"{name}.jsonl", lines=True)["text"]
                      for name in ("train", "validation")])
    exact = test["text"].isin(set(seen))
    prefix = test["text"].str[:PREFIX_CHARS].isin(set(seen.str[:PREFIX_CHARS]))
    return exact, prefix


def balanced(df):
    """Equal numbers of human and AI texts (the smaller class's count), so F1 is comparable to validation."""
    n = int(df["label"].value_counts().min())
    return pd.concat([df[df["label"] == label].sample(n=n, random_state=SEED) for label in (0, 1)])


def summary_line(r):
    return (f"F1 {r['f1']:.3f} | false-positive rate {r['false_positive_rate']:.1%} | "
            f"recall {r['recall']:.1%} | precision {r['precision']:.1%} | accuracy {r['accuracy']:.1%}")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    cal_path, out_path = model_dir / "calibration.json", model_dir / "test_results.json"
    if not cal_path.exists():
        sys.exit(f"No calibration.json in {model_dir}. Run calibrate.py first.")
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} already exists: this model was already tested.\n"
                 "The test set should be used once. Changing anything after seeing test results makes them\n"
                 "optimistic. To run again anyway (e.g. the first run crashed or used the wrong files), add --force.")
    cal = json.loads(cal_path.read_text())
    cutoff = cal["cutoff"]
    settings = f"a = {cal['a']:.3f}, b = {cal['b']:.3f}" if cal.get("method") == "platt" else f"T = {cal['temperature']:.3f}"
    print(f"Using {cal.get('method', 'temperature')} calibration ({settings}) and cutoff {cutoff:.4f} "
          f"from {cal_path.name}")

    test = pd.read_json(Path(args.data_dir) / "test.jsonl", lines=True)
    test["domain"] = test["src"].map(domain)
    counts = test["label"].value_counts()
    print(f"Test set: {len(test):,} texts (human {counts.get(0, 0):,} | AI {counts.get(1, 0):,})")

    # 1. Leak check.
    exact, prefix = leak_check(test, args.data_dir)
    print("\n=== 1. Leak check (test texts also in train or validation) ===")
    print(f"Exact copies: {int(exact.sum()):,}   Same first {PREFIX_CHARS} characters: {int(prefix.sum()):,}")
    if not prefix.any():
        print("None found: the model has not seen these texts.")

    # 2. Score.
    print()
    test["margin"] = margins(test["text"].tolist(), args.model_dir, args.batch_size, args.max_length)
    m, y = test["margin"].to_numpy(), test["label"].to_numpy()
    p_raw, p = sigmoid(m), calibrated_scores(m, cal)
    test["score"] = p

    # 3. Results.
    overall = metrics(p, y, cutoff)
    bal = balanced(test)
    bal_metrics = metrics(bal["score"].to_numpy(), bal["label"].to_numpy(), cutoff)
    print("\n=== 2. Test results at the saved cutoff ===")
    print(f"All {len(test):,} texts:      {summary_line(overall)}")
    print(f"Balanced {len(bal):,} (50/50): {summary_line(bal_metrics)}")
    print("(False-positive rate and recall don't depend on the human/AI mix; precision and F1 do.\n"
          " The balanced row is the fair comparison with validation, which was 50/50.)")
    f1_pass, fpr_pass = overall["f1"] >= TARGET_F1, overall["false_positive_rate"] <= TARGET_FPR
    print(f"\nF1 >= {TARGET_F1} (SRS 5.4.8):                 {'PASS' if f1_pass else 'FAIL'} ({overall['f1']:.3f})")
    print(f"False-positive rate <= {TARGET_FPR:.0%} (SRS 5.4.20):  "
          f"{'PASS' if fpr_pass else 'FAIL'} ({overall['false_positive_rate']:.1%})")

    val = cal.get("metrics_on_half_a_at_cutoff")
    if val:
        print(f"\nValidation (half A) for comparison: {summary_line(val)}")

    no_leak = None
    if prefix.any():
        kept = test[~prefix.to_numpy()]
        no_leak = metrics(kept["score"].to_numpy(), kept["label"].to_numpy(), cutoff)
        print(f"Without the {int(prefix.sum()):,} possible leaks: {summary_line(no_leak)}")

    print("\n=== 3. By writing type ===")
    print(f"{'type':<9}{'humans':>8}{'flagged':>9}{'AI':>8}{'missed':>8}")
    by_type = {}
    for name, g in test.groupby("domain"):
        h, a = g[g["label"] == 0], g[g["label"] == 1]
        flagged = float((h["score"] >= cutoff).mean()) if len(h) else None
        missed = float((a["score"] < cutoff).mean()) if len(a) else None
        by_type[name] = {"humans": len(h), "humans_flagged": flagged, "ai": len(a), "ai_missed": missed}
        pct = lambda x: f"{x:.1%}" if x is not None else "-"
        print(f"{name:<9}{len(h):>8,}{pct(flagged):>9}{len(a):>8,}{pct(missed):>8}")

    print("\n=== 4. Overconfidence and calibration on test ===")
    ece_raw, ece_cal = ece(p_raw, y), ece(p, y)
    loss_raw, loss_cal = nll(p_raw, y), nll(p, y)
    val_ece = cal.get("ece_on_half_b", {})
    val_loss = cal.get("loss_on_half_b", {})
    print(f"{'':<14}{'Loss':>8}{'ECE':>8}")
    print(f"{'Raw':<14}{loss_raw:>8.4f}{ece_raw:>8.3f}")
    print(f"{'Calibrated':<14}{loss_cal:>8.4f}{ece_cal:>8.3f}")
    if val_ece:
        print(f"Validation half B, calibrated: ECE {val_ece.get('platt', val_ece.get('after')):.3f}"
              + (f", loss {val_loss['platt']:.4f}" if "platt" in val_loss else " (loss 0.299 in ERROR_ANALYSIS.md)"))
    print("Loss: average penalty, much higher when the model is confidently wrong (lower is better).\n"
          "ECE: average gap between the score and how often it's AI (0 = perfectly honest).")

    # Confident mistakes: how sure the model was on the texts it got wrong at the cutoff.
    wrong = (p >= cutoff) != (y == 1)
    confident = {}
    print(f"\nWrong answers at the cutoff: {int(wrong.sum()):,} of {len(y):,}")
    for name, scores in [("raw", p_raw), ("calibrated", p)]:
        sure = np.where(y[wrong] == 0, scores[wrong], 1 - scores[wrong])  # score given to the wrong answer
        confident[name] = {f"over_{lvl}": float((sure > lvl).mean()) if wrong.any() else 0.0 for lvl in (0.9, 0.99)}
        print(f"  {name:<11} score: {confident[name]['over_0.9']:>4.0%} of them over 90% sure, "
              f"{confident[name]['over_0.99']:>4.0%} over 99% sure")
    print("(Oct 26 model on validation: 75% over 90%, 51% over 99%.)")
    print_bins_table({"Raw": p_raw, "Calibrated": p}, y)

    # 4. Save.
    record = {
        "model": model_dir.name,
        "evaluated": datetime.now().isoformat(timespec="seconds"),
        "calibration_used": {k: cal[k] for k in ("method", "a", "b", "temperature", "cutoff") if k in cal},
        "test_rows": len(test),
        "test_human": int(counts.get(0, 0)),
        "test_ai": int(counts.get(1, 0)),
        "leak_check": {"exact_copies": int(exact.sum()), f"same_first_{PREFIX_CHARS}_chars": int(prefix.sum())},
        "metrics": overall,
        "metrics_balanced": {**bal_metrics, "rows": len(bal)},
        "metrics_without_possible_leaks": no_leak,
        "targets": {"f1": TARGET_F1, "false_positive_rate": TARGET_FPR, "f1_pass": f1_pass, "fpr_pass": fpr_pass},
        "by_writing_type": by_type,
        "ece": {"raw": ece_raw, "calibrated": ece_cal},
        "loss": {"raw": loss_raw, "calibrated": loss_cal},
        "wrong_answers": int(wrong.sum()),
        "confidence_of_wrong_answers": confident,
    }
    out_path.write_text(json.dumps(record, indent=2))
    test.sort_values("score", ascending=False)[["score", "label", "domain", "src", "text"]].to_csv(
        model_dir / "test_scores.csv", index=False)
    save_chart(model_dir / "reliability_test.png",
               {"Raw scores": calibration_bins(p_raw, y), "Calibrated": calibration_bins(p, y)},
               "Reliability on the test set")
    print(f"\nSaved test_results.json, test_scores.csv and reliability_test.png to {model_dir}")


if __name__ == "__main__":
    main()
