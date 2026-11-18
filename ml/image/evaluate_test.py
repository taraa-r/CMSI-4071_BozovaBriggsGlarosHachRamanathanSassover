"""Final exam: score the held-out OpenFake test set ONCE with the saved calibration and cutoff.

Nothing is fitted or tuned here. It uses exactly what calibrate.py saved in calibration.json and
the head in head.pt, so these are the honest numbers for the report (SRS 5.4.9: F1 >= 0.75,
SRS 5.4.21: false-positive rate <= 10%). Run it only when the model, calibration and cutoff are
final: changing anything after seeing these results makes them optimistic.

The test set is out-of-distribution by design (DATASETS.md): its AI images come only from
generators never used in train or validation, and its real images from DOCCI and ImageNet
instead of LAION and Pexels. It is prepared like validation by prepare_data.py --splits test.

What it does:
  1. Checks: no test image is also in train or validation (same image_sha256), and which test
     generators, if any, were seen in train or validation.
  2. Runs CLIP once on every test image (a GPU makes it quick): its 512 features and its style
     (photo, illustration, render, graphic), the same way as for train and validation.
     The features are saved to <data-dir>/features/test.npy so a crash doesn't redo this.
  3. Scores: logit = head(features), estimate = 1 / (1 + exp(-(a * logit + b))), AI if >= cutoff.
  4. Prints F1, false-positive rate, recall, precision, accuracy and AUC with PASS/FAIL on the
     targets, next to validation; then results by real source, by style and by generator,
     calibration on test (ECE, log loss) and how sure the model was when wrong.
  5. Saves test_results.json, test_scores.csv, reliability_test.png and contact sheets of the
     mistakes (test_errors/) to the model folder.

Refuses to run a second time (test_results.json exists) unless you pass --force.

Run from the ml folder, after prepare_data.py --splits test and calibrate.py:
    python image/evaluate_test.py --model-dir /content/drive/MyDrive/RealEye/models/image/<model folder> \\
        --data-dir /content/drive/MyDrive/RealEye/data/image
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from calibrate import calibration_bins, ece, metrics, nll, print_bins_table, save_chart, sigmoid
from check_shortcuts import auc, contact_sheet
from extract_features import encode
from label_styles import CLIP_MODEL, CLIP_REVISION, STYLES
from train_clip import load_head

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "image"
SEED = 42
TARGET_F1, TARGET_FPR = 0.75, 0.10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True, help="folder with head.pt and calibration.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="folder with the prepared test-*.parquet")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="run again even though test results exist")
    return parser.parse_args()


def read_split(data_dir: Path, split: str, columns=None) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(pd.read_parquet(f, columns=columns) for f in files).reset_index(drop=True)


def clip_features(test: pd.DataFrame, data_dir: Path, batch_size: int) -> tuple[np.ndarray, pd.DataFrame]:
    """512 CLIP features and style probabilities per test image, computed once and cached."""
    features_dir = data_dir / "features"
    npy, info_path = features_dir / "test.npy", features_dir / "test.parquet"
    if npy.exists() and info_path.exists():
        info = pd.read_parquet(info_path)
        if list(info["row_id"]) == list(test["row_id"]):
            print(f"Using saved CLIP features ({npy}).")
            return np.load(npy).astype(np.float32), info
        print("Saved test features don't match the prepared test images; running CLIP again.")

    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running CLIP on {len(test):,} test images ({device})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    with torch.no_grad():
        text = processor(text=list(STYLES.values()), return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text).float()
        text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).cpu()

    out = np.zeros((len(test), 512), dtype=np.float16)
    for start in range(0, len(test), batch_size):
        batch = [Image.open(io.BytesIO(data)).convert("RGB") for data in test["image"].iloc[start:start + batch_size]]
        out[start:start + len(batch)] = encode(batch, model, processor, device)  # same as extract_features.py
        print(f"\r  {min(start + batch_size, len(test)):,} / {len(test):,}", end="")
    print()
    x = torch.from_numpy(out.astype(np.float32))
    x = x / x.norm(dim=-1, keepdim=True)
    probs = (100 * x @ text_features.T).softmax(dim=-1).numpy()  # same as label_styles.py
    info = pd.DataFrame(probs, columns=[f"p_{name}" for name in STYLES])
    info.insert(0, "row_id", test["row_id"].to_numpy())
    features_dir.mkdir(exist_ok=True)
    np.save(npy, out)
    info.to_parquet(info_path, index=False)
    return out.astype(np.float32), info


def training_validation_loss(model_dir: Path) -> dict | None:
    """The validation loss of the saved head when it was chosen (train_clip.py: lowest of all rounds)."""
    try:
        chosen = json.loads((model_dir / "config.json").read_text())["chosen"]
        history = json.loads((model_dir / "metrics.json").read_text())["validation_loss_by_weight_decay"]
        losses = history[str(float(chosen["weight_decay"]))]
        return {"loss": float(losses[chosen["epoch"] - 1]), "weight_decay": chosen["weight_decay"],
                "round": chosen["epoch"]}
    except (OSError, KeyError, IndexError, ValueError):
        return None


def summary_line(r):
    return (f"F1 {r['f1']:.3f} | false-positive rate {r['false_positive_rate']:.1%} | "
            f"recall {r['recall']:.1%} | precision {r['precision']:.1%} | accuracy {r['accuracy']:.1%}")


def balanced(df):
    n = int(df["label"].value_counts().min())
    return pd.concat([df[df["label"] == label].sample(n=n, random_state=SEED) for label in (0, 1)])


def main():
    args = parse_args()
    cal_path, out_path = args.model_dir / "calibration.json", args.model_dir / "test_results.json"
    if not cal_path.exists():
        sys.exit(f"No calibration.json in {args.model_dir}. Run calibrate.py first.")
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} already exists: this model was already tested.\n"
                 "The test set should be used once. Changing anything after seeing test results makes them\n"
                 "optimistic. To run again anyway (e.g. the first run crashed or used the wrong files), add --force.")
    cal = json.loads(cal_path.read_text())
    if cal.get("method") != "platt":
        sys.exit(f"calibration.json method is {cal.get('method')!r}; expected 'platt'. Rerun calibrate.py.")
    a, b, cutoff = cal["a"], cal["b"], cal["cutoff"]
    print(f"Using {cal.get('fitted_method', 'platt')} calibration (a = {a:.3f}, b = {b:.3f}) "
          f"and cutoff {cutoff:.4f} from {cal_path.name}")

    test = read_split(args.data_dir, "test")
    if test.empty:
        sys.exit(f"No test-*.parquet in {args.data_dir}. Run prepare_data.py --splits test first.")
    counts = test["label"].value_counts()
    print(f"Test set: {len(test):,} images (real {counts.get(0, 0):,} | AI {counts.get(1, 0):,}); "
          f"real sources: {', '.join(sorted(test[test.label == 0]['model'].unique()))}")

    # 1. Checks.
    earlier = pd.concat([read_split(args.data_dir, s, ["image_sha256", "label", "model"]) for s in ("train", "validation")])
    leaked = test["image_sha256"].isin(set(earlier["image_sha256"]))
    seen_generators = set(earlier[earlier["label"] == 1]["model"])
    test_generators = set(test[test["label"] == 1]["model"])
    overlap = sorted(test_generators & seen_generators)
    print("\n=== 1. Checks ===")
    print(f"Test images also in train or validation: {int(leaked.sum()):,}"
          + ("  (none: the model has not seen these images)" if not leaked.any() else ""))
    print(f"Test generators: {len(test_generators)}; also used in train or validation: "
          + (", ".join(overlap) if overlap else "none (all unseen, as expected)"))

    # 2. CLIP features and styles.
    print()
    features, info = clip_features(test, args.data_dir, args.batch_size)
    style_cols = [f"p_{name}" for name in STYLES]
    test = test.merge(info, on="row_id", how="left")
    test["style"] = test[style_cols].idxmax(axis=1).str.removeprefix("p_")
    test["kb"] = test["image"].map(len) / 1024

    # 3. Score with the saved head and calibration.
    x = torch.from_numpy(features)
    x = x / x.norm(dim=-1, keepdim=True)  # as in train_clip.load
    head, scale = load_head(args.model_dir / "head.pt", x.shape[-1])
    with torch.no_grad():
        logits = head((x - scale["mean"]) / scale["std"]).squeeze(1).numpy()
    test["logit"] = logits
    test["estimate"] = sigmoid(a * logits + b)
    m, y, p = test["logit"].to_numpy(), test["label"].to_numpy(), test["estimate"].to_numpy()
    p_raw = sigmoid(m)

    # 4. Results.
    overall = {**metrics(p, y, cutoff), "auc": float(auc(test["estimate"], test["label"]))}
    print("\n=== 2. Test results at the saved cutoff ===")
    print(f"All {len(test):,} images: {summary_line(overall)} | AUC {overall['auc']:.3f}")
    bal_metrics = None
    if counts.get(0, 0) != counts.get(1, 0):
        bal = balanced(test)
        bal_metrics = {**metrics(bal["estimate"].to_numpy(), bal["label"].to_numpy(), cutoff), "rows": len(bal)}
        print(f"Balanced {len(bal):,} (50/50): {summary_line(bal_metrics)}")
    f1_pass, fpr_pass = overall["f1"] >= TARGET_F1, overall["false_positive_rate"] <= TARGET_FPR
    print(f"\nF1 >= {TARGET_F1} (SRS 5.4.9):                 {'PASS' if f1_pass else 'FAIL'} ({overall['f1']:.3f})")
    print(f"False-positive rate <= {TARGET_FPR:.0%} (SRS 5.4.21):  "
          f"{'PASS' if fpr_pass else 'FAIL'} ({overall['false_positive_rate']:.1%})")
    val = cal.get("metrics_on_half_a_at_cutoff")
    if val:
        print(f"\nValidation (half A) for comparison: {summary_line(val)}")
    no_leak = None
    if leaked.any():
        kept = test[~leaked.to_numpy()]
        no_leak = metrics(kept["estimate"].to_numpy(), kept["label"].to_numpy(), cutoff)
        print(f"Without the {int(leaked.sum()):,} leaked images: {summary_line(no_leak)}")

    flagged = p >= cutoff
    real, ai = test.assign(flagged=flagged)[y == 0], test.assign(flagged=flagged)[y == 1]
    n_fa = int(real["flagged"].sum())

    print("\n=== 3. Real images wrongly flagged as AI, by source and style ===")
    by_source = {}
    for name, g in real.groupby("model"):
        by_source[name] = {"images": len(g), "flagged": int(g.flagged.sum()),
                           "false_positive_rate": float(g.flagged.mean()), "average_estimate": float(g.estimate.mean())}
        print(f"  {name:<10} {g.flagged.mean():>6.1%}  ({int(g.flagged.sum())} of {len(g)})   "
              f"average estimate {g.estimate.mean():.2f}")
    real_by_style = {}
    for (source, style), g in real.groupby(["model", "style"]):
        real_by_style[f"{source}/{style}"] = {"images": len(g), "false_positive_rate": float(g.flagged.mean())}
        share = g.flagged.sum() / n_fa if n_fa else 0
        print(f"  {source:<10} {style:<13} {g.flagged.mean():>6.1%}  ({int(g.flagged.sum())} of {len(g)}; {share:.0%} of all false alarms)")

    print("\n=== 4. AI images caught (recall), by style and by generator ===")
    ai_by_style = {}
    for name, g in ai.groupby("style"):
        ai_by_style[name] = {"images": len(g), "recall": float(g.flagged.mean())}
        print(f"  {name:<13} {g.flagged.mean():>6.1%}  ({int(g.flagged.sum())} of {len(g)})")
    by_generator = ai.groupby("model").agg(images=("flagged", "size"), caught=("flagged", "sum"))
    by_generator["recall"] = by_generator["caught"] / by_generator["images"]
    by_generator = by_generator.sort_values("recall")
    print("By generator (worst first; small groups vary a lot by chance):")
    for name, r in by_generator.iterrows():
        print(f"  {name:<40} {r.recall:>6.1%}  ({int(r.caught)} of {int(r.images)})")
    by_type = {}
    if "type" in ai and ai["type"].notna().any():
        print("By generator type:")
        for name, g in ai.groupby("type"):
            by_type[name] = {"images": len(g), "recall": float(g.flagged.mean())}
            print(f"  {name:<10} {g.flagged.mean():>6.1%}  ({int(g.flagged.sum())} of {len(g)})")

    print("\n=== 5. Calibration and overconfidence on test ===")
    ece_raw, ece_cal, loss_raw, loss_cal = ece(p_raw, y), ece(p, y), nll(p_raw, y), nll(p, y)
    print(f"{'':<14}{'ECE':>8}{'Log loss':>11}")
    print(f"{'Raw':<14}{ece_raw:>8.3f}{loss_raw:>11.4f}")
    print(f"{'Calibrated':<14}{ece_cal:>8.3f}{loss_cal:>11.4f}")
    val_b = cal.get("on_half_b", {}).get("Platt")
    val_b_raw = cal.get("on_half_b", {}).get("Raw")
    if val_b:
        print(f"Validation half B, calibrated: ECE {val_b['ece']:.3f}, log loss {val_b['log_loss']:.4f}"
              + (f" (raw: ECE {val_b_raw['ece']:.3f}, log loss {val_b_raw['log_loss']:.4f})" if val_b_raw else ""))
    training_val = training_validation_loss(args.model_dir)
    if training_val:
        print(f"Validation loss during training (raw, all validation, chosen round): {training_val['loss']:.4f} "
              f"(weight decay {training_val['weight_decay']}, round {training_val['round']})")
    print("Compare calibrated with calibrated and raw with raw. Higher on test = more often confidently wrong.")
    print_bins_table({"Raw": p_raw, "Calibrated": p}, y)
    wrong = flagged != (y == 1)
    sure = np.where(y[wrong] == 0, p[wrong], 1 - p[wrong])  # estimate given to the wrong answer
    confident = {f"over_{lvl}": float((sure > lvl).mean()) if wrong.any() else 0.0 for lvl in (0.9, 0.99)}
    print(f"\nWrong answers at the cutoff: {int(wrong.sum()):,} of {len(y):,}; "
          f"{confident['over_0.9']:.0%} of them over 90% sure, {confident['over_0.99']:.0%} over 99% sure")

    # 5. Save.
    record = {
        "model": args.model_dir.name,
        "evaluated": datetime.now().isoformat(timespec="seconds"),
        "calibration_used": {k: cal[k] for k in ("method", "fitted_method", "a", "b", "cutoff",
                                                  "target_false_positive_rate") if k in cal},
        "test_rows": len(test), "test_real": int(counts.get(0, 0)), "test_ai": int(counts.get(1, 0)),
        "checks": {"images_also_in_train_or_validation": int(leaked.sum()),
                   "test_generators": len(test_generators), "generators_seen_in_development": overlap},
        "metrics": overall,
        "metrics_balanced": bal_metrics,
        "metrics_without_leaked_images": no_leak,
        "targets": {"f1": TARGET_F1, "false_positive_rate": TARGET_FPR, "f1_pass": f1_pass, "fpr_pass": fpr_pass},
        "validation_for_comparison": {
            "metrics_half_a_at_cutoff": val,
            "half_b": {"calibrated": val_b, "raw": val_b_raw},  # ECE, log loss, Brier (calibrate.py)
            "training_validation_loss": training_val,         # raw logits, all validation (train_clip.py)
        },
        "false_positives_by_source": by_source,
        "false_positives_by_source_and_style": real_by_style,
        "recall_by_style": ai_by_style,
        "recall_by_generator": {k: {"images": int(r.images), "recall": float(r.recall)} for k, r in by_generator.iterrows()},
        "recall_by_generator_type": by_type,
        "ece": {"raw": ece_raw, "calibrated": ece_cal},
        "log_loss": {"raw": loss_raw, "calibrated": loss_cal},
        "wrong_answers": int(wrong.sum()),
        "confidence_of_wrong_answers": confident,
    }
    out_path.write_text(json.dumps(record, indent=2))
    test.sort_values("estimate", ascending=False)[["row_id", "label", "model", "type", "style", "kb", "logit", "estimate"]].to_csv(
        args.model_dir / "test_scores.csv", index=False)
    save_chart(args.model_dir / "reliability_test.png",
               {"Raw scores": calibration_bins(p_raw, y), "Calibrated": calibration_bins(p, y)},
               "Image model: reliability on the test set")
    errors_dir = args.model_dir / "test_errors"
    errors_dir.mkdir(exist_ok=True)
    title = lambda r: f"{r.estimate:.2f} {r['style']} {r.kb:.0f}KB"
    for source, g in real[real["flagged"]].groupby("model"):
        top = g.sort_values("estimate", ascending=False).head(16)
        contact_sheet(list(top["image"]), [title(r) for _, r in top.iterrows()], errors_dir / f"false_alarms_{source}_top.png")
    missed = ai[~ai["flagged"]].sort_values("estimate").head(16)
    if len(missed):
        contact_sheet(list(missed["image"]), [title(r) for _, r in missed.iterrows()], errors_dir / "missed_ai_top.png")
    print(f"\nSaved test_results.json, test_scores.csv, reliability_test.png and test_errors/ to {args.model_dir}")


if __name__ == "__main__":
    main()
