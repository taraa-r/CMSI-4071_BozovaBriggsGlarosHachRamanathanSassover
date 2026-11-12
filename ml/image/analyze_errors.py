"""Find out where a trained image model goes wrong on the validation set, before changing anything.

The first full run (INITIAL_TRAINING_RESULTS.md) flags many more LAION real images than Pexels
ones (15% vs. 5% at a 10% overall false-positive rate). Calibration can't fix that: it moves one
cutoff for every image. This script shows what the false alarms have in common, so the fix can
be chosen on purpose:
  - mostly graphics/renders        -> extend the style cap (label_styles.py) to those styles
  - mostly low-quality web photos  -> more LAION in train, or weight LAION mistakes more
  - images that look AI-made       -> probably label noise in LAION; document it

It prints:
  1. Real images by source and CLIP style: how many, how many flagged, share of all false alarms.
  2. AI images by style and by generator: how many the model catches (recall).
  3. How sure the model is when it's wrong, and whether false alarms differ in file size or style.
It saves to <model-dir>/errors/:
  false_alarms_<source>_top.png     the 16 real images the model is MOST sure are AI
  false_alarms_<source>_random.png  16 random false alarms (more typical than the top ones)
  correct_<source>_random.png       16 random real images it got right, for comparison
  missed_ai_top.png                 the 16 AI images the model is most sure are real
  errors.csv                        every false alarm and miss, with source, style, score, size

Only validation is used; the test set is never read. Needs no GPU and doesn't rerun CLIP: it
reads validation_scores.csv (from train_clip.py) and the prepared images (from prepare_data.py).

Run from the ml folder:
    python image/analyze_errors.py --model-dir PATH [--data-dir PATH] [--target-fpr 0.10 | --cutoff X]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_shortcuts import contact_sheet

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "image"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True, help="folder with validation_scores.csv")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--target-fpr", type=float, default=0.10,
                        help="put the cutoff where this share of real images is flagged (default 0.10)")
    parser.add_argument("--cutoff", type=float, default=None, help="use this cutoff instead of --target-fpr")
    return parser.parse_args()


def load(args) -> pd.DataFrame:
    scores = pd.read_csv(args.model_dir / "validation_scores.csv")
    files = sorted(args.data_dir.glob("validation-*.parquet"))
    if not files:
        raise SystemExit(f"No validation-*.parquet in {args.data_dir}; check --data-dir.")
    images = pd.concat(pd.read_parquet(f, columns=["row_id", "image"]) for f in files)
    df = scores.merge(images, on="row_id", how="left")
    styles_path = args.data_dir / "styles.parquet"
    if styles_path.exists():
        probs = pd.read_parquet(styles_path, columns=["row_id", "split", "p_photo", "p_illustration",
                                                      "p_render", "p_graphic"])
        df = df.merge(probs[probs.split == "validation"].drop(columns="split"), on="row_id", how="left")
    missing = df["image"].isna().sum()
    if missing:
        print(f"Note: {missing} scored images not found in the prepared data; they're left out of the sheets.")
    return df


def table(groups, rows: pd.DataFrame, flagged: pd.Series, total_flagged: int, label: str) -> pd.DataFrame:
    out = rows.assign(hit=flagged).groupby(groups).agg(images=("hit", "size"), count=("hit", "sum"))
    out[f"% {label}"] = (out["count"] / out.images).map("{:.1%}".format)
    if total_flagged:
        out["share of all"] = (out["count"] / total_flagged).map("{:.0%}".format)
    return out.rename(columns={"count": label})


def sheets(rows: pd.DataFrame, name: str, out_dir: Path, top_ascending: bool | None, random_too: bool = True):
    rows = rows.dropna(subset=["image"])
    if rows.empty:
        return
    title = lambda r: f"{r.score:.2f} {r['style']} {r.kb:.0f}KB"
    picks = []
    if top_ascending is not None:
        picks.append(("top", rows.sort_values("score", ascending=top_ascending).head(16)))
    if random_too:
        picks.append(("random", rows.sample(n=min(16, len(rows)), random_state=SEED)))
    for kind, sample in picks:
        contact_sheet(list(sample["image"]), [title(r) for _, r in sample.iterrows()],
                      out_dir / f"{name}_{kind}.png")


def main():
    args = parse_args()
    df = load(args)
    real, ai = df[df.label == 0], df[df.label == 1]
    cutoff = args.cutoff if args.cutoff is not None else float(np.quantile(real.score, 1 - args.target_fpr))
    how = "given" if args.cutoff is not None else f"puts {args.target_fpr:.0%} of real images above it"
    print(f"Validation: {len(real):,} real | {len(ai):,} AI. Cutoff {cutoff:.3f} ({how}).")
    print("Labels on the contact sheets: score, CLIP style, file size.")

    false_alarm = real.score >= cutoff
    caught = ai.score >= cutoff
    n_fa = int(false_alarm.sum())
    print(f"\n=== 1. Real images wrongly flagged as AI: {n_fa:,} of {len(real):,} ({false_alarm.mean():.1%}) ===")
    print(table("model", real, false_alarm, n_fa, "flagged").rename_axis("source").to_string())
    print("\nBy source and style:")
    print(table(["model", "style"], real, false_alarm, n_fa, "flagged").rename_axis(["source", "style"]).to_string())

    print(f"\n=== 2. AI images caught (recall): {int(caught.sum()):,} of {len(ai):,} ({caught.mean():.1%}) ===")
    print(table("style", ai, caught, 0, "caught").to_string())
    print("\nBy generator (worst first):")
    by_gen = table("model", ai, caught, 0, "caught").rename_axis("generator")
    print(by_gen.loc[(by_gen["caught"] / by_gen.images).sort_values().index].to_string())

    print("\n=== 3. What the mistakes look like ===")
    for source in sorted(real.model.unique()):
        rows = real[real.model == source]
        fa, ok = rows[rows.score >= cutoff], rows[rows.score < cutoff]
        if fa.empty:
            continue
        print(f"{source}: {len(fa):,} false alarms; {(fa.score >= 0.9).mean():.0%} of them scored 0.9 or higher")
        print(f"  median file size: false alarms {fa.kb.median():.1f} KB vs. correct {ok.kb.median():.1f} KB")
        if "p_photo" in rows:
            print(f"  median CLIP 'photo' score: false alarms {fa.p_photo.median():.2f} vs. correct {ok.p_photo.median():.2f}"
                  "  (lower = looks less like a photo to CLIP)")
    missed = ai[ai.score < cutoff]
    print(f"AI missed: {len(missed):,}; {(missed.score < 0.1).mean():.0%} of them scored below 0.1")

    out_dir = args.model_dir / "errors"
    out_dir.mkdir(exist_ok=True)
    print(f"\nContact sheets in {out_dir}:")
    for source in sorted(real.model.unique()):
        rows = real[real.model == source]
        sheets(rows[rows.score >= cutoff], f"false_alarms_{source}", out_dir, top_ascending=False)
        sheets(rows[rows.score < cutoff], f"correct_{source}", out_dir, top_ascending=None)
    sheets(missed, "missed_ai", out_dir, top_ascending=True, random_too=False)

    columns = ["row_id", "label", "model", "style", "kb", "score"]
    errors = pd.concat([real[false_alarm].assign(error="false alarm"), missed.assign(error="missed AI")])
    errors[["error", *columns]].sort_values(["error", "score"]).to_csv(out_dir / "errors.csv", index=False)
    print(f"  Saved {out_dir / 'errors.csv'}")


if __name__ == "__main__":
    main()
