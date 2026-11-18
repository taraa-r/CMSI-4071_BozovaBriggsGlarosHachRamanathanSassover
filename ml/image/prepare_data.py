"""Build the balanced, shortcut-free image datasets used for training and validation.

Run from the ml folder:
    .venv\\Scripts\\python image\\prepare_data.py      (Windows)
    .venv/bin/python image/prepare_data.py             (Mac)

Options:
    --train-per-class N        up to N real AND N AI images in train (default 20,000)
    --validation-per-class N   the same for validation (default 2,500)
    --test-per-class N         the same for test (default 2,500)
    --test-min-side N          test only: skip images whose shorter side is under N px (default 288,
                               the size every image is resized to, so nothing is enlarged)
    --splits train validation  which splits to build (default both, train first). The test split is
                               built only when asked for by name (--splits test), at the very end.
    --out-dir PATH             where to write (default ml/data/image; on Colab, a Drive folder)
    --workers N                row groups downloaded at once (default 8; use 2 on home wifi)
    --redo                     rebuild a split even if it is already done with these settings

If a run is cut off (e.g. Colab disconnects), run the same command again: finished splits are
skipped and an unfinished one continues from its last saved batch (<out-dir>/partial-<split>/).

Quick test on a laptop (downloads about 1 GB):
    .venv/bin/python image/prepare_data.py --train-per-class 200 --validation-per-class 100 --workers 2

How it works:
  1. Reads random row groups (chunks of ~70 mixed real and AI images) from OpenFake core,
     at the pinned revision, until it has 30% more images per class than needed.
  2. Skips duplicates (same image_sha256), the excluded generators below, and images whose
     shorter side is under 400 px (in both groups, so no image is ever upscaled). For test the
     limit is 288 px instead (--test-min-side), see below.
  3. Runs every image through preprocess.normalize_image: random crop, size and JPEG quality
     for train, fixed values for validation (see preprocess.py for why).
  4. Balances: the same number of real and AI images, and within each group every generator
     (or real source, LAION / Pexels) gets an equal share, up to what is available.
     Validation images that also appear in train are dropped; test images that also appear in
     train or validation are dropped.

The test split is the final exam. It is only read when asked for by name (--splits test), after
the model, calibration and cutoff are final. It is prepared exactly like validation (same filters,
fixed preprocessing, same balancing), so the two are comparable, with one difference: the minimum
size. The 400 px rule exists so random training crops are never enlarged; the server applies no such
rule, and many ImageNet photos (often about 500 x 375) would be dropped. Test therefore keeps every
image at least 288 px on its shorter side, the size the fixed preprocessing resizes to, so nothing
is enlarged and the test looks more like what the extension will see. Its AI images come only from
generators never used in train or validation, and its real images from DOCCI and ImageNet.

Writes to the output folder (not stored in git; the seed makes it identical for everyone):
    train-00000.parquet ...       columns: image (JPEG bytes), label (0 real, 1 AI), model,
    validation-00000.parquet ...  type, row_id, image_sha256 (of the original file)
    test-00000.parquet ...        (only with --splits test)
    manifest.json                 counts, skips, seed and settings (SRS 5.3.4.11)
"""

import argparse
import io
import json
import math
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from itertools import islice
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem
from PIL import Image

from openfake import LABELS, REVISION, normalize_type, read_group, row_groups, split_files
from preprocess import MIN_SOURCE_SIDE, SETTINGS, normalize_image

TEST_MIN_SIDE = SETTINGS["fixed"]["size"]  # 288: the fixed preprocessing never enlarges these

SEED = 42
DEFAULT_PER_CLASS = {"train": 20_000, "validation": 2_500, "test": 2_500}
SPLITS = ("train", "validation")  # built by default
ALL_SPLITS = SPLITS + ("test",)   # test only when named: it is used once, at the very end
MARGIN = 1.3          # collect 30% extra so balancing can trim the most common generators
BATCH_GROUPS = 16     # row groups per round, between progress updates
SHARD_ROWS = 2_000
COLUMNS = ["image", "label", "model", "type", "row_id", "image_sha256"]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "image"

# tiny-random-sana is a tiny, randomly initialized test model: its images are noise, not
# realistic AI images (checked Nov 10 with check_shortcuts.py --inspect-model).
EXCLUDED_MODELS = {"tiny-random-sana"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_PER_CLASS["train"])
    parser.add_argument("--validation-per-class", type=int, default=DEFAULT_PER_CLASS["validation"])
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_PER_CLASS["test"])
    parser.add_argument("--test-min-side", type=int, default=TEST_MIN_SIDE,
                        help=f"test only: smallest shorter side in px (default {TEST_MIN_SIDE}; train and "
                             f"validation always use {MIN_SOURCE_SIDE})")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=ALL_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--redo", action="store_true", help="rebuild splits that are already done")
    return parser.parse_args()


def water_fill(counts: dict, target: int) -> dict:
    """Split target across buckets as evenly as possible, never more than a bucket has.

    E.g. counts {a: 500, b: 40, c: 300}, target 400 -> {a: 180, b: 40, c: 180}.
    """
    if sum(counts.values()) <= target:
        return dict(counts)
    low, high = 0, max(counts.values())
    while low < high:  # largest equal cap that still fits in target
        mid = (low + high + 1) // 2
        if sum(min(c, mid) for c in counts.values()) <= target:
            low = mid
        else:
            high = mid - 1
    quota = {bucket: min(c, low) for bucket, c in counts.items()}
    leftover = target - sum(quota.values())
    for bucket in sorted(b for b, c in counts.items() if c > low)[:leftover]:
        quota[bucket] += 1
    return quota


def process_group(fs: HfFileSystem, path: str, group: int, split: str,
                  min_side: int = MIN_SOURCE_SIDE) -> tuple[list[dict], dict]:
    """Download one row group and prepare its usable images."""
    rows, skipped = [], {"excluded_model": 0, "too_small": 0, "unreadable": 0, "failed_groups": 0}
    downloaded = 0
    try:
        records = read_group(fs, path, group, COLUMNS)
    except Exception as error:  # still failing after huggingface_hub's own retries
        print(f"  Skipped a row group that failed to download ({type(error).__name__})")
        return [], {**skipped, "failed_groups": 1, "bytes": 0}
    for record in records:
        data = record["image"]["bytes"]
        downloaded += len(data)
        if record["model"] in EXCLUDED_MODELS:
            skipped["excluded_model"] += 1
            continue
        try:
            with Image.open(io.BytesIO(data)) as image:
                if min(image.size) < min_side:
                    skipped["too_small"] += 1
                    continue
            # Seeded per image, so results don't depend on download order.
            rng = random.Random(f"{SEED}-{record['image_sha256']}") if split == "train" else None
            prepared = normalize_image(data, rng)
        except Exception:
            skipped["unreadable"] += 1
            continue
        rows.append({"image": prepared, "label": LABELS[record["label"]], "model": record["model"],
                     "type": normalize_type(record["type"]), "row_id": record["row_id"],
                     "image_sha256": record["image_sha256"]})
    return rows, {**skipped, "bytes": downloaded}


def resume(partial_dir: Path, need: int, exclude_sha: set, min_side: int):
    """Pick up a run that was cut off (e.g. a Colab disconnect) from its saved batches.

    Returns (rows, stats, batches done), or None to start fresh. The row-group order is fixed by
    the seed, so skipping the batches already done gives exactly the same result as an
    uninterrupted run.
    """
    progress_path = partial_dir / "progress.json"
    if not progress_path.exists():
        return None
    progress = json.loads(progress_path.read_text())
    if (progress["need"] != need or progress["exclude_count"] != len(exclude_sha)
            or progress.get("min_side", MIN_SOURCE_SIDE) != min_side):
        print("  Found an unfinished run with different settings; starting fresh.")
        return None
    files = [partial_dir / f"batch-{i:05d}.parquet" for i in range(progress["batches"])]
    rows = pd.concat(pd.read_parquet(f) for f in files).to_dict("records") if files else []
    print(f"  Resuming an unfinished run: {progress['batches']} batches, {len(rows):,} images already done.")
    return rows, progress["stats"], progress["batches"]


def collect(fs: HfFileSystem, split: str, per_class: int, exclude_sha: set, min_side: int,
            workers: int, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read row groups until each class has per_class * MARGIN usable, unique images.

    Each batch is saved to <out-dir>/partial-<split>/ as it finishes, so a run that is cut off
    continues where it stopped when started again with the same settings.
    """
    need = math.ceil(per_class * MARGIN)
    stream = row_groups(fs, split_files(fs, split), random.Random(f"{SEED}-{split}"))
    partial_dir = out_dir / f"partial-{split}"
    rows, seen = [], set(exclude_sha)
    stats = {"row_groups": 0, "bytes": 0, "duplicate": 0, "in_train": 0,
             "excluded_model": 0, "too_small": 0, "unreadable": 0, "failed_groups": 0}
    batches_done = 0
    previous = resume(partial_dir, need, exclude_sha, min_side)
    if previous:
        rows, stats, batches_done = previous
        seen |= {row["image_sha256"] for row in rows}
        for _ in islice(stream, stats["row_groups"]):  # skip the row groups already read
            pass
    else:
        shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: sum(row["label"] == label for row in rows) for label in (0, 1)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while min(counts.values()) < need:
            batch = list(islice(stream, BATCH_GROUPS))
            if not batch:
                print("  Ran out of row groups before reaching the target.")
                break
            new_rows = []
            for group_rows, group_stats in pool.map(lambda pg: process_group(fs, *pg, split, min_side), batch):
                for key in ("excluded_model", "too_small", "unreadable", "failed_groups", "bytes"):
                    stats[key] += group_stats[key]
                for row in group_rows:
                    if row["image_sha256"] in seen:
                        # "in_train": already in an earlier split (train, or train/validation for test)
                        stats["in_train" if row["image_sha256"] in exclude_sha else "duplicate"] += 1
                        continue
                    if counts[row["label"]] >= need:
                        continue  # this class is already full
                    seen.add(row["image_sha256"])
                    new_rows.append(row)
                    counts[row["label"]] += 1
            stats["row_groups"] += len(batch)
            rows += new_rows
            # Save the batch first, then the progress note, so a cut-off never leaves a half batch.
            pd.DataFrame(new_rows, columns=COLUMNS).to_parquet(
                partial_dir / f"batch-{batches_done:05d}.parquet", index=False)
            batches_done += 1
            (partial_dir / "progress.json").write_text(json.dumps(
                {"need": need, "exclude_count": len(exclude_sha), "min_side": min_side,
                 "batches": batches_done, "stats": stats}))
            print(f"  {stats['row_groups']:>5} row groups ({stats['bytes'] / 2**30:.1f} GB): "
                  f"real {counts[0]:,} / AI {counts[1]:,} of {need:,} each")
    return pd.DataFrame(rows, columns=COLUMNS), stats


def balance(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Equal real and AI counts; within each, an equal share per generator or real source."""
    per_class = min(per_class, *[int((df["label"] == label).sum()) for label in (0, 1)])
    parts = []
    for label in (0, 1):
        group = df[df["label"] == label]
        quota = water_fill(group["model"].value_counts().to_dict(), per_class)
        for model, n in sorted(quota.items()):
            rows = group[group["model"] == model].sort_values("image_sha256")
            parts.append(rows.sample(n=n, random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def save(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    for old in out_dir.glob(f"{split}-*.parquet"):
        old.unlink()
    for i, start in enumerate(range(0, len(df), SHARD_ROWS)):
        df.iloc[start:start + SHARD_ROWS].to_parquet(out_dir / f"{split}-{i:05d}.parquet", index=False)


def summary(df: pd.DataFrame, stats: dict) -> dict:
    return {
        "images": len(df),
        "real": int((df["label"] == 0).sum()),
        "ai": int((df["label"] == 1).sum()),
        "real_by_source": df[df["label"] == 0]["model"].value_counts().to_dict(),
        "ai_by_generator": df[df["label"] == 1]["model"].value_counts().to_dict(),
        "ai_by_type": df[df["label"] == 1]["type"].value_counts().to_dict(),
        "read": {"gb_downloaded": round(stats["bytes"] / 2**30, 2),
                 **{k: v for k, v in stats.items() if k != "bytes"}},
    }


def split_shas(out_dir: Path, splits: tuple[str, ...]) -> set:
    """image_sha256 of every image already prepared for these splits."""
    files = [f for split in splits for f in sorted(out_dir.glob(f"{split}-*.parquet"))]
    if not files:
        return set()
    return set(pd.concat(pd.read_parquet(f, columns=["image_sha256"]) for f in files)["image_sha256"])


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "created": date.today().isoformat(), "dataset": "ComplexDataLab/OpenFake (core)",
        "dataset_revision": REVISION, "seed": SEED, "label_meaning": {"0": "real", "1": "AI"},
        "excluded_models": sorted(EXCLUDED_MODELS), "min_source_side": MIN_SOURCE_SIDE,
        "preprocess": {**SETTINGS, "train": "random values per image", "validation": "fixed values",
                       "test": "fixed values"},
    })
    per_class = {"train": args.train_per_class, "validation": args.validation_per_class,
                 "test": args.test_per_class}
    if "test" in args.splits and not any(args.out_dir.glob("validation-*.parquet")):
        raise SystemExit("Build train and validation first: test images that also appear in them are dropped.")

    built_train = False
    for split in [s for s in ALL_SPLITS if s in args.splits]:
        print(f"\n=== {split}: {per_class[split]:,} real + {per_class[split]:,} AI ===")
        min_side = args.test_min_side if split == "test" else MIN_SOURCE_SIDE
        done = manifest.get(split, {}).get("requested_per_class") == per_class[split] \
            and manifest.get(split, {}).get("min_side", MIN_SOURCE_SIDE) == min_side \
            and any(args.out_dir.glob(f"{split}-*.parquet"))
        if done and not args.redo and not (split == "validation" and built_train):
            print("  Already built with these settings; skipping (use --redo to rebuild).")
            continue
        exclude = {"train": set(), "validation": split_shas(args.out_dir, ("train",)),
                   "test": split_shas(args.out_dir, ("train", "validation"))}[split]
        df, stats = collect(fs, split, per_class[split], exclude, min_side, args.workers, args.out_dir)
        df = balance(df, per_class[split])
        save(df, args.out_dir, split)
        built_train = built_train or split == "train"
        manifest[split] = {"requested_per_class": per_class[split], "min_side": min_side, **summary(df, stats)}
        manifest_path.write_text(json.dumps(manifest, indent=2))  # after each split, in case of a cut-off
        shutil.rmtree(args.out_dir / f"partial-{split}", ignore_errors=True)
        print(f"  Kept {manifest[split]['real']:,} real + {manifest[split]['ai']:,} AI images "
              f"from {len(manifest[split]['ai_by_generator'])} generators. Skipped: "
              + ", ".join(f"{k} {v:,}" for k, v in stats.items() if k not in ("row_groups", "bytes")))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    nxt = "image/evaluate_test.py" if args.splits == ["test"] else "image/check_shortcuts.py"
    print(f"\nSaved to {args.out_dir}. Next: .venv/bin/python {nxt}")


if __name__ == "__main__":
    main()
