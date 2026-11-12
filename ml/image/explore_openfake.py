"""Look at OpenFake before training: real vs AI counts, generators, and shortcut checks.

OpenFake is ~3.4 TB, so this never downloads it whole. It reads directly from Hugging Face:
  1. Labels: only the small label/model/type columns (a few MB per file).
  2. Images: a few chunks of ~78 images (~90 MB each) from randomly chosen files.

Shortcut check: if real and AI images differ in file format, size, JPEG quality, or camera data
(EXIF), a model can learn that difference instead of real signs of AI. Anything that differs
must be evened out in data prep (see DATASETS.md).

Run from the ml folder:
    .venv\\Scripts\\python image\\explore_openfake.py                    (Windows)
    .venv/bin/python image/explore_openfake.py                         (Mac)
Options:
    --split validation|test|train   default validation (72 GB on Hugging Face; train is 3.3 TB)
    --label-files N    read labels from N random files of the split (default: all files)
    --image-files N    take images from N random files (default 8)
    --chunks-per-file N   chunks of ~78 images to download per file (default 1, ~90 MB each)

Saves one row per sampled image to ml/data/image/explore_<split>.csv.
"""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

from openfake import LABELS, describe_image, read_labels, split_files

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "image"

Image.MAX_IMAGE_PIXELS = None  # we measure oversized images instead of refusing them


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--label-files", type=int, default=None)
    parser.add_argument("--image-files", type=int, default=8)
    parser.add_argument("--chunks-per-file", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_images(fs: HfFileSystem, path: str, chunks: int, rng: random.Random) -> list[dict]:
    rows = []
    with fs.open(path) as f:
        parquet = pq.ParquetFile(f)
        for group in rng.sample(range(parquet.num_row_groups), min(chunks, parquet.num_row_groups)):
            table = parquet.read_row_group(group, columns=["image", "label", "model", "type"])
            for record in table.to_pylist():
                rows.append({"label": LABELS[record["label"]], "model": record["model"],
                             "type": record["type"], **describe_image(record["image"]["bytes"])})
    return rows


def pct(series: pd.Series) -> str:
    return f"{series.mean():.0%}"


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    fs = HfFileSystem()
    files = split_files(fs, args.split)
    print(f"OpenFake core/{args.split}: {len(files)} files")

    # 1. Labels and generators
    label_files = files if args.label_files is None else rng.sample(files, min(args.label_files, len(files)))
    print(f"\nReading labels from {len(label_files)} file(s)...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        meta = pd.concat(pool.map(lambda p: read_labels(fs, p), label_files), ignore_index=True)
    meta["label"] = meta["label"].map(LABELS)

    print(f"\n=== 1. Labels ({'all' if args.label_files is None else 'sampled'} files) ===")
    print(f"Images:            {len(meta):>9,}")
    print(f"  real (0):        {(meta.label == 0).sum():>9,}  ({(meta.label == 0).mean():.0%})")
    print(f"  AI (1):          {(meta.label == 1).sum():>9,}  ({(meta.label == 1).mean():.0%})")
    print(f"Duplicate images (same SHA-256): {meta['image_sha256'].duplicated().sum():,}")
    print("\nReal images by source:")
    print(meta[meta.label == 0]["model"].value_counts().to_string())
    fake = meta[meta.label == 1]
    print(f"\nAI images: {fake['model'].nunique()} generators. By type:")
    print(fake["type"].value_counts().to_string())
    print("\nMost common generators:")
    print(fake["model"].value_counts().head(25).to_string())

    # 2. Shortcut check on a sample of images
    image_files = rng.sample(files, min(args.image_files, len(files)))
    print(f"\nDownloading {args.chunks_per_file} chunk(s) of images from each of {len(image_files)} file(s) "
          f"(~{90 * args.chunks_per_file * len(image_files)} MB)...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        sampled = pool.map(lambda p: sample_images(fs, p, args.chunks_per_file, random.Random(f"{args.seed}{p}")),
                           image_files)
        images = pd.DataFrame([row for rows in sampled for row in rows])
    for col in ("has_exif", "within_srs_limits"):  # unreadable images count as False
        images[col] = images[col].fillna(False).astype(bool)

    print(f"\n=== 2. Shortcut check: {len(images):,} sampled images "
          f"(real {(images.label == 0).sum():,} | AI {(images.label == 1).sum():,}) ===")
    print("\nFile format (share within each group):")
    formats = pd.crosstab(images["format"], images["label"].map({0: "real", 1: "AI"}), normalize="columns")
    print((formats * 100).round(0).astype(int).astype(str).add("%").to_string())

    print(f"\n{'':<28}{'real':>12}{'AI':>12}")
    for name, col in [("Median width (px)", "width"), ("Median height (px)", "height"),
                      ("Median megapixels", "megapixels"), ("Median file size (KB)", "bytes")]:
        r, a = images[images.label == 0][col].median(), images[images.label == 1][col].median()
        if col == "bytes":
            r, a = r / 1024, a / 1024
        print(f"{name:<28}{r:>12,.1f}{a:>12,.1f}")
    jpeg = images[images["format"] == "JPEG"]
    for label, word in [(0, "real"), (1, "AI")]:
        q = jpeg[jpeg.label == label]["jpeg_quality"]
        print(f"JPEG quality, {word:<5} (n={len(q):>4}): "
              + (f"median {q.median():.0f}, middle half {q.quantile(0.25):.0f}-{q.quantile(0.75):.0f}" if len(q) else "-"))
    print(f"{'Has camera data (EXIF)':<28}{pct(images[images.label == 0]['has_exif']):>12}"
          f"{pct(images[images.label == 1]['has_exif']):>12}")
    print(f"{'Within SRS image limits':<28}{pct(images[images.label == 0]['within_srs_limits']):>12}"
          f"{pct(images[images.label == 1]['within_srs_limits']):>12}")
    print("\nAny big real-vs-AI gap above is a shortcut to remove in data prep.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"explore_{args.split}.csv"
    images.to_csv(out, index=False)
    print(f"\nSaved per-image details to {out}")


if __name__ == "__main__":
    main()
