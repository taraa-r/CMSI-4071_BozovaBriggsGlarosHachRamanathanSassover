"""Check that prepare_data.py removed the file shortcuts, before any training.

The Nov 9 check (explore_openfake.py) found real and AI images differ in format, JPEG
quality, file size, image size and camera data. This reruns that table on the PREPARED
images, then asks for each file property: could it alone tell real from AI? It reports the
ROC AUC, where 0.5 means no better than a coin flip and 1.0 means a perfect shortcut.
Any property outside 0.45-0.55 is flagged.

Run from the ml folder:
    .venv/bin/python image/check_shortcuts.py                       (checks train and validation)
    .venv/bin/python image/check_shortcuts.py --inspect-model tiny-random-sana
Options:
    --data-dir PATH         where prepare_data.py wrote its files (default ml/data/image)
    --inspect-model NAME    save a contact sheet of 16 raw OpenFake validation images from one
                            generator, to look at by eye (e.g. to check an excluded generator)

Also saves contact sheets of 16 prepared real and 16 prepared AI images to the data folder.
"""

import argparse
import io
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem
from PIL import Image, ImageDraw

from openfake import describe_image, read_group, row_groups, split_files

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "image"
PROPERTIES = {"bytes": "File size", "width": "Width", "height": "Height", "jpeg_quality": "JPEG quality"}
FLAG_LOW, FLAG_HIGH = 0.45, 0.55
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--inspect-model", default=None)
    return parser.parse_args()


def auc(values: pd.Series, labels: pd.Series) -> float:
    """ROC AUC of one property for telling AI (1) from real (0), via average ranks."""
    ranks = values.rank()
    n_ai, n_real = int((labels == 1).sum()), int((labels == 0).sum())
    return (ranks[labels == 1].sum() - n_ai * (n_ai + 1) / 2) / (n_ai * n_real)


def contact_sheet(images: list[bytes], titles: list[str], path: Path, thumb: int = 160) -> None:
    sheet = Image.new("RGB", (thumb * 4, (thumb + 14) * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (data, title) in enumerate(zip(images[:16], titles[:16])):
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb, thumb))
            x, y = (i % 4) * thumb, (i // 4) * (thumb + 14)
            sheet.paste(image, (x, y))
            draw.text((x + 2, y + thumb), title[:26], fill="black")
    sheet.save(path)
    print(f"  Saved {path}")


def check_split(df: pd.DataFrame, split: str) -> list[str]:
    info = pd.DataFrame([describe_image(data) for data in df["image"]])
    info["label"] = df["label"].values
    real, ai = info[info.label == 0], info[info.label == 1]
    print(f"\n=== {split}: {len(real):,} real | {len(ai):,} AI prepared images ===")
    print(f"{'':<26}{'real':>10}{'AI':>10}")
    for name, rows in [("JPEG (share)", lambda d: f"{(d['format'] == 'JPEG').mean():.0%}"),
                       ("Median width (px)", lambda d: f"{d['width'].median():.0f}"),
                       ("Median file size (KB)", lambda d: f"{d['bytes'].median() / 1024:.1f}"),
                       ("Median JPEG quality", lambda d: f"{d['jpeg_quality'].median():.0f}"),
                       ("Has camera data (EXIF)", lambda d: f"{d['has_exif'].mean():.0%}"),
                       ("Square", lambda d: f"{(d['width'] == d['height']).mean():.0%}"),
                       ("Within SRS image limits", lambda d: f"{d['within_srs_limits'].mean():.0%}")]:
        print(f"{name:<26}{rows(real):>10}{rows(ai):>10}")

    print("\nCould this property alone tell real from AI? (AUC: 0.5 = coin flip, 1.0 = perfect shortcut)")
    flagged = []
    for col, name in PROPERTIES.items():
        score = auc(info[col].astype(float), info["label"])
        flag = not FLAG_LOW <= score <= FLAG_HIGH
        print(f"  {name:<14} {score:.3f}  {'<- SHORTCUT LEFT' if flag else 'ok'}")
        if flag:
            flagged.append(f"{split}: {name} AUC {score:.3f}")
    return flagged


def inspect_model(model: str, out_dir: Path) -> None:
    """Find 16 raw validation images from one generator and save them as a contact sheet."""
    fs = HfFileSystem()
    images, titles = [], []
    for path, group in row_groups(fs, split_files(fs, "validation"), random.Random(SEED)):
        labels = read_group(fs, path, group, ["model"])
        if not any(r["model"] == model for r in labels):
            continue
        for record in read_group(fs, path, group, ["image", "model", "prompt"]):
            if record["model"] == model:
                data = record["image"]["bytes"]
                info = describe_image(data)
                print(f"  {info.get('format')} {info.get('width')}x{info.get('height')} "
                      f"{len(data) / 1024:.0f} KB  prompt: {(record['prompt'] or '')[:60]}")
                images.append(data)
                titles.append((record["prompt"] or "")[:26])
        if len(images) >= 16:
            break
    if not images:
        print(f"  No {model} images found in validation.")
        return
    contact_sheet(images, titles, out_dir / f"inspect_{model}.png")


def main():
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.inspect_model:
        print(f"Looking for {args.inspect_model} images in OpenFake validation...")
        inspect_model(args.inspect_model, args.data_dir)
        return

    flagged = []
    for split in ("train", "validation"):
        files = sorted(args.data_dir.glob(f"{split}-*.parquet"))
        if not files:
            print(f"\nNo {split} files in {args.data_dir}; run prepare_data.py first.")
            continue
        df = pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)
        flagged += check_split(df, split)
        for label, word in [(0, "real"), (1, "ai")]:
            sample = df[df.label == label].sample(n=min(16, (df.label == label).sum()), random_state=SEED)
            contact_sheet(list(sample["image"]), list(sample["model"]), args.data_dir / f"sheet_{split}_{word}.png")

    print("\nResult: " + ("no file shortcuts found." if not flagged
                          else "shortcuts left, decide before training:\n  " + "\n  ".join(flagged)))


if __name__ == "__main__":
    main()
