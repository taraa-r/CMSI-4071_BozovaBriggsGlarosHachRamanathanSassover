"""Turn each prepared image into CLIP features once, so the classifier can train in minutes.

CLIP's image encoder stays frozen (DATASETS.md): every image becomes 512 numbers describing it,
and only a small classifier on top is trained (train_clip.py). Running the encoder is the slow
part, so it happens once here and the numbers are saved.

Blur and compression augmentation (train only): besides the clean image, each train image gets
extra versions with random blur and/or an extra round of JPEG compression, applied the same way
to real and AI images. Training picks one version at random each round, so the classifier can't
rely on how sharp or how compressed a picture is (the file-size gap in DATASETS.md) and holds up
better on social-media-compressed images. Validation uses only the clean image.

Run after prepare_data.py and label_styles.py, from the ml folder (a GPU makes it quick):
    .venv/bin/python image/extract_features.py [--data-dir PATH] [--views 3]

Writes to <data-dir>/features/:
    train.npy         float16, shape (images, views, 512); view 0 is the clean image
    validation.npy    float16, shape (images, 1, 512)
    train.parquet / validation.parquet   one row per image, in the same order: row_id, label,
                      model, style, kb (prepared file size, for the file-size check)
Only images with keep == True in styles.parquet are included (the illustration cap).
"""

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor

from label_styles import CLIP_MODEL, CLIP_REVISION

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "image"
SEED = 42

# Each applied with probability 0.5, independently, to both real and AI images
# (the blur + JPEG recipe from Wang et al. 2020, "CNN-generated images are surprisingly easy to spot").
AUGMENT = {"blur_probability": 0.5, "blur_sigma": (0.1, 3.0),
           "jpeg_probability": 0.5, "jpeg_quality": (30, 95)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--views", type=int, default=3, help="versions per train image: 1 clean + augmented")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < AUGMENT["blur_probability"]:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*AUGMENT["blur_sigma"])))
    if rng.random() < AUGMENT["jpeg_probability"]:
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=rng.randint(*AUGMENT["jpeg_quality"]))
        image = Image.open(io.BytesIO(out.getvalue())).convert("RGB")
    return image


def encode(images: list[Image.Image], model, processor, device) -> np.ndarray:
    pixels = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
        features = model.get_image_features(**pixels)
    return features.float().cpu().numpy().astype(np.float16)


def extract(df: pd.DataFrame, views: int, split: str, model, processor, device, batch_size: int) -> np.ndarray:
    out = np.zeros((len(df), views, 512), dtype=np.float16)
    for view in range(views):
        print(f"  {split}, version {view + 1} of {views} ({'clean' if view == 0 else 'augmented'})")
        for start in range(0, len(df), batch_size):
            batch = []
            for sha, data in zip(df["image_sha256"].iloc[start:start + batch_size],
                                 df["image"].iloc[start:start + batch_size]):
                image = Image.open(io.BytesIO(data)).convert("RGB")
                if view > 0:  # seeded per image and version, so reruns give the same features
                    image = augment(image, random.Random(f"{SEED}-{sha}-{view}"))
                batch.append(image)
            out[start:start + len(batch), view] = encode(batch, model, processor, device)
            print(f"\r    {min(start + batch_size, len(df)):,} / {len(df):,}", end="")
        print()
    return out


def main():
    args = parse_args()
    styles_path = args.data_dir / "styles.parquet"
    if not styles_path.exists():
        raise SystemExit("styles.parquet not found: run label_styles.py first.")
    styles = pd.read_parquet(styles_path, columns=["split", "row_id", "style", "keep"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    out_dir = args.data_dir / "features"
    out_dir.mkdir(exist_ok=True)

    for split, views in (("train", args.views), ("validation", 1)):
        files = sorted(args.data_dir.glob(f"{split}-*.parquet"))
        df = pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)
        df = df.merge(styles[styles["split"] == split].drop(columns="split"), on="row_id")
        df = df[df["keep"]].reset_index(drop=True)
        print(f"\n=== {split}: {len(df):,} images (real {(df.label == 0).sum():,} | AI {(df.label == 1).sum():,}) ===")
        features = extract(df, views, split, model, processor, device, args.batch_size)
        np.save(out_dir / f"{split}.npy", features)
        df["kb"] = df["image"].map(len) / 1024
        df[["row_id", "label", "model", "style", "kb"]].to_parquet(out_dir / f"{split}.parquet", index=False)

    (out_dir / "settings.json").write_text(json.dumps({
        "clip_model": CLIP_MODEL, "clip_revision": CLIP_REVISION, "feature": "get_image_features (512, projected)",
        "train_views": args.views, "augment": AUGMENT, "seed": SEED}, indent=2))
    print(f"\nSaved to {out_dir}. Next: .venv/bin/python image/train_clip.py")


if __name__ == "__main__":
    main()
