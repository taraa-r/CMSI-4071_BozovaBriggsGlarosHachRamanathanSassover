"""Label each prepared image's style with CLIP and cap AI illustrations (decided Nov 11).

Why: in OpenFake, AI images include many illustrations, anime and paintings, while real
images are almost all photos (CLIP sample, Nov 11: illustrations 10% of AI vs 2% of real).
A model trained on that could learn "illustration = AI". So AI images that CLIP is confident
are illustrations are capped at the same share as on the real side; the model then mostly
compares photos with photo-style AI images. Real images are trimmed to keep real = AI.

Run after prepare_data.py, from the ml folder (a GPU makes it quick; CPU works for small runs):
    .venv/bin/python image/label_styles.py [--data-dir PATH]

Writes styles.parquet next to the data: one row per image (row_id, split, label, model,
style scores, style, keep). Training and evaluation use only rows with keep == True, and
the style column lets errors be checked by style after training.
"""

import argparse
import io
import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-base-patch16"
CLIP_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"  # pinned in DATASETS.md
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "image"
SEED = 42
BATCH = 64

# CLIP compares each image with these descriptions. Only "illustration" is acted on: in the
# Nov 11 spot check, CLIP's illustration label was mostly right, while "render" also caught
# real studio photos.
STYLES = {
    "photo": "a photo",
    "illustration": "a drawing, painting, cartoon, anime or digital illustration",
    "render": "a 3D render or video game graphics",
    "graphic": "a poster, advertisement or graphic design with text",
}
MIN_ILLUSTRATION_SCORE = 0.5  # count as an illustration only when CLIP is fairly sure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def style_scores(images: list[bytes], model, processor, device) -> pd.DataFrame:
    with torch.no_grad():
        text = processor(text=list(STYLES.values()), return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = []
        for start in range(0, len(images), BATCH):
            batch = [Image.open(io.BytesIO(data)).convert("RGB") for data in images[start:start + BATCH]]
            pixels = processor(images=batch, return_tensors="pt").to(device)
            features = model.get_image_features(**pixels)
            features = features / features.norm(dim=-1, keepdim=True)
            scores.append((100 * features @ text_features.T).softmax(dim=-1).cpu())
            print(f"\r  {min(start + BATCH, len(images)):,} / {len(images):,} images", end="")
    print()
    return pd.DataFrame(torch.cat(scores).numpy(), columns=[f"p_{name}" for name in STYLES])


def cap_illustrations(df: pd.DataFrame, rng: random.Random) -> pd.Series:
    """keep flags: AI illustrations capped at the real illustration share, then real = AI."""
    real, ai = df[df["label"] == 0], df[df["label"] == 1]
    real_share = real["is_illustration"].mean()
    ai_ill = ai[ai["is_illustration"]].index.tolist()
    ai_other = len(ai) - len(ai_ill)
    # Keep k AI illustrations so that k / (ai_other + k) is at most real_share.
    allowed = int(real_share * ai_other / (1 - real_share)) if real_share < 1 else len(ai_ill)
    drop = set(rng.sample(sorted(ai_ill), max(0, len(ai_ill) - allowed)))
    n_ai = len(ai) - len(drop)
    drop |= set(rng.sample(sorted(real.index), max(0, len(real) - n_ai)))
    return ~df.index.isin(drop)


def style_table(df: pd.DataFrame) -> str:
    table = pd.crosstab(df["style"], df["label"].map({0: "real", 1: "AI"}), normalize="columns")
    return (table * 100).round(0).astype(int).astype(str).add("%").to_string()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)

    results, manifest_note = [], {}
    for split in ("train", "validation"):
        files = sorted(args.data_dir.glob(f"{split}-*.parquet"))
        if not files:
            print(f"No {split} files in {args.data_dir}; run prepare_data.py first.")
            continue
        df = pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)
        print(f"\n=== {split}: {len(df):,} images ===")
        scores = style_scores(list(df["image"]), model, processor, device)
        df = pd.concat([df.drop(columns="image"), scores], axis=1)
        df["style"] = scores.idxmax(axis=1).str.removeprefix("p_")
        df["is_illustration"] = (df["style"] == "illustration") & (df["p_illustration"] >= MIN_ILLUSTRATION_SCORE)
        df["keep"] = cap_illustrations(df, random.Random(f"{SEED}-{split}"))
        df["split"] = split

        kept = df[df["keep"]]
        print("Style share before the cap:\n" + style_table(df))
        print("After the cap:\n" + style_table(kept))
        print(f"Kept {(kept.label == 0).sum():,} real + {(kept.label == 1).sum():,} AI "
              f"(dropped {(~df['keep']).sum():,})")
        manifest_note[split] = {
            "illustration_share_real": round(float(df[df.label == 0]["is_illustration"].mean()), 4),
            "illustration_share_ai_before": round(float(df[df.label == 1]["is_illustration"].mean()), 4),
            "illustration_share_ai_after": round(float(kept[kept.label == 1]["is_illustration"].mean()), 4),
            "kept_real": int((kept.label == 0).sum()), "kept_ai": int((kept.label == 1).sum()),
        }
        results.append(df)

    if not results:
        return
    out = pd.concat(results, ignore_index=True)
    out.to_parquet(args.data_dir / "styles.parquet", index=False)
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["style_cap"] = {"clip_model": CLIP_MODEL, "clip_revision": CLIP_REVISION, "styles": STYLES,
                             "min_illustration_score": MIN_ILLUSTRATION_SCORE, **manifest_note}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {args.data_dir / 'styles.parquet'}")


if __name__ == "__main__":
    main()
