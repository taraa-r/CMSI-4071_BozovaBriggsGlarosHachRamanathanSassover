"""Label each prepared image's style with CLIP, then match styles between real and AI (Nov 12).

Why: in OpenFake, AI images include many more illustrations, graphics and 3D renders than
real images do (full validation set: 44% of AI vs 24% of real are not photos). The first
model, trained with only illustrations capped (Nov 11), learned "doesn't look like a normal
photo = AI": it flagged real posters, CGI and glossy studio photos, and missed photorealistic
AI (ERROR_ANALYSIS.md). So now every style is matched: in each style group, real and AI keep
the same number of images. The model can no longer use style as a shortcut and has to find
signs of AI within each style.

Style groups (CLIP's best-matching description; photos split by how sure CLIP is):
    photo (clear)   style "photo" and p_photo >= 0.8
    photo (unclear) style "photo" and p_photo < 0.8: polished, staged or edited-looking photos
    illustration, render, graphic
The groups don't need to be exactly right: the classifier only sees CLIP features, so
balancing what CLIP perceives is what removes the shortcut.

Run after prepare_data.py, from the ml folder:
    python image/label_styles.py [--data-dir PATH] [--rescore]
The first run scores every image with CLIP (a GPU makes it quick). Later runs reuse the scores
in styles.parquet and only redo the matching (seconds, CPU is fine); --rescore forces CLIP again.
After this, rerun extract_features.py: it builds features only for the kept images.

Writes styles.parquet next to the data: one row per image (row_id, split, label, model,
style scores, style, style_group, keep). Training and evaluation use only rows with
keep == True; the style columns let errors be checked by style after training.
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

# CLIP compares each image with these descriptions; the best match is the image's style.
STYLES = {
    "photo": "a photo",
    "illustration": "a drawing, painting, cartoon, anime or digital illustration",
    "render": "a 3D render or video game graphics",
    "graphic": "a poster, advertisement or graphic design with text",
}
MIN_ILLUSTRATION_SCORE = 0.5  # is_illustration column (the Nov 11 cap); kept for reference
CLEAR_PHOTO_SCORE = 0.8  # photos below this go in the "photo (unclear)" group


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--rescore", action="store_true", help="run CLIP again even if styles.parquet has scores")
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


def style_group(df: pd.DataFrame) -> pd.Series:
    clear = (df["style"] == "photo") & (df["p_photo"] >= CLEAR_PHOTO_SCORE)
    group = df["style"].where(df["style"] != "photo", "photo (unclear)")
    return group.where(~clear, "photo (clear)")


def match_styles(df: pd.DataFrame, rng: random.Random) -> pd.Series:
    """keep flags: in every style group, keep the same number of real and AI images.

    The larger side is trimmed at random, so the kept set stays balanced (real = AI) and no
    style is more common on one side. Images are sorted by row_id first so reruns match."""
    keep = set()
    for _, rows in df.groupby("style_group"):
        real = sorted(rows.index[rows["label"] == 0], key=lambda i: df.at[i, "row_id"])
        ai = sorted(rows.index[rows["label"] == 1], key=lambda i: df.at[i, "row_id"])
        n = min(len(real), len(ai))
        keep |= set(rng.sample(real, n)) | set(rng.sample(ai, n))
    return df.index.isin(keep)


def style_table(df: pd.DataFrame) -> str:
    counts = pd.crosstab(df["style_group"], df["label"].map({0: "real", 1: "AI"}))
    shares = (counts / counts.sum() * 100).round(0).astype(int).astype(str).add("%")
    return counts.astype(str).add(" (").add(shares).add(")").to_string()


def scored_split(split: str, args, saved: pd.DataFrame | None, clip: list) -> pd.DataFrame | None:
    """One split's images with style scores: reused from styles.parquet when possible."""
    files = sorted(args.data_dir.glob(f"{split}-*.parquet"))
    if not files:
        print(f"No {split} files in {args.data_dir}; run prepare_data.py first.")
        return None
    df = pd.concat(pd.read_parquet(f, columns=["row_id", "label", "model", "type", "image_sha256"])
                   for f in files).reset_index(drop=True)
    print(f"\n=== {split}: {len(df):,} images ===")
    score_cols = [f"p_{name}" for name in STYLES]
    if saved is not None:
        old = saved[saved["split"] == split].drop_duplicates("row_id").set_index("row_id")
        if set(df["row_id"]) <= set(old.index):
            print("  Reusing the CLIP style scores in styles.parquet (--rescore to run CLIP again).")
            return df.join(old[score_cols], on="row_id")
        print("  styles.parquet doesn't cover these images (the data was rebuilt); running CLIP.")
    if not clip:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading CLIP on {device}...")
        clip += [CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(device).eval(),
                 CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION), device]
    images = pd.concat(pd.read_parquet(f, columns=["image"]) for f in files)["image"].tolist()
    return pd.concat([df, style_scores(images, *clip)], axis=1)


def main():
    args = parse_args()
    styles_path = args.data_dir / "styles.parquet"
    saved = None
    if styles_path.exists() and not args.rescore:
        saved = pd.read_parquet(styles_path)
        if not {f"p_{name}" for name in STYLES} <= set(saved.columns):
            saved = None

    results, manifest_note, clip = [], {}, []
    for split in ("train", "validation"):
        df = scored_split(split, args, saved, clip)
        if df is None:
            continue
        df["style"] = df[[f"p_{name}" for name in STYLES]].idxmax(axis=1).str.removeprefix("p_")
        df["is_illustration"] = (df["style"] == "illustration") & (df["p_illustration"] >= MIN_ILLUSTRATION_SCORE)
        df["style_group"] = style_group(df)
        df["keep"] = match_styles(df, random.Random(f"{SEED}-{split}"))
        df["split"] = split

        kept = df[df["keep"]]
        print("Style groups before matching (count and share within real / AI):\n" + style_table(df))
        print("After matching:\n" + style_table(kept))
        print(f"Kept {(kept.label == 0).sum():,} real + {(kept.label == 1).sum():,} AI "
              f"(dropped {(~df['keep']).sum():,} of {len(df):,}, {(~df['keep']).mean():.0%})")
        manifest_note[split] = {
            "groups_before": {f"{g} {'real' if l == 0 else 'ai'}": int(n)
                              for (g, l), n in df.groupby(["style_group", "label"]).size().items()},
            "kept_real": int((kept.label == 0).sum()), "kept_ai": int((kept.label == 1).sum()),
        }
        results.append(df)

    if not results:
        return
    pd.concat(results, ignore_index=True).to_parquet(styles_path, index=False)
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.pop("style_cap", None)  # the Nov 11 illustration-only cap, replaced by matching
    manifest["style_matching"] = {"clip_model": CLIP_MODEL, "clip_revision": CLIP_REVISION, "styles": STYLES,
                                  "clear_photo_score": CLEAR_PHOTO_SCORE, **manifest_note}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {styles_path}. Next: extract_features.py (features for the kept images).")


if __name__ == "__main__":
    main()
