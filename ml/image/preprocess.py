"""The one way every image is prepared, real and AI alike, for training and for real use.

Why: in OpenFake, real and AI images differ in file format, JPEG quality, file size, image size,
shape and camera data (DATASETS.md, Nov 9 shortcut check). A model trained on raw files could
learn those instead of signs of AI. After this function every image is a square, metadata-free
JPEG of the same size range and quality range, so only the picture itself is left to learn from.

Training passes an rng: each image gets a random crop, size and JPEG quality from the ranges
below. Validation, test and the server pass no rng and get the fixed middle values, so the
same image always gives the same result.
"""

import io
import random

from PIL import Image, ImageOps

# Only images at least this large on their shorter side are used for training, in both groups.
# A crop (at least 80% of the side) is then at least 320 px, so it is never upscaled.
MIN_SOURCE_SIDE = 400

SETTINGS = {
    "crop_fraction": (0.8, 1.0),  # random square crop, as a share of the shorter side
    "size": (224, 320),           # output side in pixels; CLIP and EfficientNet read 224 px
    "jpeg_quality": (70, 95),
    "fixed": {"crop_fraction": 1.0, "size": 288, "jpeg_quality": 90},  # no rng: centered
}


def normalize_image(data: bytes, rng: random.Random | None = None) -> bytes:
    """Return the image as a square RGB JPEG with no metadata."""
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")  # applies rotation, drops alpha

    width, height = image.size
    short = min(width, height)
    if rng is None:
        fixed = SETTINGS["fixed"]
        side = round(short * fixed["crop_fraction"])
        left, top = (width - side) // 2, (height - side) // 2
        size, quality = fixed["size"], fixed["jpeg_quality"]
    else:
        side = round(short * rng.uniform(*SETTINGS["crop_fraction"]))
        left, top = rng.randint(0, width - side), rng.randint(0, height - side)
        size, quality = rng.randint(*SETTINGS["size"]), rng.randint(*SETTINGS["jpeg_quality"])

    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality)  # no exif/icc passed, so none is written
    return out.getvalue()
