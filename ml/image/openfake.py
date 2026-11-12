"""Shared OpenFake settings and readers, used by explore_openfake.py, prepare_data.py and
check_shortcuts.py. OpenFake is read straight from Hugging Face; it is never downloaded whole.
"""

import io
import random
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

REPO = "datasets/ComplexDataLab/OpenFake"
REVISION = "00b2f0ae3f5f7684c4b00f794f4c500076a9355d"  # pinned in DATASETS.md
LABELS = {"real": 0, "fake": 1}

# SRS 5.5.3 image limits
FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_SIDE, MAX_SIDE = 32, 4096
MAX_PIXELS = 16_000_000
MAX_BYTES = 10 * 1024 * 1024

# Standard JPEG luminance table at quality 50, used to estimate an image's JPEG quality.
STANDARD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def split_files(fs: HfFileSystem, split: str) -> list[str]:
    return sorted(fs.glob(f"{REPO}@{REVISION}/core/{split}-*.parquet"))


def read_labels(fs: HfFileSystem, path: str):
    with fs.open(path) as f:
        return pq.ParquetFile(f).read(columns=["label", "model", "type", "image_sha256"]).to_pandas()


def row_groups(fs: HfFileSystem, files: list[str], rng: random.Random):
    """Yield (file, row group) pairs in random order, one group per file per pass.

    A row group is a chunk of about 70 images, mixed real and AI. Spreading picks across files
    keeps the sample varied. A file's footer is only opened when it is first needed.
    """
    order = list(files)
    rng.shuffle(order)
    remaining = {}
    while order:
        for path in list(order):
            if path not in remaining:
                n = num_row_groups(fs, path)
                remaining[path] = rng.sample(range(n), n)
            if remaining[path]:
                yield path, remaining[path].pop()
            else:
                order.remove(path)


def num_row_groups(fs: HfFileSystem, path: str, attempts: int = 3) -> int:
    """Row groups in a file; 0 (skip the file) if it still can't be opened after a few tries."""
    for attempt in range(attempts):
        try:
            with fs.open(path) as f:
                return pq.ParquetFile(f).num_row_groups
        except Exception as error:  # e.g. the wifi dropped
            print(f"  Couldn't open {path.rsplit('/', 1)[-1]} ({type(error).__name__}), "
                  + ("retrying in 30 s" if attempt + 1 < attempts else "skipping it"))
            if attempt + 1 < attempts:
                time.sleep(30)
    return 0


def read_group(fs: HfFileSystem, path: str, group: int, columns: list[str]) -> list[dict]:
    with fs.open(path) as f:
        return pq.ParquetFile(f).read_row_group(group, columns=columns).to_pylist()


def normalize_type(value):
    """One spelling per generator type: Base -> base, Fine-tune -> finetune, LoRA -> lora."""
    if not isinstance(value, str):
        return value
    return value.lower().replace("-", "").replace(" ", "")


def jpeg_quality(image: Image.Image) -> float | None:
    """Rough JPEG quality (1-100) from the image's luminance quantization table."""
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return None
    scale = sum(tables[0]) * 100 / sum(STANDARD_LUMA)  # order-independent
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return round(max(1.0, min(100.0, quality)))


def describe_image(data: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            info = {
                "format": image.format,
                "mode": image.mode,
                "width": width,
                "height": height,
                "jpeg_quality": jpeg_quality(image) if image.format == "JPEG" else None,
                "has_exif": len(image.getexif()) > 0,
                "animated": getattr(image, "is_animated", False),
            }
    except Exception as error:
        return {"format": f"unreadable: {type(error).__name__}"}
    info["bytes"] = len(data)
    info["megapixels"] = width * height / 1e6
    info["within_srs_limits"] = (
        info["format"] in FORMATS and not info["animated"]
        and MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE
        and width * height <= MAX_PIXELS and len(data) <= MAX_BYTES
    )
    return info
