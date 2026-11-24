
"""Validate submitted images before running detection models."""

import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_SIDE = 32
MAX_SIDE = 4096
MAX_PIXELS = 16_000_000

ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def validate_image(image_base64: str, mime_type: str):
    """
    Validate an image before model inference.

    Returns:
        (image_bytes, None) if valid
        (None, reason) if unsupported
    """

    # Check approximate decoded size before decoding.
    encoded = image_base64.strip()

    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        return None, "Image exceeds the 10 MiB limit."

    try:
        image_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
    except (binascii.Error, ValueError):
        return None, "Invalid base64 image data."

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None, "Image exceeds the 10 MiB limit."

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            actual_format = image.format

            if actual_format not in ALLOWED_FORMATS:
                return None, "Unsupported image format."

            if ALLOWED_FORMATS[actual_format] != mime_type:
                return None, "Image MIME type does not match its content."

            width, height = image.size

            if not (
                MIN_SIDE <= width <= MAX_SIDE
                and MIN_SIDE <= height <= MAX_SIDE
            ):
                return None, (
                    "Image dimensions must be between "
                    "32 and 4096 pixels per side."
                )

            if width * height > MAX_PIXELS:
                return None, (
                    "Image exceeds the 16 million pixel limit."
                )

            if actual_format == "WEBP":
                if getattr(image, "is_animated", False) or (
                    getattr(image, "n_frames", 1) > 1
                ):
                    return None, "Animated WebP is not supported."

            # Verify that the image can actually be decoded.
            image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        return None, "Invalid or corrupted image data."

    return image_bytes, None
