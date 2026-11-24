
"""Inspect C2PA credentials without saving uploaded images."""

import io
import logging
from pathlib import Path

from c2pa import Context, Reader

logger = logging.getLogger("realeye.c2pa")

TRUST_LIST_PATH = (
    Path(__file__).resolve().parent.parent
    / "c2pa-trust-list.pem"
)

C2PA_SETTINGS = {
    "verify": {
        "verify_after_reading": True,
        "verify_trust": True,
        "remote_manifest_fetch": False,
        "ocsp_fetch": False,
    }
}


def check_c2pa(image_bytes: bytes, mime_type: str) -> str:
    """Inspect C2PA credentials with certificate trust verification."""

    try:
        settings = {
            "verify": C2PA_SETTINGS["verify"].copy()
        }

        if TRUST_LIST_PATH.is_file():
            trust_list = TRUST_LIST_PATH.read_text(
                encoding="utf-8"
            )

            settings["trust"] = {
                "trust_anchors": trust_list
            }
        else:
            logger.warning("C2PA trust list is missing")
            return "unavailable"

        with Context.from_dict(settings) as context:
            with Reader(
                format_or_path=mime_type,
                stream=io.BytesIO(image_bytes),
                context=context,
            ) as reader:

                manifest = reader.get_active_manifest()

                if not manifest:
                    return "missing_credential"

                results = reader.get_validation_results() or {}
                active = results.get("activeManifest") or {}

                failures = {
                    item.get("code")
                    for item in active.get("failure", [])
                    if isinstance(item, dict)
                }

                if "signingCredential.untrusted" in failures:
                    return "unavailable"

                if failures:
                    return "invalid_credential"

                # Even a trusted credential is not automatically
                # evidence that an image was AI-generated.
                # AI assertions must be inspected separately.
                return "unavailable"

    except Exception as exc:
        if type(exc).__name__ == "_C2paManifestNotFound":
            return "missing_credential"

        logger.warning(
            "C2PA inspection failed: %s",
            type(exc).__name__,
        )
        return "error"

