
"""Run independent image detectors concurrently with one shared deadline."""

import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from threading import BoundedSemaphore

from .c2pa_check import check_c2pa

logger = logging.getLogger("realeye.detectors")

DETECTOR_TIMEOUT_SECONDS = 5
MAX_ACTIVE_CHECKS = 8

_executor = ThreadPoolExecutor(
    max_workers=MAX_ACTIVE_CHECKS,
    thread_name_prefix="realeye-detector",
)
_slots = BoundedSemaphore(MAX_ACTIVE_CHECKS)


def check_synthid(image_bytes: bytes, mime_type: str) -> str:
    """SynthID is not integrated yet."""
    return "unavailable"


def _submit_detector(detector, image_bytes, mime_type):
    if not _slots.acquire(blocking=False):
        return None

    def run():
        try:
            return detector(image_bytes, mime_type)
        finally:
            _slots.release()

    try:
        return _executor.submit(run)
    except RuntimeError:
        _slots.release()
        return None


def _get_result(future, detector_name, deadline):
    if future is None:
        return "unavailable"

    remaining = max(0, deadline - time.monotonic())

    try:
        return future.result(timeout=remaining)
    except FutureTimeoutError:
        logger.warning("%s detector timed out", detector_name)
        future.cancel()
        return "unavailable"
    except Exception:
        logger.exception("%s detector failed", detector_name)
        return "error"


def run_image_detectors(image_bytes: bytes, mime_type: str):
    """Run C2PA and SynthID with a shared five-second deadline."""

    deadline = time.monotonic() + DETECTOR_TIMEOUT_SECONDS

    c2pa_future = _submit_detector(
        check_c2pa, image_bytes, mime_type
    )
    synthid_future = _submit_detector(
        check_synthid, image_bytes, mime_type
    )

    c2pa_status = _get_result(
        c2pa_future, "C2PA", deadline
    )
    synthid_status = _get_result(
        synthid_future, "SynthID", deadline
    )

    return c2pa_status, synthid_status
