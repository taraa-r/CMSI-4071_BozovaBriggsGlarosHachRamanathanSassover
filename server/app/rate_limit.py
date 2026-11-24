
"""Per-IP rate limiting for the RealEye API."""

import math
import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

MAX_REQUESTS = 60
WINDOW_SECONDS = 60

# In-memory request history for each client IP.
_requests = defaultdict(deque)
_lock = threading.Lock()


def check_rate_limit(request: Request):
    """
    Allow at most 60 requests per rolling 60-second window
    for each client IP address.

    Returns None if allowed, otherwise an HTTP 429 response.
    """
    client_ip = (
        request.client.host
        if request.client is not None
        else "unknown"
    )

    now = time.monotonic()

    with _lock:
        # Remove expired IP entries to limit memory growth.
        for ip, timestamps in list(_requests.items()):
            while timestamps and now - timestamps[0] >= WINDOW_SECONDS:
                timestamps.popleft()
            if not timestamps:
                del _requests[ip]

        history = _requests[client_ip]

        if len(history) >= MAX_REQUESTS:
            retry_after = max(
                1,
                math.ceil(
                    WINDOW_SECONDS - (now - history[0])
                ),
            )

            return JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                },
                content={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": (
                            "Too many requests. Please try again later."
                        ),
                    }
                },
            )

        history.append(now)

    return None
