from __future__ import annotations

import threading
import time
from collections import deque

from wm_platform.config import Settings
from wm_platform.errors import AppError


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, key: str, limit: int, window_seconds: float) -> bool:
        if limit <= 0 or window_seconds <= 0:
            return True

        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


_submit_rate_limiter = SlidingWindowRateLimiter()


def enforce_submit_rate_limit(api_key: str, settings: Settings) -> None:
    if not _submit_rate_limiter.allow(
        key=api_key,
        limit=settings.submit_rate_limit_count,
        window_seconds=settings.submit_rate_limit_window_seconds,
    ):
        raise AppError(
            "RATE_LIMITED",
            "submit rate limit exceeded",
            429,
            headers={"Retry-After": str(max(1, int(settings.submit_rate_limit_window_seconds)))},
        )


def reset_submit_rate_limiter() -> None:
    _submit_rate_limiter.reset()
