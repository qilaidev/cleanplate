from __future__ import annotations

from pathlib import Path

from wm_platform.config import QUALITY_MODE_VALUES
from wm_platform.errors import AppError
from wm_platform.models import JobStatus, ProviderName

ALLOWED_PROVIDERS: frozenset[str] = frozenset(ProviderName.__args__)
ALLOWED_STATUSES: frozenset[str] = frozenset(JobStatus.__args__)
ALLOWED_MEDIA_TYPES: frozenset[str] = frozenset({"video", "image"})
MAX_IDEMPOTENCY_KEY_LENGTH = 128
MAX_CALLBACK_URL_LENGTH = 2048
MIN_PRIORITY = -1000
MAX_PRIORITY = 1000


def blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def normalize_quality_mode(raw: str | None, default: str) -> str:
    requested = blank_to_none(raw)
    if requested is None:
        fallback = (default or "balanced").strip().lower() or "balanced"
        return fallback if fallback in QUALITY_MODE_VALUES else "balanced"
    normalized = requested.lower()
    if normalized not in QUALITY_MODE_VALUES:
        raise AppError("VALIDATION_ERROR", "quality_mode is invalid", 400)
    return normalized


def normalize_idempotency_key(raw: str | None) -> str | None:
    key = blank_to_none(raw)
    if key is None:
        return None
    if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise AppError("VALIDATION_ERROR", "Idempotency-Key is too long", 400)
    return key


def normalize_priority(value: int) -> int:
    if value < MIN_PRIORITY or value > MAX_PRIORITY:
        raise AppError(
            "VALIDATION_ERROR",
            f"priority must be between {MIN_PRIORITY} and {MAX_PRIORITY}",
            400,
        )
    return value


def normalize_callback_url(raw: str | None) -> str | None:
    url = blank_to_none(raw)
    if url is None:
        return None
    if len(url) > MAX_CALLBACK_URL_LENGTH:
        raise AppError("VALIDATION_ERROR", "callback_url is too long", 400)
    return url


def require_known_provider(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in ALLOWED_PROVIDERS:
        raise AppError("VALIDATION_ERROR", "provider is invalid", 400)
    return value


def require_known_status(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in ALLOWED_STATUSES:
        raise AppError("VALIDATION_ERROR", "status is invalid", 400)
    return value


def require_known_media_type(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in ALLOWED_MEDIA_TYPES:
        raise AppError("VALIDATION_ERROR", "media_type is invalid", 400)
    return value


def assert_under_root(path: Path, root: Path, *, error_code: str, error_message: str, status_code: int) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AppError(error_code, error_message, status_code) from exc
    return resolved
