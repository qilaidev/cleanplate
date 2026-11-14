from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import UploadFile

from wm_platform.config import Settings
from wm_platform.errors import AppError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}


def ensure_storage_dirs(settings: Settings) -> None:
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    settings.outbox_dir.mkdir(parents=True, exist_ok=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    settings.comfyui_output_dir.mkdir(parents=True, exist_ok=True)


def _validate_extension(path: Path) -> None:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise AppError("MEDIA_TYPE_NOT_SUPPORTED", "only mp4/mov/mkv are supported", 400)


def _copy_stream_with_hash(source, destination: Path, max_upload_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    # Atomic write: write to temp file first, then rename
    temp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        with temp_path.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise AppError("FILE_TOO_LARGE", "uploaded file exceeds configured size limit", 400)
                digest.update(chunk)
                output.write(chunk)
        temp_path.rename(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return digest.hexdigest(), total_bytes


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_upload_file(upload: UploadFile, settings: Settings) -> tuple[Path, str]:
    suffix = Path(upload.filename or "upload.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise AppError("MEDIA_TYPE_NOT_SUPPORTED", "only mp4/mov/mkv are supported", 400)

    target = settings.inbox_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        digest, _ = _copy_stream_with_hash(upload.file, target, settings.max_upload_bytes)
        return target, digest
    except Exception:
        # Clean up target file on failure (atomic write already handles temp file cleanup)
        if target.exists():
            target.unlink()
        raise
    finally:
        upload.file.close()


def validate_local_input_path(raw_path: str, settings: Settings) -> tuple[Path, str]:
    if not raw_path or not raw_path.strip():
        raise AppError("VALIDATION_ERROR", "input_path is required", 400)
    resolved = Path(raw_path).expanduser().resolve()
    allowed_root = settings.storage_root.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise AppError("FILE_NOT_FOUND", "input_path must stay under storage/", 400) from exc

    if not resolved.exists() or not resolved.is_file():
        raise AppError("FILE_NOT_FOUND", "input_path does not exist", 404)

    _validate_extension(resolved)
    return resolved, _hash_file(resolved)


def build_output_path(job_id: str, source_path: str, settings: Settings) -> Path:
    suffix = Path(source_path).suffix.lower() or ".mp4"
    return settings.outbox_dir / f"{job_id}{suffix}"


def is_under_directory(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
