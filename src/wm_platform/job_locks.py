from __future__ import annotations

import fcntl
from pathlib import Path

from wm_platform.config import Settings


def job_lock_path(settings: Settings, job_id: str) -> Path:
    lock_dir = settings.storage_root / ".locks" / "jobs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{job_id}.lock"


class JobFileLock:
    def __init__(self, settings: Settings, job_id: str) -> None:
        self.path = job_lock_path(settings, job_id)
        self._handle = None

    def acquire(self) -> bool:
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def is_job_lock_held(settings: Settings, job_id: str) -> bool:
    lock = JobFileLock(settings, job_id)
    acquired = lock.acquire()
    if acquired:
        lock.release()
        return False
    return True
