from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wm_platform.config import Settings
from wm_platform.repository import JobRepository


def run_file_cleanup(settings: Settings, repository: JobRepository, execute: bool = False) -> dict[str, int | str]:
    retention_days = max(0, settings.file_retention_days)
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    cutoff_timestamp = cutoff.timestamp()

    report: dict[str, int | str] = {
        "mode": "execute" if execute else "dry_run",
        "retention_days": retention_days,
        "cutoff": cutoff.isoformat(),
        "candidate_files": 0,
        "candidate_jobs": 0,
        "candidate_db_files": 0,
        "deleted_files": 0,
        "failed_files": 0,
        "cleared_db_references": 0,
    }

    jobs = repository.list_jobs_for_cleanup(cutoff)
    protected_paths = repository.list_protected_file_paths(cutoff)
    report["candidate_jobs"] = len(jobs)

    for job in jobs:
        clear_input = _is_expired_file(job.input_path, cutoff_timestamp, protected_paths)
        clear_output = _is_expired_file(job.output_path, cutoff_timestamp, protected_paths)
        report["candidate_files"] += int(clear_input) + int(clear_output)
        if not execute:
            continue

        input_deleted = _delete_if_needed(job.input_path, clear_input)
        output_deleted = _delete_if_needed(job.output_path, clear_output)
        report["deleted_files"] += int(input_deleted) + int(output_deleted)
        report["failed_files"] += int(clear_input and not input_deleted and bool(job.input_path))
        report["failed_files"] += int(clear_output and not output_deleted and bool(job.output_path))
        repository.clear_job_artifacts(job.job_id, clear_input=input_deleted, clear_output=output_deleted)

    for directory in (settings.inbox_dir, settings.outbox_dir):
        for path in directory.iterdir():
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if path.stat().st_mtime >= cutoff_timestamp:
                continue
            if str(path.resolve()) in protected_paths:
                continue
            report["candidate_files"] += 1
            if not execute:
                continue
            try:
                path.unlink()
                report["deleted_files"] += 1
                report["cleared_db_references"] += repository.clear_file_references(str(path.resolve()), cutoff)
            except OSError:
                report["failed_files"] += 1

    # Work directories of jobs whose worker died mid-run (normally removed at job end).
    if settings.work_dir.exists():
        for path in settings.work_dir.iterdir():
            if not path.is_dir() or path.stat().st_mtime >= cutoff_timestamp:
                continue
            report["candidate_files"] += 1
            if execute:
                shutil.rmtree(path, ignore_errors=True)
                report["deleted_files"] += 1

    db_summary = repository.cleanup_expired_files(cutoff)
    report["candidate_db_files"] = db_summary["candidate_files"]
    if execute:
        report["cleared_db_references"] += repository.clear_missing_file_references(cutoff)
    return report


def _is_expired_file(raw_path: str | None, cutoff_timestamp: float, protected_paths: set[str]) -> bool:
    if not raw_path:
        return False
    path = Path(raw_path)
    resolved = str(path.resolve())
    return (
        resolved not in protected_paths
        and path.exists()
        and path.is_file()
        and path.stat().st_mtime < cutoff_timestamp
    )


def _delete_if_needed(raw_path: str | None, should_delete: bool) -> bool:
    if not raw_path or not should_delete:
        return False
    path = Path(raw_path)
    try:
        path.unlink()
    except OSError:
        return False
    return True
