from __future__ import annotations

import shutil
import sqlite3
import subprocess

from wm_platform.config import DEFAULT_DEV_API_KEY, PRODUCTION_ENVIRONMENTS, Settings
from wm_platform.provider_runtime import ProviderRuntime
from wm_platform.runtime_contract import (
    expected_model_entries,
    expected_repo_paths,
    model_destination,
    model_is_present,
)


def _command_dependency(command: str) -> dict[str, object]:
    executable = shutil.which(command)
    if not executable:
        return {"available": False, "path": None, "version": None}

    version_flag = "-version" if command == "ffmpeg" else "--version"
    completed = subprocess.run(
        [executable, version_flag],
        capture_output=True,
        text=True,
        check=False,
    )
    version_line = ""
    if completed.stdout:
        version_line = completed.stdout.splitlines()[0].strip()
    elif completed.stderr:
        version_line = completed.stderr.splitlines()[0].strip()
    return {"available": completed.returncode == 0, "path": executable, "version": version_line or None}


def system_dependency_report() -> dict[str, dict[str, object]]:
    with sqlite3.connect(":memory:") as connection:
        compile_options = [row[0] for row in connection.execute("PRAGMA compile_options").fetchall()]
    return {
        "sqlite3": {
            "available": True,
            "version": sqlite3.sqlite_version,
            "compile_options": compile_options,
        },
        "git": _command_dependency("git"),
        "ffmpeg": _command_dependency("ffmpeg"),
    }


def production_safety_report(settings: Settings) -> dict[str, object]:
    findings: list[str] = []
    if settings.environment in PRODUCTION_ENVIRONMENTS and settings.default_api_key == DEFAULT_DEV_API_KEY:
        findings.append("DWM_DEFAULT_API_KEY is still the development default")
    if settings.allow_private_callback_urls and settings.environment in PRODUCTION_ENVIRONMENTS:
        findings.append("DWM_ALLOW_PRIVATE_CALLBACK_URLS is enabled in production")
    if settings.submit_rate_limit_count <= 0:
        findings.append("submit rate limiting is disabled")
    if settings.file_retention_days < 1:
        findings.append("file retention is shorter than 1 day")
    if settings.auto_allow_copy_fallback:
        findings.append("auto provider is allowed to succeed via ffmpeg_copy fallback")
    findings.extend(_storage_space_findings(settings))

    return {
        "environment": settings.environment,
        "ready": not findings,
        "findings": findings,
    }


def _storage_space_report(settings: Settings) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for label, path in (("inbox", settings.inbox_dir), ("outbox", settings.outbox_dir)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(path)
            device = path.stat().st_dev
        except OSError:
            rows.append({"path": label, "available": False})
            continue
        if device in seen:
            continue
        seen.add(device)
        total = int(usage.total or 0)
        free = int(usage.free)
        rows.append(
            {
                "path": label,
                "available": True,
                "free_bytes": free,
                "total_bytes": total,
                "free_ratio": (free / total) if total else None,
            }
        )
    return rows


def _storage_space_findings(settings: Settings) -> list[str]:
    findings: list[str] = []
    min_free_bytes = 5 * 1024 * 1024 * 1024
    for item in _storage_space_report(settings):
        if not item.get("available"):
            continue
        free = int(item.get("free_bytes") or 0)
        total = int(item.get("total_bytes") or 0)
        ratio = float(item["free_ratio"]) if item.get("free_ratio") is not None else 1.0
        if free < min_free_bytes or ratio < 0.10:
            free_gb = free / (1024 ** 3)
            pct = f"{ratio:.0%}" if total else "unknown"
            findings.append(f"inbox/outbox filesystem low on space: {free_gb:.1f}GB free ({pct})")
    return findings


def model_status_report(settings: Settings) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in expected_model_entries(settings):
        dest = model_destination(settings, entry)
        present = model_is_present(entry, dest)
        required = bool(entry.get("required", False))
        rows.append(
            {
                "name": entry.get("name"),
                "required": required,
                "expected_path": str(dest),
                "present": present,
                "required_missing": required and not present,
                "license": entry.get("license"),
                "commercial_use": entry.get("commercial_use"),
            }
        )
    return rows


def license_report(settings: Settings) -> dict[str, object]:
    """Weights whose license forbids commercial use. Informational: it does not block startup,
    but anyone running this for a business has to see it before the first job."""
    restricted = [
        {"name": entry.get("name"), "license": entry.get("license"), "providers": entry.get("providers")}
        for entry in expected_model_entries(settings)
        if entry.get("commercial_use") is False
    ]
    return {"non_commercial_models": restricted, "commercial_ready": not restricted}


def provider_doctor_report(settings: Settings) -> dict[str, object]:
    probes = [item.model_dump() for item in ProviderRuntime(settings).probe_all()]
    return {
        "environment": settings.environment,
        "storage_root": str(settings.storage_root),
        "comfyui_api_url": settings.comfyui_api_url,
        "comfyui_dir": str(settings.comfyui_dir),
        "comfyui_custom_nodes_dir": str(settings.comfyui_custom_nodes_dir),
        "comfyui_models_dir": str(settings.comfyui_models_dir),
        "comfyui_workflow": str(settings.comfyui_diffueraser_workflow),
        "runtime_root": str(settings.runtime_root),
        "runtime_lock_file": str(settings.runtime_root / "lock.yaml"),
        "runtime_model_manifest": str(settings.runtime_root / "models" / "manifest.yaml"),
        "system_dependencies": system_dependency_report(),
        "production_safety": production_safety_report(settings),
        "expected_repo_paths": [str(path) for path in expected_repo_paths(settings)],
        "expected_model_entries": expected_model_entries(settings),
        "model_status": model_status_report(settings),
        "licenses": license_report(settings),
        "storage_space": _storage_space_report(settings),
        "providers": probes,
    }
