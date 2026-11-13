from __future__ import annotations

from wm_platform.config import DEFAULT_DEV_API_KEY, PRODUCTION_ENVIRONMENTS, Settings, load_settings
from wm_platform.db import init_db
from wm_platform.repository import JobRepository
from wm_platform.storage import ensure_storage_dirs


def _validate_startup_safety(settings: Settings) -> None:
    if settings.environment in PRODUCTION_ENVIRONMENTS and settings.default_api_key == DEFAULT_DEV_API_KEY:
        raise RuntimeError("DWM_DEFAULT_API_KEY must be changed when DWM_ENV=production")


def bootstrap(settings: Settings | None = None) -> tuple[Settings, JobRepository]:
    resolved_settings = settings or load_settings()
    _validate_startup_safety(resolved_settings)
    ensure_storage_dirs(resolved_settings)
    init_db(resolved_settings)
    repository = JobRepository(resolved_settings)
    repository.seed_api_key(resolved_settings.default_tenant_id, resolved_settings.default_api_key)
    return resolved_settings, repository
