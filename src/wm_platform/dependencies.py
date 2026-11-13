from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header

from wm_platform.bootstrap import bootstrap
from wm_platform.config import Settings, load_settings
from wm_platform.errors import AppError
from wm_platform.repository import JobRepository


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_repository() -> JobRepository:
    return JobRepository(get_settings())


def init_runtime() -> None:
    bootstrap(get_settings())


def get_tenant_id(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    repository: JobRepository = Depends(get_repository),
) -> str:
    return _authenticate_tenant_id(x_api_key=x_api_key, repository=repository)


def _authenticate_tenant_id(x_api_key: str | None, repository: JobRepository) -> str:
    if not x_api_key:
        raise AppError("AUTH_ERROR", "missing X-API-Key", 401)

    tenant_id = repository.authenticate_api_key(x_api_key)
    if not tenant_id:
        raise AppError("AUTH_ERROR", "invalid X-API-Key", 401)
    return tenant_id
