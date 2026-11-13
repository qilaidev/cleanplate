from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from wm_platform.config import load_settings
from wm_platform.db import db_connection, init_db
from wm_platform.repository import JobRepository
from wm_platform.storage import ensure_storage_dirs


@pytest.fixture(scope="session", autouse=True)
def temp_storage_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("storage_root")
    os.environ["DWM_STORAGE_ROOT"] = str(root)
    os.environ["DWM_CALLBACK_RETRY_DELAY_SECONDS"] = "0"
    return root


@pytest.fixture(scope="session")
def settings(temp_storage_root):
    settings = load_settings()
    ensure_storage_dirs(settings)
    init_db(settings)
    JobRepository(settings).seed_api_key(settings.default_tenant_id, settings.default_api_key)
    return settings


@pytest.fixture(scope="session")
def job_repo(settings) -> JobRepository:
    repository = JobRepository(settings)
    repository.seed_api_key(settings.default_tenant_id, settings.default_api_key)
    return repository


@pytest.fixture(scope="session")
def api_client(settings):
    api_module = pytest.importorskip("apps.api.main")
    app = getattr(api_module, "app", None)
    if app is None:
        pytest.skip("apps.api.main has no FastAPI app")
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers(settings):
    return {"X-API-Key": settings.default_api_key}


@pytest.fixture(autouse=True)
def clean_runtime_state(settings):
    for directory in (settings.inbox_dir, settings.outbox_dir):
        for path in directory.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file():
                path.unlink()

    with db_connection(settings) as connection:
        connection.execute("DELETE FROM callback_events")
        connection.execute("DELETE FROM callback_outbox")
        connection.execute("DELETE FROM run_metadata")
        connection.execute("DELETE FROM jobs")

    yield
