from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from wm_platform.config import Settings

logger = logging.getLogger(__name__)

# Maximum number of retries for database operations
MAX_DB_RETRIES = 3
# Base delay for exponential backoff (seconds)
BASE_RETRY_DELAY = 0.1


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    media_type TEXT NOT NULL,
    status TEXT NOT NULL,
    provider_requested TEXT NOT NULL,
    provider_selected TEXT,
    fallback_chain_json TEXT NOT NULL,
    idempotency_key TEXT,
    input_path TEXT NOT NULL,
    input_signature TEXT,
    output_path TEXT,
    callback_url TEXT,
    callback_secret TEXT,
    priority INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    lock_owner TEXT,
    quality_mode TEXT,
    report_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_priority_created
ON jobs(status, priority DESC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_jobs_idempotency
ON jobs(tenant_id, idempotency_key, input_signature, input_path);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_jobs_tenant_idem_signature
ON jobs(tenant_id, idempotency_key, input_signature)
WHERE idempotency_key IS NOT NULL AND input_signature IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uniq_jobs_tenant_idem_path
ON jobs(tenant_id, idempotency_key, input_path)
WHERE idempotency_key IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uniq_jobs_tenant_idempotency_key
ON jobs(tenant_id, idempotency_key)
WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS api_keys (
    tenant_id TEXT PRIMARY KEY,
    api_key_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS callback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    response_code INTEGER,
    response_body TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS callback_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    callback_url TEXT NOT NULL,
    callback_secret TEXT,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    next_attempt_at TEXT NOT NULL,
    last_error TEXT,
    last_response_code INTEGER,
    last_response_body TEXT,
    claimed_at TEXT,
    lock_owner TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_callback_outbox_status_next_attempt
ON callback_outbox(status, next_attempt_at);

CREATE INDEX IF NOT EXISTS idx_callback_outbox_job_id
ON callback_outbox(job_id);

CREATE INDEX IF NOT EXISTS idx_callback_outbox_status_lock_claimed
ON callback_outbox(status, lock_owner, claimed_at);
CREATE TABLE IF NOT EXISTS run_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    workflow_name TEXT,
    quality_profile TEXT,
    steps INTEGER,
    subvideo_length INTEGER,
    neighbor_length INTEGER,
    mask_dilation_iter INTEGER,
    device TEXT,
    seed INTEGER,
    scene_type TEXT,
    confidence_level TEXT,
    mask_ratio REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_metadata_job_id
ON run_metadata(job_id);
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA foreign_keys = ON")


def init_db(settings: Settings) -> None:
    ensure_parent_dir(settings.db_path)
    with sqlite3.connect(settings.db_path) as connection:
        _configure_connection(connection)
        try:
            connection.executescript(SCHEMA)
        except sqlite3.OperationalError as exc:
            if "uniq_jobs_tenant_idempotency_key" not in str(exc):
                raise
            logger.warning("skipped tenant idempotency unique index: %s", exc)
            _init_schema_without_tenant_key_index(connection)
        _ensure_run_metadata_columns(connection)
        _ensure_job_columns(connection)
        connection.commit()


def _init_schema_without_tenant_key_index(connection: sqlite3.Connection) -> None:
    stripped = SCHEMA.replace(
        """
CREATE UNIQUE INDEX IF NOT EXISTS uniq_jobs_tenant_idempotency_key
ON jobs(tenant_id, idempotency_key)
WHERE idempotency_key IS NOT NULL;
""",
        "",
    )
    connection.executescript(stripped)


def _ensure_run_metadata_columns(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(run_metadata)").fetchall()}
    if "mask_ratio" not in columns:
        connection.execute("ALTER TABLE run_metadata ADD COLUMN mask_ratio REAL")


def _ensure_job_columns(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()}
    if "quality_mode" not in columns:
        connection.execute("ALTER TABLE jobs ADD COLUMN quality_mode TEXT")
    if "report_json" not in columns:
        connection.execute("ALTER TABLE jobs ADD COLUMN report_json TEXT")


def _is_retryable_error(exc: Exception) -> bool:
    """Check if the error is retryable (database locked, busy, etc.)"""
    if isinstance(exc, sqlite3.OperationalError):
        error_msg = str(exc).lower()
        return any(keyword in error_msg for keyword in ["locked", "busy", "database is locked"])
    return False


def _execute_with_retry(func, description: str = "database operation"):
    """Execute a database operation with retry logic for transient errors."""
    last_exc: Exception | None = None
    for attempt in range(MAX_DB_RETRIES):
        try:
            return func()
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            last_exc = exc
            delay = BASE_RETRY_DELAY * (2 ** attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s, retrying in %.2fs",
                description,
                attempt + 1,
                MAX_DB_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


@contextmanager
def db_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
    def _create_connection():
        connection = sqlite3.connect(settings.db_path, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
        connection.row_factory = sqlite3.Row
        _configure_connection(connection)
        return connection

    connection = _execute_with_retry(_create_connection, "create connection")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
