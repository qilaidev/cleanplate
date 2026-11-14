from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from wm_platform.config import Settings
from wm_platform.db import db_connection, sha256_text
from wm_platform.job_locks import is_job_lock_held
from wm_platform.models import CallbackOutboxRecord, CallbackPayload, JobCreate, JobRecord, RunMetadataRecord


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _parse_job(row: Any) -> JobRecord:
    return JobRecord.model_validate(
        {
            "job_id": row["job_id"],
            "tenant_id": row["tenant_id"],
            "media_type": row["media_type"],
            "status": row["status"],
            "provider_requested": row["provider_requested"],
            "provider_selected": row["provider_selected"],
            "fallback_chain_json": row["fallback_chain_json"],
            "idempotency_key": row["idempotency_key"],
            "input_path": row["input_path"],
            "input_signature": row["input_signature"],
            "output_path": row["output_path"],
            "callback_url": row["callback_url"],
            "callback_secret": row["callback_secret"],
            "priority": row["priority"],
            "attempt_count": row["attempt_count"],
            "duration_ms": row["duration_ms"],
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "claimed_at": row["claimed_at"],
            "lock_owner": row["lock_owner"],
            "quality_mode": row["quality_mode"] if "quality_mode" in row.keys() else None,
            "report_json": row["report_json"] if "report_json" in row.keys() else None,
        }
    )


def _dump_report(report: dict[str, Any] | None) -> str | None:
    if report is None:
        return None
    return json.dumps(report, ensure_ascii=False, default=str)


def _parse_callback_outbox(row: Any) -> CallbackOutboxRecord:
    return CallbackOutboxRecord.model_validate(
        {
            "id": row["id"],
            "job_id": row["job_id"],
            "tenant_id": row["tenant_id"],
            "callback_url": row["callback_url"],
            "callback_secret": row["callback_secret"],
            "payload_json": row["payload_json"],
            "status": row["status"],
            "attempt_count": row["attempt_count"],
            "max_attempts": row["max_attempts"],
            "next_attempt_at": row["next_attempt_at"],
            "last_error": row["last_error"],
            "last_response_code": row["last_response_code"],
            "last_response_body": row["last_response_body"],
            "claimed_at": row["claimed_at"],
            "lock_owner": row["lock_owner"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    )


class JobRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def seed_api_key(self, tenant_id: str, api_key: str) -> None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO api_keys (tenant_id, api_key_hash, status, created_at)
                VALUES (?, ?, 'active', ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                  api_key_hash = excluded.api_key_hash,
                  status = 'active'
                """,
                (tenant_id, sha256_text(api_key), now),
            )

    def authenticate_api_key(self, api_key: str) -> str | None:
        with db_connection(self.settings) as connection:
            row = connection.execute(
                """
                SELECT tenant_id
                FROM api_keys
                WHERE api_key_hash = ? AND status = 'active'
                """,
                (sha256_text(api_key),),
            ).fetchone()
        return row["tenant_id"] if row else None

    def create_job(self, payload: JobCreate) -> JobRecord:
        now = utc_now().isoformat()
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        with db_connection(self.settings) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._find_idempotent_job_row(
                connection=connection,
                tenant_id=payload.tenant_id,
                idempotency_key=payload.idempotency_key,
            )
            if existing is not None:
                return _parse_job(existing)

            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, tenant_id, media_type, status, provider_requested,
                    provider_selected, fallback_chain_json, idempotency_key,
                    input_path, input_signature, output_path, callback_url, callback_secret,
                    priority, attempt_count, duration_ms, error_code, error_message,
                    created_at, updated_at, claimed_at, lock_owner, quality_mode
                )
                VALUES (?, ?, ?, 'queued', ?, NULL, ?, ?, ?, ?, NULL, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?, NULL, NULL, ?)
                """,
                (
                    job_id,
                    payload.tenant_id,
                    payload.media_type,
                    payload.provider_requested,
                    payload.fallback_chain_json,
                    payload.idempotency_key,
                    payload.input_path,
                    payload.input_signature,
                    payload.callback_url,
                    payload.callback_secret,
                    payload.priority,
                    now,
                    now,
                    payload.quality_mode,
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _parse_job(row)

    def find_idempotent_job(
        self,
        tenant_id: str,
        idempotency_key: str | None,
    ) -> JobRecord | None:
        if not idempotency_key:
            return None

        with db_connection(self.settings) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE tenant_id = ?
                  AND idempotency_key = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (tenant_id, idempotency_key),
            ).fetchone()
        return _parse_job(row) if row else None

    def _find_idempotent_job_row(
        self,
        connection,
        tenant_id: str,
        idempotency_key: str | None,
    ) -> Any | None:
        if not idempotency_key:
            return None
        return connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE tenant_id = ?
              AND idempotency_key = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (tenant_id, idempotency_key),
        ).fetchone()

    def get_job(self, job_id: str) -> JobRecord | None:
        with db_connection(self.settings) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _parse_job(row) if row else None

    def list_jobs(
        self,
        tenant_id: str,
        status: str | None = None,
        provider: str | None = None,
        media_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[JobRecord]:
        conditions = ["tenant_id = ?"]
        params: list[str | int] = [tenant_id]
        if status:
            conditions.append("status = ?")
            params.append(status)
        if provider:
            conditions.append("(provider_requested = ? OR provider_selected = ?)")
            params.extend([provider, provider])
        if media_type:
            conditions.append("media_type = ?")
            params.append(media_type)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM jobs WHERE {where_clause} ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, max(0, offset)])
        with db_connection(self.settings) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_parse_job(row) for row in rows]

    def reset_stale_claims(self, worker_timeout_seconds: int) -> int:
        cutoff = (utc_now() - timedelta(seconds=worker_timeout_seconds)).isoformat()
        with db_connection(self.settings) as connection:
            rows = connection.execute(
                """
                SELECT job_id
                FROM jobs
                WHERE status = 'running' AND claimed_at < ?
                """,
                (cutoff,),
            ).fetchall()
        reclaimable_job_ids = [
            row["job_id"] for row in rows if not is_job_lock_held(self.settings, str(row["job_id"]))
        ]
        if not reclaimable_job_ids:
            return 0

        placeholders = ", ".join("?" for _ in reclaimable_job_ids)
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                f"""
                UPDATE jobs
                SET status = 'queued',
                    claimed_at = NULL,
                    lock_owner = NULL,
                    updated_at = ?
                WHERE status = 'running'
                  AND job_id IN ({placeholders})
                """,
                (utc_now().isoformat(), *reclaimable_job_ids),
            )
        return int(cursor.rowcount or 0)

    def renew_job_claim(self, job_id: str, lock_owner: str) -> bool:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET claimed_at = ?,
                    updated_at = ?
                WHERE job_id = ? AND status = 'running' AND lock_owner = ?
                """,
                (now, now, job_id, lock_owner),
            )
        return bool(cursor.rowcount)

    def claim_next_job(self, lock_owner: str) -> JobRecord | None:
        now = utc_now().isoformat()
        max_attempts = max(1, self.settings.job_max_attempts)
        with db_connection(self.settings) as connection:
            connection.execute("BEGIN IMMEDIATE")
            while True:
                row = connection.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE status = 'queued'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                    """,
                ).fetchone()
                if row is None:
                    return None

                if int(row["attempt_count"]) >= max_attempts:
                    connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed',
                            error_code = 'JOB_RETRY_EXHAUSTED',
                            error_message = ?,
                            claimed_at = NULL,
                            lock_owner = NULL,
                            updated_at = ?
                        WHERE job_id = ? AND status = 'queued'
                        """,
                        (
                            f"job exceeded max attempts ({max_attempts})",
                            now,
                            row["job_id"],
                        ),
                    )
                    continue

                cursor = connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'running',
                        claimed_at = ?,
                        lock_owner = ?,
                        attempt_count = attempt_count + 1,
                        updated_at = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (now, lock_owner, now, row["job_id"]),
                )
                if cursor.rowcount == 0:
                    return None

                claimed_row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                return _parse_job(claimed_row)

    def cancel_job(self, job_id: str, tenant_id: str) -> JobRecord | None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    claimed_at = NULL,
                    lock_owner = NULL,
                    updated_at = ?
                WHERE job_id = ? AND tenant_id = ? AND status = 'queued'
                """,
                (now, job_id, tenant_id),
            )
            if cursor.rowcount == 0:
                updated_row = connection.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE job_id = ? AND tenant_id = ?
                    """,
                    (job_id, tenant_id),
                ).fetchone()
            else:
                updated_row = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ? AND tenant_id = ?",
                    (job_id, tenant_id),
                ).fetchone()
        return _parse_job(updated_row)

    def release_job_claim(self, job_id: str, lock_owner: str) -> bool:
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    claimed_at = NULL,
                    lock_owner = NULL,
                    updated_at = ?
                WHERE job_id = ? AND status = 'running' AND lock_owner = ?
                """,
                (utc_now().isoformat(), job_id, lock_owner),
            )
        return bool(cursor.rowcount)

    def cleanup_expired_files(self, older_than: datetime) -> dict[str, int]:
        with db_connection(self.settings) as connection:
            rows = connection.execute(
                """
                SELECT job_id, input_path, output_path
                FROM jobs
                WHERE updated_at <= ?
                  AND status NOT IN ('queued', 'running')
                """,
                (older_than.isoformat(),),
            ).fetchall()
        return {
            "candidate_jobs": len(rows),
            "candidate_files": sum(1 for row in rows for path in (row["input_path"], row["output_path"]) if path),
        }

    def list_protected_file_paths(self, older_than: datetime) -> set[str]:
        protected: set[str] = set()
        with db_connection(self.settings) as connection:
            rows = connection.execute(
                """
                SELECT input_path, output_path
                FROM jobs
                WHERE status IN ('queued', 'running') OR updated_at > ?
                """,
                (older_than.isoformat(),),
            ).fetchall()
        for row in rows:
            for path in (row["input_path"], row["output_path"]):
                if path:
                    protected.add(str(path))
                    protected.add(str(Path(path).resolve()))
        return protected

    def list_jobs_for_cleanup(self, older_than: datetime) -> list[JobRecord]:
        with db_connection(self.settings) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE status NOT IN ('queued', 'running')
                  AND updated_at <= ?
                ORDER BY updated_at ASC
                """,
                (older_than.isoformat(),),
            ).fetchall()
        return [_parse_job(row) for row in rows]

    def clear_job_artifacts(self, job_id: str, *, clear_input: bool = False, clear_output: bool = False) -> None:
        updates: list[str] = []
        if clear_input:
            updates.append("input_path = ''")
        if clear_output:
            updates.append("output_path = NULL")
        if not updates:
            return

        updates.append("updated_at = ?")
        with db_connection(self.settings) as connection:
            connection.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?",
                (utc_now().isoformat(), job_id),
            )

    def clear_file_references(self, file_path: str, older_than: datetime) -> int:
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET input_path = CASE WHEN input_path = ? THEN '' ELSE input_path END,
                    output_path = CASE WHEN output_path = ? THEN NULL ELSE output_path END,
                    updated_at = ?
                WHERE updated_at <= ?
                  AND status NOT IN ('queued', 'running')
                  AND (input_path = ? OR output_path = ?)
                """,
                (
                    file_path,
                    file_path,
                    utc_now().isoformat(),
                    older_than.isoformat(),
                    file_path,
                    file_path,
                ),
            )
        return int(cursor.rowcount or 0)

    def clear_missing_file_references(self, older_than: datetime) -> int:
        cleared = 0
        jobs = self.list_jobs_for_cleanup(older_than)
        for job in jobs:
            clear_input = bool(job.input_path) and not Path(str(job.input_path)).exists()
            clear_output = bool(job.output_path) and not Path(str(job.output_path)).exists()
            if not clear_input and not clear_output:
                continue
            self.clear_job_artifacts(job.job_id, clear_input=clear_input, clear_output=clear_output)
            cleared += 1
        return cleared

    def mark_job_succeeded(
        self,
        job_id: str,
        lock_owner: str,
        provider_selected: str,
        output_path: str,
        duration_ms: int,
        report: dict[str, Any] | None = None,
    ) -> JobRecord | None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    provider_selected = ?,
                    output_path = ?,
                    duration_ms = ?,
                    error_code = NULL,
                    error_message = NULL,
                    report_json = ?,
                    updated_at = ?,
                    claimed_at = NULL,
                    lock_owner = NULL
                WHERE job_id = ? AND status = 'running' AND lock_owner = ?
                """,
                (provider_selected, output_path, duration_ms, _dump_report(report), now, job_id, lock_owner),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _parse_job(row)

    def mark_job_failed(
        self,
        job_id: str,
        lock_owner: str,
        provider_selected: str | None,
        error_code: str,
        error_message: str,
        duration_ms: int | None = None,
        report: dict[str, Any] | None = None,
    ) -> JobRecord | None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    provider_selected = COALESCE(?, provider_selected),
                    duration_ms = ?,
                    error_code = ?,
                    error_message = ?,
                    report_json = COALESCE(?, report_json),
                    updated_at = ?,
                    claimed_at = NULL,
                    lock_owner = NULL
                WHERE job_id = ? AND status = 'running' AND lock_owner = ?
                """,
                (provider_selected, duration_ms, error_code, error_message, _dump_report(report), now, job_id, lock_owner),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _parse_job(row)

    def record_callback_event(
        self,
        job_id: str,
        attempt_no: int,
        status: str,
        response_code: int | None,
        response_body: str | None,
    ) -> None:
        with db_connection(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO callback_events (job_id, attempt_no, status, response_code, response_body, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, attempt_no, status, response_code, response_body, utc_now().isoformat()),
            )

    def get_callback_events(self, job_id: str) -> list[dict[str, Any]]:
        with db_connection(self.settings) as connection:
            rows = connection.execute(
                "SELECT * FROM callback_events WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "attempt_no": row["attempt_no"],
                "status": row["status"],
                "response_code": row["response_code"],
                "response_body": row["response_body"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def enqueue_callback(self, job: JobRecord) -> None:
        if not job.callback_url:
            return
        now = utc_now().isoformat()
        payload_json = CallbackPayload(
            job_id=job.job_id,
            status=job.status,
            provider=job.provider_selected,
            degraded=job.provider_selected == "local_fallback",
            output_path=job.output_path,
            error_code=job.error_code,
            error_message=job.error_message,
            needs_review=job.needs_review,
            report=job.report,
        ).model_dump_json()
        with db_connection(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO callback_outbox (
                    job_id, tenant_id, callback_url, callback_secret, payload_json,
                    status, attempt_count, max_attempts, next_attempt_at, last_error,
                    last_response_code, last_response_body, claimed_at, lock_owner,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)
                """,
                (
                    job.job_id,
                    job.tenant_id,
                    job.callback_url,
                    job.callback_secret,
                    payload_json,
                    max(1, self.settings.callback_retry_count),
                    now,
                    now,
                    now,
                ),
            )

    def reset_stale_callback_claims(self, worker_timeout_seconds: int) -> int:
        cutoff = (utc_now() - timedelta(seconds=worker_timeout_seconds)).isoformat()
        with db_connection(self.settings) as connection:
            cursor = connection.execute(
                """
                UPDATE callback_outbox
                SET status = CASE
                      WHEN attempt_count >= max_attempts THEN 'failed'
                      ELSE 'pending'
                    END,
                    claimed_at = NULL,
                    lock_owner = NULL,
                    updated_at = ?
                WHERE status = 'delivering' AND claimed_at < ?
                """,
                (utc_now().isoformat(), cutoff),
            )
        return int(cursor.rowcount or 0)

    def claim_next_callback(self, lock_owner: str) -> CallbackOutboxRecord | None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM callback_outbox
                WHERE status = 'pending'
                  AND next_attempt_at <= ?
                ORDER BY next_attempt_at ASC, id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None

            cursor = connection.execute(
                """
                UPDATE callback_outbox
                SET status = 'delivering',
                    claimed_at = ?,
                    lock_owner = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, lock_owner, now, row["id"]),
            )
            if cursor.rowcount == 0:
                return None

            claimed_row = connection.execute("SELECT * FROM callback_outbox WHERE id = ?", (row["id"],)).fetchone()
        return _parse_callback_outbox(claimed_row)

    def mark_callback_succeeded(
        self,
        outbox_id: int,
        lock_owner: str,
        response_code: int,
        response_body: str | None,
    ) -> None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM callback_outbox
                WHERE id = ? AND status = 'delivering' AND lock_owner = ?
                """,
                (outbox_id, lock_owner),
            ).fetchone()
            if row is None:
                return

            next_attempt = int(row["attempt_count"]) + 1
            connection.execute(
                """
                UPDATE callback_outbox
                SET status = 'succeeded',
                    attempt_count = ?,
                    last_error = NULL,
                    last_response_code = ?,
                    last_response_body = ?,
                    claimed_at = NULL,
                    lock_owner = NULL,
                    updated_at = ?
                WHERE id = ? AND status = 'delivering' AND lock_owner = ?
                """,
                (next_attempt, response_code, response_body, now, outbox_id, lock_owner),
            )
            connection.execute(
                """
                INSERT INTO callback_events (job_id, attempt_no, status, response_code, response_body, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row["job_id"], next_attempt, "succeeded", response_code, response_body, now),
            )

    def mark_callback_retry(
        self,
        outbox_id: int,
        lock_owner: str,
        response_code: int | None,
        response_body: str | None,
        next_attempt_at: datetime,
    ) -> None:
        now = utc_now().isoformat()
        with db_connection(self.settings) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM callback_outbox
                WHERE id = ? AND status = 'delivering' AND lock_owner = ?
                """,
                (outbox_id, lock_owner),
            ).fetchone()
            if row is None:
                return

            next_attempt = int(row["attempt_count"]) + 1
            final_status = "failed" if next_attempt >= int(row["max_attempts"]) else "pending"
            event_status = "failed" if final_status == "failed" else "retrying"
            connection.execute(
                """
                UPDATE callback_outbox
                SET status = ?,
                    attempt_count = ?,
                    next_attempt_at = ?,
                    last_error = ?,
                    last_response_code = ?,
                    last_response_body = ?,
                    claimed_at = NULL,
                    lock_owner = NULL,
                    updated_at = ?
                WHERE id = ? AND status = 'delivering' AND lock_owner = ?
                """,
                (
                    final_status,
                    next_attempt,
                    next_attempt_at.isoformat(),
                    response_body,
                    response_code,
                    response_body,
                    now,
                    outbox_id,
                    lock_owner,
                ),
            )
            connection.execute(
                """
                INSERT INTO callback_events (job_id, attempt_no, status, response_code, response_body, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row["job_id"], next_attempt, event_status, response_code, response_body, now),
            )

    def get_callback_outbox(self, job_id: str) -> list[CallbackOutboxRecord]:
        with db_connection(self.settings) as connection:
            rows = connection.execute(
                "SELECT * FROM callback_outbox WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        return [_parse_callback_outbox(row) for row in rows]

    @staticmethod
    def default_fallback_chain(provider_requested: str) -> str:
        if provider_requested == "auto":
            chain = ["minimax_remover", "comfy_diffueraser"]
        else:
            chain = [provider_requested]
        return json.dumps(chain)

    def record_run_metadata(self, metadata: RunMetadataRecord) -> None:
        with db_connection(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO run_metadata (
                    job_id, workflow_name, quality_profile, steps,
                    subvideo_length, neighbor_length, mask_dilation_iter,
                    device, seed, scene_type, confidence_level, mask_ratio, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.job_id,
                    metadata.workflow_name,
                    metadata.quality_profile,
                    metadata.steps,
                    metadata.subvideo_length,
                    metadata.neighbor_length,
                    metadata.mask_dilation_iter,
                    metadata.device,
                    metadata.seed,
                    metadata.scene_type,
                    metadata.confidence_level,
                    metadata.mask_ratio,
                    metadata.created_at.isoformat(),
                ),
            )

    def get_run_metadata(self, job_id: str) -> RunMetadataRecord | None:
        with db_connection(self.settings) as connection:
            row = connection.execute(
                "SELECT * FROM run_metadata WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return RunMetadataRecord(
            id=row["id"],
            job_id=row["job_id"],
            workflow_name=row["workflow_name"],
            quality_profile=row["quality_profile"],
            steps=row["steps"],
            subvideo_length=row["subvideo_length"],
            neighbor_length=row["neighbor_length"],
            mask_dilation_iter=row["mask_dilation_iter"],
            device=row["device"],
            seed=row["seed"],
            scene_type=row["scene_type"],
            confidence_level=row["confidence_level"],
            mask_ratio=row["mask_ratio"] if "mask_ratio" in row.keys() else None,
            created_at=row["created_at"],
        )
