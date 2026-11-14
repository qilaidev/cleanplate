from __future__ import annotations

import hashlib
import hmac
import inspect
import logging
import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx

from wm_platform.callbacks import validate_callback_url
from wm_platform.config import Settings
from wm_platform.job_locks import JobFileLock
from wm_platform.models import CallbackOutboxRecord, JobRecord
from wm_platform.provider_runtime import ProviderExecutionError, ProviderRuntime
from wm_platform.repository import JobRepository

logger = logging.getLogger(__name__)


class WorkerService:
    def __init__(self, settings: Settings, repository: JobRepository, providers: ProviderRuntime) -> None:
        self.settings = settings
        self.repository = repository
        self.providers = providers
        self.lock_owner = f"{socket.gethostname()}:{int(time.time())}"
        self._running = True
        self.callback_service = CallbackWorkerService(
            settings=settings,
            repository=repository,
            lock_owner=f"{self.lock_owner}:callbacks",
        )

    def run_forever(self) -> None:
        logger.info("worker started lock_owner=%s db=%s", self.lock_owner, self.settings.db_path)
        callback_thread = threading.Thread(target=self.callback_service.run_forever, name="callback-worker", daemon=True)
        callback_thread.start()
        try:
            while self._running:
                try:
                    processed = self.run_once()
                    if not processed:
                        time.sleep(max(0.0, self.settings.worker_poll_interval_seconds))
                except Exception as exc:
                    # Catch any unexpected exceptions in run_once to prevent worker from crashing
                    logger.exception("unhandled exception in worker run_once: %s", exc)
                    # Wait a bit before retrying to prevent tight error loops
                    time.sleep(max(1.0, self.settings.worker_poll_interval_seconds * 5))
        except Exception as exc:
            # Log the final exception before exiting
            logger.critical("worker shutting down due to unhandled exception: %s", exc)
        finally:
            self.callback_service.stop()
            callback_thread.join(timeout=2.0)

    def run_once(self) -> bool:
        reclaimed = self.repository.reset_stale_claims(self.settings.job_claim_timeout_seconds)
        if reclaimed:
            logger.warning("reclaimed %d stale running jobs", reclaimed)

        job = self.repository.claim_next_job(lock_owner=self.lock_owner)
        if not job:
            return False

        self._process_job(job)
        return True

    def stop(self) -> None:
        self._running = False
        self.callback_service.stop()

    def _process_job(self, job: JobRecord) -> None:
        job_lock = JobFileLock(self.settings, job.job_id)
        if not job_lock.acquire():
            logger.warning("job lock already held, releasing duplicate claim job_id=%s", job.job_id)
            self.repository.release_job_claim(job_id=job.job_id, lock_owner=self.lock_owner)
            return

        started_at = time.monotonic()
        logger.info("processing job=%s provider_requested=%s", job.job_id, job.provider_requested)
        progress_lock = threading.Lock()
        progress_state: dict[str, float | str] = {"last_progress": started_at, "stage": "claimed"}

        def _touch_progress(stage: str) -> None:
            with progress_lock:
                progress_state["last_progress"] = time.monotonic()
                progress_state["stage"] = stage

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_job_claim,
            args=(job.job_id, heartbeat_stop, progress_lock, progress_state, job_lock),
            name=f"job-heartbeat-{job.job_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        updated: JobRecord | None = None
        try:
            provider_name, output_path, *rest = self._invoke_run_with_fallback(job, _touch_progress)
            report = rest[0] if rest else None
            duration_ms = self._duration_ms(started_at)
            updated = self.repository.mark_job_succeeded(
                job_id=job.job_id,
                lock_owner=self.lock_owner,
                provider_selected=provider_name,
                output_path=output_path,
                duration_ms=duration_ms,
                report=report,
            )
            if updated is None:
                logger.warning("skip success finalize for stale claim job_id=%s", job.job_id)
                return
            logger.info(
                "job succeeded job_id=%s provider=%s duration_ms=%s output_path=%s",
                job.job_id,
                provider_name,
                duration_ms,
                output_path,
            )
        except ProviderExecutionError as exc:
            duration_ms = self._duration_ms(started_at)
            updated = self.repository.mark_job_failed(
                job_id=job.job_id,
                lock_owner=self.lock_owner,
                provider_selected=exc.provider_selected,
                error_code=exc.error_code,
                error_message=exc.error_message,
                duration_ms=duration_ms,
                report=exc.report,
            )
            if updated is None:
                logger.warning("skip failure finalize for stale claim job_id=%s", job.job_id)
                return
            logger.error(
                "job failed job_id=%s provider=%s error_code=%s error=%s duration_ms=%s",
                job.job_id,
                exc.provider_selected,
                exc.error_code,
                exc.error_message,
                duration_ms,
            )
        except Exception as exc:
            duration_ms = self._duration_ms(started_at)
            updated = self.repository.mark_job_failed(
                job_id=job.job_id,
                lock_owner=self.lock_owner,
                provider_selected=None,
                error_code="INTERNAL_ERROR",
                error_message=f"worker exception: {exc}",
                duration_ms=duration_ms,
            )
            if updated is None:
                logger.warning("skip exception finalize for stale claim job_id=%s", job.job_id)
                return
            logger.exception("unhandled worker exception job_id=%s duration_ms=%s", job.job_id, duration_ms)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
            job_lock.release()

        # Only enqueue callback if we have a valid updated job record
        if updated is not None:
            self.repository.enqueue_callback(updated)

    def _invoke_run_with_fallback(
        self,
        job: JobRecord,
        progress_callback: Callable[[str], None],
    ):
        runner = self.providers.run_with_fallback
        try:
            params = inspect.signature(runner).parameters
        except (TypeError, ValueError):
            params = {}
        if "progress_callback" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        ):
            return runner(job, progress_callback=progress_callback)
        return runner(job)

    def _heartbeat_job_claim(
        self,
        job_id: str,
        stop_event: threading.Event,
        progress_lock: threading.Lock | None = None,
        progress_state: dict[str, float | str] | None = None,
        job_lock: JobFileLock | None = None,
    ) -> None:
        interval = max(
            0.01,
            min(
                self.settings.job_claim_heartbeat_seconds,
                max(0.01, self.settings.job_claim_timeout_seconds / 3),
            ),
        )
        watchdog_limit = float(getattr(self.settings, "job_watchdog_timeout_seconds", 0.0) or 0.0)
        while not stop_event.wait(interval):
            if watchdog_limit > 0 and progress_lock is not None and progress_state is not None:
                with progress_lock:
                    last_progress = float(progress_state.get("last_progress", 0.0))
                    stage = str(progress_state.get("stage", "unknown"))
                stalled_seconds = time.monotonic() - last_progress
                if stalled_seconds > watchdog_limit:
                    logger.error(
                        "job watchdog expired after %.1fs without stage progress job_id=%s stage=%s; "
                        "stopping heartbeat and releasing file lock so stale claim recovery can proceed",
                        stalled_seconds,
                        job_id,
                        stage,
                    )
                    if job_lock is not None:
                        job_lock.release()
                    return
            renewed = self.repository.renew_job_claim(job_id=job_id, lock_owner=self.lock_owner)
            if not renewed:
                logger.warning("lost job claim during heartbeat job_id=%s lock_owner=%s", job_id, self.lock_owner)
                return

    @staticmethod
    def _build_signature(secret: str, timestamp: str, body: str) -> str:
        message = f"{timestamp}.{body}".encode()
        return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()

    @staticmethod
    def _duration_ms(started_at: float) -> int:
        return int((time.monotonic() - started_at) * 1000)


class CallbackWorkerService:
    def __init__(self, settings: Settings, repository: JobRepository, lock_owner: str | None = None) -> None:
        self.settings = settings
        self.repository = repository
        self.lock_owner = lock_owner or f"{socket.gethostname()}:{int(time.time())}:callbacks"
        self._running = True

    def run_forever(self) -> None:
        logger.info("callback worker started lock_owner=%s db=%s", self.lock_owner, self.settings.db_path)
        try:
            while self._running:
                try:
                    processed = self.run_once()
                    if not processed:
                        time.sleep(max(0.0, self.settings.worker_poll_interval_seconds))
                except Exception as exc:
                    # Catch any unexpected exceptions to prevent callback worker from crashing
                    logger.exception("unhandled exception in callback worker run_once: %s", exc)
                    time.sleep(max(1.0, self.settings.worker_poll_interval_seconds * 5))
        except Exception as exc:
            logger.critical("callback worker shutting down due to unhandled exception: %s", exc)

    def run_once(self) -> bool:
        reclaimed = self.repository.reset_stale_callback_claims(self.settings.job_claim_timeout_seconds)
        if reclaimed:
            logger.warning("reclaimed %d stale callback deliveries", reclaimed)

        delivery = self.repository.claim_next_callback(lock_owner=self.lock_owner)
        if not delivery:
            return False

        self._process_delivery(delivery)
        return True

    def stop(self) -> None:
        self._running = False

    def _process_delivery(self, delivery: CallbackOutboxRecord) -> None:
        headers = {"Content-Type": "application/json"}
        timestamp = str(int(datetime.now(UTC).timestamp()))
        if delivery.callback_secret:
            signature = WorkerService._build_signature(delivery.callback_secret, timestamp, delivery.payload_json)
            headers["X-Timestamp"] = timestamp
            headers["X-Signature"] = signature

        try:
            callback_url = validate_callback_url(delivery.callback_url, self.settings)
            with httpx.Client(timeout=10.0) as client:
                response = client.post(callback_url, content=delivery.payload_json, headers=headers)
            if 200 <= response.status_code < 300:
                self.repository.mark_callback_succeeded(
                    outbox_id=delivery.id,
                    lock_owner=self.lock_owner,
                    response_code=response.status_code,
                    response_body=response.text[:1000],
                )
                return

            self.repository.mark_callback_retry(
                outbox_id=delivery.id,
                lock_owner=self.lock_owner,
                response_code=response.status_code,
                response_body=response.text[:1000],
                next_attempt_at=self._next_attempt_at(delivery.attempt_count),
            )
        except Exception as exc:
            self.repository.mark_callback_retry(
                outbox_id=delivery.id,
                lock_owner=self.lock_owner,
                response_code=None,
                response_body=str(exc)[:1000],
                next_attempt_at=self._next_attempt_at(delivery.attempt_count),
            )

    def _next_attempt_at(self, attempt_count: int = 0) -> datetime:
        # Exponential backoff (delay, 2x, 4x, ...) capped at one hour, so a receiver that
        # is down for a few minutes does not burn through every retry.
        delay = max(0.0, self.settings.callback_retry_delay_seconds) * (2 ** max(0, attempt_count))
        return datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=min(delay, 3600.0))
