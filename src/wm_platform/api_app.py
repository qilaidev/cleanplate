from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from wm_platform import __version__
from wm_platform.api_validation import (
    blank_to_none,
    normalize_callback_url,
    normalize_idempotency_key,
    normalize_priority,
    normalize_quality_mode,
    require_known_media_type,
    require_known_provider,
    require_known_status,
)
from wm_platform.callbacks import validate_callback_url
from wm_platform.config import QUALITY_MODE_VALUES
from wm_platform.db import db_connection
from wm_platform.dependencies import get_repository, get_settings, get_tenant_id, init_runtime
from wm_platform.errors import AppError, error_payload
from wm_platform.media import probe_video_duration_seconds
from wm_platform.models import (
    CancelJobResponse,
    ErrorResponse,
    HealthResponse,
    JobCreate,
    JobListResponse,
    JobRecord,
    JobResponse,
    JobResultResponse,
    JobSubmitResponse,
    ProviderListResponse,
)
from wm_platform.provider_runtime import ProviderRuntime
from wm_platform.rate_limit import enforce_submit_rate_limit
from wm_platform.repository import JobRepository
from wm_platform.storage import is_under_directory, save_upload_file, validate_local_input_path

logger = logging.getLogger(__name__)


def _same_idempotent_request(
    existing,
    *,
    media_type: str,
    provider: str,
    callback_url: str | None,
    callback_secret: str | None,
    priority: int,
    input_signature: str | None,
    input_path: str,
    quality_mode: str | None,
    default_quality_mode: str,
) -> bool:
    same_input = False
    if existing.input_signature and input_signature:
        same_input = existing.input_signature == input_signature
    else:
        same_input = existing.input_path == input_path

    return (
        same_input
        and existing.media_type == media_type
        and existing.provider_requested == provider
        and existing.callback_url == callback_url
        and existing.callback_secret == callback_secret
        and existing.priority == priority
        and _same_quality_mode(getattr(existing, "quality_mode", None), quality_mode, default_quality_mode)
    )


def _same_quality_mode(
    existing_mode: str | None,
    requested_mode: str | None,
    default_mode: str,
) -> bool:
    default = (default_mode or "balanced").strip().lower() or "balanced"
    if default not in QUALITY_MODE_VALUES:
        default = "balanced"
    existing = (existing_mode or "").strip().lower() or default
    requested = (requested_mode or "").strip().lower() or default
    return existing == requested


OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Liveness, SQLite WAL connectivity, and provider/license readiness probes.",
    },
    {
        "name": "jobs",
        "description": "Asynchronous video watermark detection, multi-ROI inpainting, QA verification, and artifact retrieval.",
    },
]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        init_runtime()
        yield

    app = FastAPI(
        title="Cleanplate API",
        summary="Self-hosted asynchronous video watermark detection, multi-ROI inpainting, and QA verification service.",
        description=(
            "REST queue contract for Cleanplate. Submits video clips via `POST /v1/jobs`, runs automatic "
            "OpenCV + ComfyUI SAM 3.1 watermark detection, 81-frame (`4k+1`) multi-ROI diffusion inpainting "
            "with clean-chunk GPU bypass, lossless audio/SDR color-tag preservation, and post-inpaint QA gates, "
            "then delivers artifacts and structured JSON `report` payloads via polling or HMAC-SHA256 webhooks."
        ),
        version=__version__,
        contact={"name": "qilaidev", "url": "https://github.com/qilaidev/cleanplate"},
        license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
        openapi_tags=OPENAPI_TAGS,
        lifespan=_lifespan,
    )

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.error_code, exc.error_message),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_payload("VALIDATION_ERROR", str(exc.errors()[0].get("msg", "request validation failed"))),
        )

    @app.exception_handler(ValidationError)
    async def _handle_model_validation_error(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content=error_payload("VALIDATION_ERROR", str(exc)))

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled api exception")
        return JSONResponse(
            status_code=500,
            content=error_payload("INTERNAL_ERROR", "internal server error"),
        )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["health"],
        operation_id="healthCheck",
        summary="Check API and SQLite database health",
    )
    def healthz() -> HealthResponse:
        settings = get_settings()
        with db_connection(settings) as connection:
            connection.execute("SELECT 1").fetchone()
        return HealthResponse(status="ok", db="ok")

    @app.post(
        "/v1/jobs",
        response_model=JobSubmitResponse,
        tags=["jobs"],
        operation_id="submitJob",
        summary="Submit an asynchronous video watermark removal job",
        responses={
            400: {"model": ErrorResponse, "description": "Validation, media type, file size, or duration error"},
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
            409: {"model": ErrorResponse, "description": "Idempotency-Key reused with different parameters"},
            429: {"model": ErrorResponse, "description": "Per-tenant submit rate limit exceeded (includes Retry-After)"},
        },
    )
    def submit_job(
        media_type: str = Form(...),
        provider: str = Form(default="auto"),
        callback_url: str | None = Form(default=None),
        callback_secret: str | None = Form(default=None),
        input_path: str | None = Form(default=None),
        priority: int = Form(default=0),
        quality_mode: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        tenant_id: str = Depends(get_tenant_id),
        repository: JobRepository = Depends(get_repository),
    ) -> JobSubmitResponse:
        settings = get_settings()
        enforce_submit_rate_limit(tenant_id, settings)
        if media_type != "video":
            raise AppError("MEDIA_TYPE_NOT_SUPPORTED", "only video is supported", 400)
        require_known_provider(provider)
        quality_mode = normalize_quality_mode(quality_mode, settings.quality_mode)
        priority = normalize_priority(priority)
        idempotency_key = normalize_idempotency_key(idempotency_key)
        callback_url = normalize_callback_url(callback_url)
        callback_secret = blank_to_none(callback_secret)
        input_path = blank_to_none(input_path)
        if (file is None and input_path is None) or (file is not None and input_path is not None):
            raise AppError("VALIDATION_ERROR", "provide exactly one of file or input_path", 400)
        if callback_url is not None:
            callback_url = validate_callback_url(callback_url, settings)

        if file is not None:
            final_input_path, input_signature = save_upload_file(file, settings)
        else:
            validated_path, input_signature = validate_local_input_path(input_path or "", settings)
            final_input_path = validated_path
        try:
            _enforce_video_duration(final_input_path, settings)
        except AppError:
            _discard_unused_upload(file, final_input_path, None)
            raise

        existing = repository.find_idempotent_job(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if not _same_idempotent_request(
                existing,
                media_type=media_type,
                provider=provider,
                callback_url=callback_url,
                callback_secret=callback_secret,
                priority=priority,
                input_signature=input_signature,
                input_path=str(final_input_path),
                quality_mode=quality_mode,
                default_quality_mode=settings.quality_mode,
            ):
                _discard_unused_upload(file, final_input_path, existing)
                raise AppError(
                    "IDEMPOTENCY_CONFLICT",
                    "idempotency key already used with different request parameters",
                    409,
                )
            _discard_unused_upload(file, final_input_path, existing)
            return _submit_response(existing)

        payload = JobCreate(
            tenant_id=tenant_id,
            media_type="video",
            provider_requested=provider,  # type: ignore[arg-type]
            fallback_chain_json=repository.default_fallback_chain(provider_requested=provider),
            idempotency_key=idempotency_key,
            input_path=str(final_input_path),
            input_signature=input_signature,
            callback_url=callback_url,
            callback_secret=callback_secret,
            priority=priority,
            quality_mode=quality_mode,
        )
        created = repository.create_job(payload)
        if not _same_idempotent_request(
            created,
            media_type=media_type,
            provider=provider,
            callback_url=callback_url,
            callback_secret=callback_secret,
            priority=priority,
            input_signature=input_signature,
            input_path=str(final_input_path),
            quality_mode=quality_mode,
            default_quality_mode=settings.quality_mode,
        ):
            _discard_unused_upload(file, final_input_path, created)
            raise AppError(
                "IDEMPOTENCY_CONFLICT",
                "idempotency key already used with different request parameters",
                409,
            )
        _discard_unused_upload(file, final_input_path, created)
        return _submit_response(created)

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        tags=["jobs"],
        operation_id="getJob",
        summary="Get job status, error state, and structured QA report",
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
            404: {"model": ErrorResponse, "description": "Job not found for tenant"},
        },
    )
    def get_job(
        job_id: str,
        tenant_id: str = Depends(get_tenant_id),
        repository: JobRepository = Depends(get_repository),
    ) -> JobResponse:
        job = repository.get_job(job_id)
        if job is None or job.tenant_id != tenant_id:
            raise AppError("JOB_NOT_FOUND", "job not found", 404)
        return JobResponse.from_record(job)

    @app.get(
        "/v1/jobs",
        response_model=JobListResponse,
        tags=["jobs"],
        operation_id="listJobs",
        summary="List tenant jobs with optional status and provider filters",
        responses={401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"}},
    )
    def list_jobs(
        status: str | None = None,
        provider: str | None = None,
        media_type: str | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        tenant_id: str = Depends(get_tenant_id),
        repository: JobRepository = Depends(get_repository),
    ) -> JobListResponse:
        status = require_known_status(blank_to_none(status))
        provider = require_known_provider(blank_to_none(provider))
        media_type = require_known_media_type(blank_to_none(media_type))

        records = repository.list_jobs(
            tenant_id=tenant_id,
            status=status,
            provider=provider,
            media_type=media_type,
            limit=page_size + 1,
            offset=(page - 1) * page_size,
        )
        return JobListResponse(
            jobs=[JobResponse.from_record(item) for item in records[:page_size]],
            page=page,
            page_size=page_size,
            has_more=len(records) > page_size,
        )

    @app.get(
        "/v1/jobs/{job_id}/result",
        response_model=JobResultResponse,
        tags=["jobs"],
        operation_id="getJobResult",
        summary="Get final job result metadata, download URL, and QA report",
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
            404: {"model": ErrorResponse, "description": "Job not found for tenant"},
            410: {"model": ErrorResponse, "description": "Succeeded job artifact was pruned or moved"},
        },
    )
    def get_job_result(
        job_id: str,
        tenant_id: str = Depends(get_tenant_id),
        repository: JobRepository = Depends(get_repository),
    ) -> JobResultResponse:
        job = repository.get_job(job_id)
        if job is None or job.tenant_id != tenant_id:
            raise AppError("JOB_NOT_FOUND", "job not found", 404)
        if job.status == "succeeded" and (not job.output_path or not Path(job.output_path).exists()):
            raise AppError("ARTIFACT_MISSING", "job artifact missing", 410)
        return JobResultResponse(
            job_id=job.job_id,
            status=job.status,
            provider_selected=job.provider_selected,
            degraded=job.provider_selected == "local_fallback",
            output_path=job.output_path,
            download_url=f"/v1/jobs/{job.job_id}/artifact" if job.status == "succeeded" and job.output_path else None,
            needs_review=job.needs_review,
            report=job.report,
        )

    @app.get(
        "/v1/jobs/{job_id}/artifact",
        tags=["jobs"],
        operation_id="downloadJobArtifact",
        summary="Download the clean output video artifact for a succeeded job",
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
            404: {"model": ErrorResponse, "description": "Job not found for tenant"},
            409: {"model": ErrorResponse, "description": "Job has not succeeded yet"},
            410: {"model": ErrorResponse, "description": "Artifact file missing or outside storage root"},
        },
    )
    def download_job_artifact(
        job_id: str,
        tenant_id: str = Depends(get_tenant_id),
        repository: JobRepository = Depends(get_repository),
    ) -> FileResponse:
        settings = get_settings()
        job = repository.get_job(job_id)
        if job is None or job.tenant_id != tenant_id:
            raise AppError("JOB_NOT_FOUND", "job not found", 404)
        if job.status != "succeeded" or not job.output_path:
            raise AppError("JOB_NOT_READY", "job artifact is not available", 409)
        artifact = Path(job.output_path)
        if not is_under_directory(artifact, settings.storage_root):
            raise AppError("ARTIFACT_MISSING", "job artifact missing", 410)
        if not artifact.exists() or not artifact.is_file():
            raise AppError("ARTIFACT_MISSING", "job artifact missing", 410)
        media_type = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }.get(artifact.suffix.lower(), "application/octet-stream")
        return FileResponse(path=artifact, filename=artifact.name, media_type=media_type)

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        response_model=CancelJobResponse,
        tags=["jobs"],
        operation_id="cancelJob",
        summary="Cancel a queued job before execution starts",
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
            404: {"model": ErrorResponse, "description": "Job not found for tenant"},
            409: {"model": ErrorResponse, "description": "Job is already running or finalized"},
        },
    )
    def cancel_job(
        job_id: str,
        tenant_id: str = Depends(get_tenant_id),
        repository: JobRepository = Depends(get_repository),
    ) -> CancelJobResponse:
        job = repository.cancel_job(job_id=job_id, tenant_id=tenant_id)
        if job is None:
            raise AppError("JOB_NOT_FOUND", "job not found", 404)
        if job.status != "canceled":
            raise AppError("JOB_NOT_CANCELABLE", f"job in status '{job.status}' cannot be canceled", 409)
        return CancelJobResponse(job_id=job.job_id, status=job.status, updated_at=job.updated_at)

    @app.get(
        "/v1/providers",
        response_model=ProviderListResponse,
        tags=["health"],
        operation_id="listProviders",
        summary="Probe installed/runnable status of all watermark removal providers",
        responses={401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"}},
    )
    def list_providers(_: str = Depends(get_tenant_id)) -> ProviderListResponse:
        providers = ProviderRuntime(get_settings()).probe_all()
        return ProviderListResponse(providers=providers)

    return app


def _submit_response(job: JobRecord) -> JobSubmitResponse:
    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        provider_requested=job.provider_requested,
        quality_mode=job.quality_mode,
        created_at=job.created_at,
    )


def _discard_unused_upload(file: UploadFile | None, path: Path, existing: JobRecord | None) -> None:
    if file is None or not path.exists():
        return
    if existing is not None and str(path) == existing.input_path:
        return
    try:
        path.unlink()
    except OSError:
        logger.warning("failed to remove unused upload path=%s", path)


def _enforce_video_duration(path: Path, settings) -> None:
    if settings.max_video_seconds <= 0:
        return
    duration = probe_video_duration_seconds(path)
    if duration is None:
        return
    if duration > settings.max_video_seconds:
        raise AppError(
            "VIDEO_TOO_LONG",
            f"video duration {duration:.1f}s exceeds limit of {settings.max_video_seconds}s",
            400,
        )
