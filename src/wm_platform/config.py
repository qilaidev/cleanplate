from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEV_API_KEY = "dev-secret-key"
PRODUCTION_ENVIRONMENTS = {"prod", "production"}
QUALITY_MODE_VALUES = frozenset({"fast", "balanced", "quality", "corner_hq"})
NO_WATERMARK_POLICIES = frozenset({"fail", "passthrough"})
QA_RESIDUAL_POLICIES = frozenset({"flag", "fail"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    environment: str
    repo_root: Path
    storage_root: Path
    inbox_dir: Path
    outbox_dir: Path
    db_path: Path
    runtime_root: Path
    default_tenant_id: str
    default_api_key: str
    max_upload_bytes: int
    worker_poll_interval_seconds: float
    job_claim_timeout_seconds: int
    job_claim_heartbeat_seconds: float
    callback_retry_count: int
    callback_retry_delay_seconds: float
    allow_private_callback_urls: bool
    submit_rate_limit_count: int
    submit_rate_limit_window_seconds: float
    provider_probe_cache_seconds: float
    provider_runtime_delay_seconds: float
    comfyui_api_url: str
    auto_start_comfyui: bool
    comfyui_dir: Path
    comfyui_venv_dir: Path
    comfyui_custom_nodes_dir: Path
    comfyui_models_dir: Path
    comfyui_workflows_dir: Path
    comfyui_diffueraser_workflow: Path
    local_fallback_mode: str
    local_fallback_delogo_x: int | None
    local_fallback_delogo_y: int | None
    local_fallback_delogo_w: int | None
    local_fallback_delogo_h: int | None
    file_retention_days: int
    quality_mode: str
    max_video_seconds: int
    job_max_attempts: int
    auto_allow_copy_fallback: bool
    watermark_detect_prompt: str
    detector_mode: str
    comfyui_minimax_workflow: Path
    comfyui_sam3_workflow: Path
    comfyui_minimax_roi_workflow: Path
    minimax_roi_enabled: bool
    minimax_roi_min_side: int
    minimax_roi_max_frame_fraction: float
    work_dir: Path
    keep_work_files: bool
    chunk_frames: int
    comfyui_prompt_timeout_seconds: float
    no_watermark_policy: str
    qa_enabled: bool
    qa_residual_policy: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_choice(name: str, default: str, allowed: frozenset[str]) -> str:
    raw = (os.getenv(name) or default).strip().lower()
    if raw not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {raw!r}")
    return raw


def load_settings() -> Settings:
    repo_root = _repo_root()
    storage_root = Path(os.getenv("DWM_STORAGE_ROOT", repo_root / "storage")).expanduser().resolve()
    inbox_dir = Path(os.getenv("DWM_INBOX_DIR", storage_root / "inbox")).expanduser().resolve()
    outbox_dir = Path(os.getenv("DWM_OUTBOX_DIR", storage_root / "outbox")).expanduser().resolve()
    db_path = Path(os.getenv("DWM_DB_PATH", storage_root / "app.db")).expanduser().resolve()
    runtime_root = Path(os.getenv("DWM_RUNTIME_ROOT", repo_root / ".runtime")).expanduser().resolve()
    comfyui_dir = Path(
        os.getenv("DWM_COMFYUI_DIR", runtime_root / "ComfyUI")
    ).expanduser().resolve()
    comfyui_venv_dir = Path(os.getenv("DWM_COMFYUI_VENV_DIR", runtime_root / ".venv")).expanduser().resolve()
    comfyui_custom_nodes_dir = Path(
        os.getenv("DWM_COMFYUI_CUSTOM_NODES_DIR", comfyui_dir / "custom_nodes")
    ).expanduser().resolve()
    comfyui_models_dir = Path(os.getenv("DWM_COMFYUI_MODELS_DIR", comfyui_dir / "models")).expanduser().resolve()
    comfyui_workflows_dir = Path(
        os.getenv("DWM_COMFYUI_WORKFLOWS_DIR", repo_root / "workflows")
    ).expanduser().resolve()
    comfyui_diffueraser_workflow = Path(
        os.getenv("DWM_COMFYUI_DIFFUERASER_WORKFLOW", comfyui_workflows_dir / "diffueraser_mask_api.json")
    ).expanduser().resolve()
    comfyui_minimax_workflow = Path(
        os.getenv("DWM_COMFYUI_MINIMAX_WORKFLOW", comfyui_workflows_dir / "minimax_remover_api.json")
    ).expanduser().resolve()
    comfyui_sam3_workflow = Path(
        os.getenv("DWM_COMFYUI_SAM3_WORKFLOW", comfyui_workflows_dir / "sam3_native_api.json")
    ).expanduser().resolve()
    comfyui_minimax_roi_workflow = Path(
        os.getenv("DWM_COMFYUI_MINIMAX_ROI_WORKFLOW", comfyui_workflows_dir / "minimax_remover_roi_api.json")
    ).expanduser().resolve()
    delogo_x = os.getenv("DWM_LOCAL_FALLBACK_DELOGO_X")
    delogo_y = os.getenv("DWM_LOCAL_FALLBACK_DELOGO_Y")
    delogo_w = os.getenv("DWM_LOCAL_FALLBACK_DELOGO_W")
    delogo_h = os.getenv("DWM_LOCAL_FALLBACK_DELOGO_H")

    return Settings(
        environment=os.getenv("DWM_ENV", "development").strip().lower() or "development",
        repo_root=repo_root,
        storage_root=storage_root,
        inbox_dir=inbox_dir,
        outbox_dir=outbox_dir,
        db_path=db_path,
        runtime_root=runtime_root,
        default_tenant_id=os.getenv("DWM_DEFAULT_TENANT_ID", "local-dev"),
        default_api_key=os.getenv("DWM_DEFAULT_API_KEY", DEFAULT_DEV_API_KEY),
        max_upload_bytes=int(os.getenv("DWM_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))),
        worker_poll_interval_seconds=float(os.getenv("DWM_WORKER_POLL_INTERVAL_SECONDS", "1.0")),
        job_claim_timeout_seconds=int(os.getenv("DWM_JOB_CLAIM_TIMEOUT_SECONDS", "300")),
        job_claim_heartbeat_seconds=float(os.getenv("DWM_JOB_CLAIM_HEARTBEAT_SECONDS", "30.0")),
        callback_retry_count=int(os.getenv("DWM_CALLBACK_RETRY_COUNT", "3")),
        callback_retry_delay_seconds=float(os.getenv("DWM_CALLBACK_RETRY_DELAY_SECONDS", "1.0")),
        allow_private_callback_urls=_env_bool("DWM_ALLOW_PRIVATE_CALLBACK_URLS", False),
        submit_rate_limit_count=int(os.getenv("DWM_SUBMIT_RATE_LIMIT_COUNT", "60")),
        submit_rate_limit_window_seconds=float(os.getenv("DWM_SUBMIT_RATE_LIMIT_WINDOW_SECONDS", "60.0")),
        provider_probe_cache_seconds=float(os.getenv("DWM_PROVIDER_PROBE_CACHE_SECONDS", "10.0")),
        provider_runtime_delay_seconds=float(os.getenv("DWM_PROVIDER_RUNTIME_DELAY_SECONDS", "0.2")),
        comfyui_api_url=os.getenv("DWM_COMFYUI_API_URL", "http://127.0.0.1:8188"),
        auto_start_comfyui=_env_bool("DWM_AUTO_START_COMFYUI", False),
        comfyui_dir=comfyui_dir,
        comfyui_venv_dir=comfyui_venv_dir,
        comfyui_custom_nodes_dir=comfyui_custom_nodes_dir,
        comfyui_models_dir=comfyui_models_dir,
        comfyui_workflows_dir=comfyui_workflows_dir,
        comfyui_diffueraser_workflow=comfyui_diffueraser_workflow,
        local_fallback_mode=os.getenv("DWM_LOCAL_FALLBACK_MODE", "ffmpeg_copy"),
        local_fallback_delogo_x=int(delogo_x) if delogo_x is not None else None,
        local_fallback_delogo_y=int(delogo_y) if delogo_y is not None else None,
        local_fallback_delogo_w=int(delogo_w) if delogo_w is not None else None,
        local_fallback_delogo_h=int(delogo_h) if delogo_h is not None else None,
        file_retention_days=int(os.getenv("DWM_FILE_RETENTION_DAYS", "7")),
        quality_mode=os.getenv("DWM_QUALITY_MODE", "balanced"),
        max_video_seconds=int(os.getenv("DWM_MAX_VIDEO_SECONDS", "180")),
        job_max_attempts=int(os.getenv("DWM_JOB_MAX_ATTEMPTS", "2")),
        auto_allow_copy_fallback=_env_bool("DWM_AUTO_ALLOW_COPY_FALLBACK", False),
        watermark_detect_prompt=os.getenv(
            "DWM_WATERMARK_DETECT_PROMPT",
            "watermark, logo, on-screen text overlay, corner badge, platform stamp",
        ),
        detector_mode=os.getenv("DWM_DETECTOR_MODE", "auto").strip().lower() or "auto",
        comfyui_minimax_workflow=comfyui_minimax_workflow,
        comfyui_sam3_workflow=comfyui_sam3_workflow,
        comfyui_minimax_roi_workflow=comfyui_minimax_roi_workflow,
        minimax_roi_enabled=_env_bool("DWM_MINIMAX_ROI_ENABLED", True),
        minimax_roi_min_side=int(os.getenv("DWM_MINIMAX_ROI_MIN_SIDE", "256")),
        minimax_roi_max_frame_fraction=float(os.getenv("DWM_MINIMAX_ROI_MAX_FRAME_FRACTION", "0.6")),
        work_dir=Path(os.getenv("DWM_WORK_DIR", storage_root / "work")).expanduser().resolve(),
        keep_work_files=_env_bool("DWM_KEEP_WORK_FILES", False),
        chunk_frames=int(os.getenv("DWM_CHUNK_FRAMES", "81")),
        comfyui_prompt_timeout_seconds=float(os.getenv("DWM_COMFYUI_PROMPT_TIMEOUT_SECONDS", "1800")),
        no_watermark_policy=_env_choice("DWM_NO_WATERMARK_POLICY", "fail", NO_WATERMARK_POLICIES),
        qa_enabled=_env_bool("DWM_QA_ENABLED", True),
        qa_residual_policy=_env_choice("DWM_QA_RESIDUAL_POLICY", "flag", QA_RESIDUAL_POLICIES),
    )
