from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin

import httpx

from wm_platform.comfy_graph import expand_roi_chain, set_frame_window
from wm_platform.comfy_jobs import ComfyJobClient
from wm_platform.comfy_runtime import comfy_http_get, open_comfy_http_client
from wm_platform.config import QUALITY_MODE_VALUES, Settings
from wm_platform.errors import AppError
from wm_platform.media import (
    Chunk,
    NormalizedInput,
    assemble_output,
    extract_frame_window,
    normalize_input,
    plan_chunks,
)
from wm_platform.models import JobRecord, ProviderProbeResult, RunMetadataRecord
from wm_platform.pipeline import AI_INPAINT_PROVIDERS, PROBE_ORDER, auto_inpaint_chain
from wm_platform.quality_gate import check_output
from wm_platform.repository import JobRepository
from wm_platform.runtime_contract import (
    expected_model_entries,
    expected_repo_paths,
    model_destination,
    model_is_present,
)
from wm_platform.storage import build_output_path
from wm_platform.watermark_detect import DetectionResult, detect_watermark, sam3_detector_status

_PROBE_CACHE: dict[tuple[str, ...], tuple[float, list[ProviderProbeResult]]] = {}
DEFAULT_QUALITY_PROFILE = "balanced"
MASK_WORKFLOW_NODE_TYPES = ("DiffuEraser_PreData", "Propainter_Sampler", "DiffuEraser_Sampler", "ImageToMask")
# DiffuEraser_PreData requires a seg_repo string even though it is bypassed when video_mask is
# connected. We never run that segmentation: the mask always comes from the external detector.
DIFFUERASER_PREDATA_SEG_REPO = "briaai/RMBG-2.0"
MINIMAX_REQUIRED_NODE_TYPES = ("MinimaxModelLoader", "MinimaxVideoRemover", "ImageToMask")
MASK_WORKFLOW_NAME = "diffueraser_mask_api.json"
logger = logging.getLogger(__name__)

QUALITY_PROFILES = {
    "fast": {
        "steps": 2,
        "subvideo_length": 50,
        "neighbor_length": 10,
        "mask_dilation_iter": 1,
        "ref_stride": 10,
        "minimax_steps": 6,
        "minimax_mask_dilation": 1,
        "composite_expand": 4,
        "composite_feather": 3,
    },
    "balanced": {
        "steps": 5,
        "subvideo_length": 70,
        "neighbor_length": 14,
        "mask_dilation_iter": 2,
        "ref_stride": 10,
        "minimax_steps": 8,
        "minimax_mask_dilation": 2,
        "composite_expand": 6,
        "composite_feather": 5,
    },
    "quality": {
        "steps": 7,
        "subvideo_length": 100,
        "neighbor_length": 20,
        "mask_dilation_iter": 2,
        "ref_stride": 10,
        "minimax_steps": 12,
        "minimax_mask_dilation": 2,
        "composite_expand": 8,
        "composite_feather": 7,
    },
    "corner_hq": {
        "steps": 7,
        "subvideo_length": 100,
        "neighbor_length": 20,
        "mask_dilation_iter": 1,
        "ref_stride": 10,
        "minimax_steps": 12,
        "minimax_mask_dilation": 1,
        "composite_expand": 6,
        "composite_feather": 5,
    },
}

MINIMAX_ROI_ALIGN = 16
MINIMAX_ROI_PAD_FRACTION = 0.35
MINIMAX_ROI_PAD_MIN = 32
MINIMAX_ROI_REQUIRED_NODE_TYPES = ("ImageCrop", "CropMask", "ImageCompositeMasked")
MINIMAX_MAX_ROIS = 4
PASSTHROUGH_PROVIDER = "passthrough"


class ProviderExecutionError(Exception):
    def __init__(
        self,
        provider_selected: str | None,
        error_code: str,
        error_message: str,
        report: dict[str, object] | None = None,
    ) -> None:
        super().__init__(error_message)
        self.provider_selected = provider_selected
        self.error_code = error_code
        self.error_message = error_message
        self.report = report


class RunOutcome(NamedTuple):
    provider: str
    output_path: str
    report: dict[str, object]


@dataclass
class PipelineContext:
    """Everything decided once per job and shared by every provider attempt."""

    job: JobRecord
    work_dir: Path
    source: NormalizedInput
    detection: DetectionResult
    chunks: list[Chunk]
    report: dict[str, object] = field(default_factory=dict)

    @property
    def input_path(self) -> Path:
        return self.source.path


def _cleanup_job_spool_files(settings: Settings, job_id: str) -> None:
    if not job_id:
        return
    for root in (settings.comfyui_output_dir, settings.outbox_dir):
        for sub in ("video", "mask"):
            subdir = root / sub
            if not subdir.is_dir():
                continue
            for leftover in subdir.glob(f"{job_id}-*"):
                if leftover.is_file():
                    try:
                        leftover.unlink()
                    except OSError:
                        pass


def build_pipeline_context(
    settings: Settings,
    job: JobRecord,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineContext:
    """Normalize the input and run detection once. Raises AppError when there is nothing to inpaint."""
    work_dir = settings.work_dir / job.job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback("normalize_input")
    source = normalize_input(Path(job.input_path), work_dir, timeout=settings.media_timeout_seconds)
    report: dict[str, object] = {
        "input": {**source.source_info.as_report(), "normalized": source.normalized, "normalize_reasons": source.reasons},
    }
    if progress_callback is not None:
        progress_callback("detect_watermark")
    max_static_frames = settings.chunk_frames if settings.chunk_frames > 0 else None
    detection = detect_watermark(
        source.path,
        work_dir / "mask.mkv",
        settings,
        max_static_frames=max_static_frames,
    )
    report["detection"] = detection.as_report()
    chunks = plan_chunks(source.info.frame_count, settings.chunk_frames)
    context = PipelineContext(
        job=job, work_dir=work_dir, source=source, detection=detection, chunks=chunks, report=report
    )
    if detection.found:
        return context
    if detection.nothing_found:
        raise _DetectionOutcome(
            "NO_WATERMARK_DETECTED", detection.message or "no watermark found", context
        )
    raise _DetectionOutcome(
        "DETECTOR_LOW_CONFIDENCE",
        detection.message or "automatic watermark detection did not find a stable overlay",
        context,
    )


class _DetectionOutcome(AppError):
    def __init__(self, error_code: str, error_message: str, context: PipelineContext) -> None:
        super().__init__(error_code, error_message, 422)
        self.context = context


def run_comfy_chunks(
    settings: Settings,
    context: PipelineContext,
    label: str,
    build_prompt: Callable[[str], dict[str, dict[str, object]]],
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Path, str]:
    """Queue one prompt per frame window, then stitch the windows and mux the source audio.

    `build_prompt(device)` returns a fresh prompt; the window and a unique output prefix
    are injected here. Windows whose temporal mask is completely empty bypass ComfyUI and
    are sliced losslessly from the normalized input. Returns (assembled video, device).
    """
    backend = ComfyJobClient(settings)
    segments: list[tuple[Path, Chunk]] = []
    segment_dir = context.work_dir / f"segments-{label}"
    chunks = context.chunks or [Chunk(start=0, count=0)]
    static_mask_video = (
        str(context.detection.mask_video_path)
        if (context.detection.mask_video_path is not None and not context.detection.temporal)
        else None
    )
    device: str | None = None
    bypassed_chunks = 0
    inpainted_chunks = 0
    try:
        with open_comfy_http_client(httpx.Client, settings.comfyui_api_url, timeout=None) as client:
            for index, chunk in enumerate(chunks):
                if progress_callback is not None:
                    progress_callback(f"{label}:chunk:{index}/{len(chunks)}")
                if (
                    chunk.count > 0
                    and context.detection.temporal
                    and not context.detection.has_active_window(chunk.start, chunk.count)
                ):
                    segment = extract_frame_window(
                        context.source.path,
                        segment_dir / f"{index:04d}.mp4",
                        start=chunk.start,
                        count=chunk.count,
                        fps=context.source.info.fps,
                        color_info=context.source.source_info,
                        timeout=settings.media_timeout_seconds,
                    )
                    segments.append((segment, chunk))
                    bypassed_chunks += 1
                    continue

                if device is None:
                    device = backend.ensure_ready(client)
                prompt = build_prompt(device)
                if chunk.count > 0:
                    set_frame_window(
                        prompt,
                        chunk.start,
                        chunk.count,
                        static_mask_video=static_mask_video,
                    )
                prefix = f"video/{context.job.job_id}-{label}-{index:04d}"
                for node in prompt.values():
                    inputs = node.get("inputs") if isinstance(node, dict) else None
                    if isinstance(inputs, dict) and "filename_prefix" in inputs:
                        inputs["filename_prefix"] = prefix
                prompt_id = f"{context.job.job_id}-{label}-{index:04d}-{uuid.uuid4().hex[:6]}"
                queued = backend.queue_prompt(client, prompt, prompt_id)
                artifact = backend.wait_for_artifact(client, queued)
                suffix = Path(artifact.get("filename", "")).suffix or ".mp4"
                segment = backend.download_to(client, artifact, segment_dir / f"{index:04d}{suffix}")
                segments.append((segment, chunk))
                inpainted_chunks += 1
    finally:
        _cleanup_job_spool_files(settings, context.job.job_id)

    plan_meta = context.report.get("plan")
    if isinstance(plan_meta, dict):
        plan_meta["inpainted_chunks"] = inpainted_chunks
        plan_meta["bypassed_chunks"] = bypassed_chunks
    else:
        context.report["plan"] = {
            "chunks": len(chunks),
            "chunk_frames": settings.chunk_frames,
            "inpainted_chunks": inpainted_chunks,
            "bypassed_chunks": bypassed_chunks,
        }

    if progress_callback is not None:
        progress_callback(f"{label}:assemble_output")
    audio = context.source.path if context.source.info.has_audio else None
    output = assemble_output(
        segments,
        audio,
        context.work_dir / f"{label}.mp4",
        fps=context.source.info.fps,
        color_info=context.source.source_info,
        timeout=settings.media_timeout_seconds,
    )
    return output, device or "cpu"


@dataclass
class _BaseProvider:
    name: str
    settings: Settings

    def probe(self) -> ProviderProbeResult:
        raise NotImplementedError

    def run(self, job: JobRecord) -> dict[str, str]:
        raise NotImplementedError



class _ComfyDiffuEraserProvider(_BaseProvider):
    def __init__(self, name: str, settings: Settings, repository: JobRepository | None = None) -> None:
        super().__init__(name=name, settings=settings)
        self.repository = repository

    def probe(self) -> ProviderProbeResult:
        profile = self._quality_profile_name()
        workflow_path = self._workflow_path()
        installation_issues = self._missing_installation_bits()
        workflow_issue = self._workflow_issue()
        workflow_ready = workflow_issue is None
        missing_models = self._missing_models()
        api_issue = self._api_issue()
        reasons = [*installation_issues]
        if workflow_issue:
            reasons.append(workflow_issue)
        if missing_models:
            reasons.append(f"missing models: {', '.join(missing_models[:3])}")
        if api_issue:
            reasons.append(api_issue)
        details = {
            "api_url": self.settings.comfyui_api_url,
            "auto_start_comfyui": self.settings.auto_start_comfyui,
            "comfyui_dir": str(self.settings.comfyui_dir),
            "venv_python": str(self.settings.comfyui_venv_dir / "bin" / "python"),
            "custom_nodes_dir": str(self.settings.comfyui_custom_nodes_dir),
            "models_dir": str(self.settings.comfyui_models_dir),
            "workflow_path": str(workflow_path),
            "workflow_name": workflow_path.name,
            "quality_profile": profile,
            "runtime_root": str(self.settings.runtime_root),
            "missing_installation_bits": installation_issues,
            "missing_models": missing_models,
            "workflow_ready": workflow_ready,
            "workflow_issue": workflow_issue,
            "api_issue": api_issue,
            "automatic_ai_pipeline": "detect_then_inpaint",
            "detector": self.settings.detector_mode,
            "detector_note": "inpaints from the external static/SAM3 detector mask (DiffuEraser_PreData.video_mask)",
            "role": "secondary_inpainter",
        }
        return ProviderProbeResult(
            name=self.name,
            installed=not installation_issues,
            runnable=not reasons,
            message="ready" if not reasons else "; ".join(reasons[:3]),
            details=details,
        )

    def run(
        self,
        job: JobRecord,
        context: PipelineContext | None = None,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        if context is None:
            context = _context_or_raise(self.settings, job, self._record_detection_failure)
        detection = context.detection
        mask_video_path = detection.mask_video_path
        assert mask_video_path is not None
        base_prompt = self._build_prompt(job, mask_video_path=mask_video_path, input_path=context.input_path)

        def _prompt(device: str) -> dict[str, dict[str, object]]:
            return self._inject_prompt_runtime_values(
                copy.deepcopy(base_prompt), job, device, mask_video_path=mask_video_path, input_path=context.input_path
            )

        output_path, device = run_comfy_chunks(
            self.settings,
            context,
            self.name,
            _prompt,
            progress_callback=progress_callback,
        )
        self._record_run_metadata(
            job,
            device,
            scene_type=detection.scene_type,
            confidence_level=detection.confidence_level,
            mask_ratio=detection.mask_ratio,
        )
        return {"output_path": str(output_path), "workflow_name": self._workflow_path().name}

    def _record_detection_failure(self, job: JobRecord, detection: DetectionResult) -> None:
        self._record_run_metadata(
            job,
            None,
            scene_type=detection.scene_type,
            confidence_level=detection.confidence_level,
            mask_ratio=detection.mask_ratio,
        )

    def _record_run_metadata(
        self,
        job: JobRecord,
        device: str | None,
        scene_type: str | None = None,
        confidence_level: str | None = None,
        mask_ratio: float | None = None,
    ) -> None:
        if self.repository is None:
            return
        try:
            profile, profile_config = self._quality_profile(job)
            metadata = RunMetadataRecord(
                id=0,
                job_id=job.job_id,
                workflow_name=self._workflow_path().name,
                quality_profile=profile,
                steps=profile_config["steps"],
                subvideo_length=profile_config["subvideo_length"],
                neighbor_length=profile_config["neighbor_length"],
                mask_dilation_iter=profile_config["mask_dilation_iter"],
                device=device,
                seed=self._job_seed(job.job_id),
                scene_type=scene_type,
                confidence_level=confidence_level,
                mask_ratio=mask_ratio,
                created_at=datetime.now(UTC).replace(microsecond=0),
            )
            self.repository.record_run_metadata(metadata)
        except Exception:
            logger.exception("failed to record run metadata job_id=%s", job.job_id)

    def _missing_installation_bits(self) -> list[str]:
        repo_paths = expected_repo_paths(self.settings)
        reasons: list[str] = []
        if not repo_paths:
            reasons.append("runtime lock file missing or empty")
        missing_paths = [path for path in repo_paths if not path.exists()]
        if missing_paths:
            labels = [path.name or str(path) for path in missing_paths[:3]]
            reasons.append(f"missing runtime paths: {', '.join(labels)}")
        if not (self.settings.comfyui_venv_dir / "bin" / "python").exists():
            reasons.append("missing ComfyUI venv")
        return reasons

    def _missing_models(self) -> list[str]:
        entries = expected_model_entries(self.settings, provider=self.name)
        if not entries:
            return ["model manifest missing or empty"]
        missing: list[str] = []
        for entry in entries:
            if not bool(entry.get("required", False)):
                continue
            expected_path = model_destination(self.settings, entry)
            if not model_is_present(entry, expected_path):
                missing.append(str(entry.get("name", expected_path.name)))
        return missing

    def _workflow_issue(self) -> str | None:
        workflow_path = self._workflow_path()
        required = MASK_WORKFLOW_NODE_TYPES
        if not workflow_path.exists():
            return f"missing workflow {workflow_path.name}"
        try:
            prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return f"invalid workflow json: {exc.msg}"
        if not isinstance(prompt, dict) or not prompt:
            return "workflow prompt is empty"
        if any(not isinstance(node, dict) or "class_type" not in node for node in prompt.values()):
            return "workflow prompt must use ComfyUI API export format"
        missing_nodes = [
            node_type for node_type in required if self._find_node_inputs(prompt, node_type) is None
        ]
        if missing_nodes:
            return f"workflow missing required nodes: {', '.join(missing_nodes)}"
        predata = self._find_node_inputs(prompt, "DiffuEraser_PreData")
        if predata is None or "video_mask" not in predata:
            return "workflow missing DiffuEraser_PreData.video_mask"
        for node_type in ("Propainter_Sampler", "DiffuEraser_Sampler"):
            inputs = self._find_node_inputs(prompt, node_type)
            if inputs is None or "conditioning" not in inputs:
                return f"workflow missing {node_type}.conditioning"
        return None

    def _api_issue(self) -> str | None:
        try:
            response = comfy_http_get(httpx.get, urljoin(self.settings.comfyui_api_url, "/system_stats"), timeout=2.0)
        except Exception:
            return "ComfyUI API unreachable"
        if response.status_code != 200:
            return f"ComfyUI API returned {response.status_code}"
        return None

    def _build_prompt(
        self,
        job: JobRecord,
        mask_video_path: Path,
        input_path: Path | None = None,
    ) -> dict[str, dict[str, object]]:
        workflow_issue = self._workflow_issue()
        if workflow_issue:
            raise AppError("PROVIDER_NOT_AVAILABLE", workflow_issue, 503)
        _profile, profile_config = self._quality_profile(job)
        workflow_path = self._workflow_path()
        template = json.loads(workflow_path.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            raise AppError("PROVIDER_NOT_AVAILABLE", "workflow prompt is invalid", 503)

        replacements: dict[str, object] = {
            "__INPUT_VIDEO__": str(input_path or job.input_path),
            "__SEG_REPO__": DIFFUERASER_PREDATA_SEG_REPO,
            "__OUTPUT_PREFIX__": f"video/{job.job_id}",
            "__MASK_VIDEO__": str(mask_video_path),
            "__VAE_MODEL__": self._resolve_model_input(
                roots=[self.settings.comfyui_models_dir / "vae"],
                candidates=["sd-vae-ft-mse.safetensors"],
                label="vae",
            ),
            "__LORA_MODEL__": self._resolve_model_input(
                roots=[self.settings.comfyui_models_dir / "loras"],
                candidates=["sd15/pcm_sd15_smallcfg_2step_converted.safetensors", "pcm_sd15_smallcfg_2step_converted.safetensors"],
                label="lora",
            ),
            "__CLIP_MODEL__": self._resolve_model_input(
                roots=[self.settings.comfyui_models_dir / "text_encoders", self.settings.comfyui_models_dir / "clip"],
                candidates=["clip_l.safetensors"],
                label="clip",
            ),
            "__PROPAINTER_MODEL__": self._resolve_model_input(
                roots=[self.settings.comfyui_models_dir / "DiffuEraser"],
                candidates=["propainter/ProPainter.pth", "ProPainter.pth"],
                label="propainter",
            ),
            "__FLOW_MODEL__": self._resolve_model_input(
                roots=[self.settings.comfyui_models_dir / "DiffuEraser"],
                candidates=["propainter/recurrent_flow_completion.pth", "recurrent_flow_completion.pth"],
                label="flow",
            ),
            "__FIX_RAFT_MODEL__": self._resolve_model_input(
                roots=[self.settings.comfyui_models_dir / "DiffuEraser"],
                candidates=["propainter/raft-things.pth", "raft-things.pth"],
                label="fix_raft",
            ),
            "__SEED__": self._job_seed(job.job_id),
        }
        prompt = self._replace_placeholders(copy.deepcopy(template), replacements)
        prompt = self._apply_quality_profile(prompt, profile_config)
        return prompt

    def _apply_quality_profile(
        self, prompt: dict[str, dict[str, object]], profile_config: dict[str, int]
    ) -> dict[str, dict[str, object]]:
        propainter_inputs = self._find_node_inputs(prompt, "Propainter_Sampler")
        if propainter_inputs is None:
            raise AppError("PROVIDER_NOT_AVAILABLE", "workflow missing required Propainter_Sampler node", 503)
        if "mask_dilation_iter" in propainter_inputs:
            propainter_inputs["mask_dilation_iter"] = profile_config.get(
                "mask_dilation_iter",
                propainter_inputs["mask_dilation_iter"],
            )
        if "ref_stride" in propainter_inputs:
            propainter_inputs["ref_stride"] = profile_config.get("ref_stride", propainter_inputs.get("ref_stride", 10))
        if "neighbor_length" in propainter_inputs:
            propainter_inputs["neighbor_length"] = profile_config.get("neighbor_length", propainter_inputs["neighbor_length"])
        if "subvideo_length" in propainter_inputs:
            propainter_inputs["subvideo_length"] = profile_config.get("subvideo_length", propainter_inputs["subvideo_length"])

        diffueraser_inputs = self._find_node_inputs(prompt, "DiffuEraser_Sampler")
        if diffueraser_inputs is None:
            raise AppError("PROVIDER_NOT_AVAILABLE", "workflow missing required DiffuEraser_Sampler node", 503)
        if "steps" in diffueraser_inputs:
            diffueraser_inputs["steps"] = profile_config.get("steps", diffueraser_inputs["steps"])

        return prompt

    @staticmethod
    def _job_seed(job_id: str) -> int:
        return int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16)

    def _quality_profile_name(self, job: JobRecord | None = None) -> str:
        return resolve_quality_profile(self.settings, job)

    def _quality_profile(self, job: JobRecord | None = None) -> tuple[str, dict[str, int]]:
        profile = self._quality_profile_name(job)
        return profile, QUALITY_PROFILES[profile]

    def _workflow_path(self) -> Path:
        """The external-mask DiffuEraser workflow. Quality profiles only inject parameters."""
        configured = self.settings.comfyui_diffueraser_workflow
        if configured.exists():
            return configured
        candidate = self.settings.comfyui_workflows_dir / MASK_WORKFLOW_NAME
        if candidate.exists():
            return candidate.resolve()
        return configured

    @staticmethod
    def _find_node_inputs(prompt: dict[str, dict[str, object]], class_type: str) -> dict[str, object] | None:
        for node in prompt.values():
            if not isinstance(node, dict) or node.get("class_type") != class_type:
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                return inputs
        return None

    def _inject_prompt_runtime_values(
        self,
        prompt: dict[str, dict[str, object]],
        job: JobRecord,
        device: str,
        mask_video_path: Path,
        input_path: Path | None = None,
    ) -> dict[str, dict[str, object]]:
        replacements: dict[str, object] = {
            "__PROP_DEVICE__": device,
            "__INPUT_VIDEO__": str(input_path or job.input_path),
            "__OUTPUT_PREFIX__": f"video/{job.job_id}",
            "__SEG_REPO__": DIFFUERASER_PREDATA_SEG_REPO,
            "__MASK_VIDEO__": str(mask_video_path),
        }
        return self._replace_placeholders(prompt, replacements)

    def _resolve_model_input(self, roots: list[Path], candidates: list[str], label: str) -> str:
        for root in roots:
            if not root.exists():
                continue
            for candidate in candidates:
                direct_path = root / candidate
                if direct_path.exists():
                    return direct_path.relative_to(root).as_posix()
            for candidate in candidates:
                candidate_name = Path(candidate).name
                matches = sorted(path for path in root.rglob(candidate_name) if path.is_file())
                if matches:
                    return matches[0].relative_to(root).as_posix()
        raise AppError("PROVIDER_NOT_AVAILABLE", f"missing required {label} model input", 503)

    def _replace_placeholders(self, payload: object, replacements: dict[str, object]) -> object:
        if isinstance(payload, str):
            return replacements.get(payload, payload)
        if isinstance(payload, list):
            return [self._replace_placeholders(item, replacements) for item in payload]
        if isinstance(payload, dict):
            return {key: self._replace_placeholders(value, replacements) for key, value in payload.items()}
        return payload

class _MiniMaxRemoverProvider(_BaseProvider):
    def __init__(self, name: str, settings: Settings, repository: JobRepository | None = None) -> None:
        super().__init__(name=name, settings=settings)
        self.repository = repository

    def probe(self) -> ProviderProbeResult:
        installation_issues = self._missing_installation_bits()
        workflow_issue = self._workflow_issue()
        missing_models = self._missing_models()
        api_issue = self._api_issue()
        reasons = [*installation_issues]
        if workflow_issue:
            reasons.append(workflow_issue)
        if missing_models:
            reasons.append(f"missing models: {', '.join(missing_models[:3])}")
        if api_issue:
            reasons.append(api_issue)
        sam3_status = sam3_detector_status(self.settings)
        roi_workflow_issue = self._roi_workflow_issue()
        details = {
            "api_url": self.settings.comfyui_api_url,
            "workflow_path": str(self.settings.comfyui_minimax_workflow),
            "workflow_name": self.settings.comfyui_minimax_workflow.name,
            "roi_enabled": bool(self.settings.minimax_roi_enabled),
            "roi_workflow_name": self.settings.comfyui_minimax_roi_workflow.name,
            "roi_workflow_ready": roi_workflow_issue is None,
            "roi_workflow_issue": roi_workflow_issue,
            "quality_profile": self._quality_profile_name(),
            "detector": self.settings.detector_mode,
            "watermark_prompt": self.settings.watermark_detect_prompt,
            "missing_installation_bits": installation_issues,
            "missing_models": missing_models,
            "workflow_ready": workflow_issue is None,
            "workflow_issue": workflow_issue,
            "api_issue": api_issue,
            "automatic_ai_pipeline": "detect_then_inpaint",
            "role": "primary_inpainter",
            "sam3_available": bool(sam3_status.get("available")),
            "sam3_issues": sam3_status.get("issues") or [],
        }
        return ProviderProbeResult(
            name=self.name,
            installed=not installation_issues,
            runnable=not reasons,
            message="ready" if not reasons else "; ".join(reasons[:3]),
            details=details,
        )

    def run(
        self,
        job: JobRecord,
        context: PipelineContext | None = None,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        if context is None:
            context = _context_or_raise(self.settings, job, self._record_detection_failure)
        detection = context.detection
        assert detection.mask_video_path is not None
        rois = self.plan_rois(detection)
        prompt, workflow_name = self._build_prompt(
            job, detection.mask_video_path, rois=rois, input_path=context.input_path
        )
        output_path, device = run_comfy_chunks(
            self.settings,
            context,
            self.name,
            lambda _device: copy.deepcopy(prompt),
            progress_callback=progress_callback,
        )
        self._record_run_metadata(
            job,
            device,
            detection.scene_type,
            detection.confidence_level,
            detection.mask_ratio,
            workflow_name=workflow_name,
        )
        result = {"output_path": str(output_path), "workflow_name": workflow_name}
        if rois:
            result["roi"] = ";".join(",".join(str(value) for value in roi) for roi in rois)
        return result

    def _record_detection_failure(self, job: JobRecord, detection: DetectionResult) -> None:
        self._record_run_metadata(
            job, None, detection.scene_type, detection.confidence_level, detection.mask_ratio
        )

    def plan_rois(self, detection: DetectionResult) -> list[tuple[int, int, int, int]] | None:
        """Crop windows around the detected overlays, or None for full-frame inpainting.

        Cropping gives the remover a higher effective resolution on the logo and keeps
        the rest of the frame out of the VAE. Separate overlays (a logo top-left and a
        user id bottom-right is the common case) get separate windows; windows that
        touch are merged. Full frame is used when the windows would cover most of it.
        """
        if not self.settings.minimax_roi_enabled:
            return None
        if detection.bbox is None or detection.frame_size is None:
            return None
        if self._roi_workflow_issue() is not None:
            return None
        return plan_minimax_rois(
            list(detection.regions) if detection.regions and not detection.temporal else [detection.bbox],
            detection.frame_size,
            min_side=self.settings.minimax_roi_min_side,
            max_frame_fraction=self.settings.minimax_roi_max_frame_fraction,
        )

    def _missing_installation_bits(self) -> list[str]:
        reasons: list[str] = []
        node_dir = self.settings.comfyui_custom_nodes_dir / "ComfyUI-MiniMax-Remover"
        if not node_dir.exists():
            reasons.append("missing ComfyUI-MiniMax-Remover custom node")
        if not (self.settings.comfyui_venv_dir / "bin" / "python").exists():
            reasons.append("missing ComfyUI venv")
        return reasons

    def _missing_models(self) -> list[str]:
        root = self.settings.comfyui_models_dir / "MiniMax-Remover"
        missing: list[str] = []
        for name in ("vae", "transformer", "scheduler"):
            if not (root / name).exists():
                missing.append(f"MiniMax-Remover/{name}")
        return missing

    def _workflow_issue(self) -> str | None:
        workflow_path = self.settings.comfyui_minimax_workflow
        if not workflow_path.exists():
            return f"missing workflow {workflow_path.name}"
        try:
            prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return f"invalid workflow json: {exc.msg}"
        if not isinstance(prompt, dict) or not prompt:
            return "workflow prompt is empty"
        if any(not isinstance(node, dict) or "class_type" not in node for node in prompt.values()):
            return "workflow prompt must use ComfyUI API export format"
        missing_nodes = [
            node_type
            for node_type in MINIMAX_REQUIRED_NODE_TYPES
            if _find_node_inputs(prompt, node_type) is None
        ]
        if missing_nodes:
            return f"workflow missing required nodes: {', '.join(missing_nodes)}"
        return None

    def _api_issue(self) -> str | None:
        try:
            response = comfy_http_get(httpx.get, urljoin(self.settings.comfyui_api_url, "/system_stats"), timeout=2.0)
        except Exception:
            return "ComfyUI API unreachable"
        if response.status_code != 200:
            return f"ComfyUI API returned {response.status_code}"
        return None

    def _roi_workflow_issue(self) -> str | None:
        workflow_path = self.settings.comfyui_minimax_roi_workflow
        if not workflow_path.exists():
            return f"missing workflow {workflow_path.name}"
        try:
            prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return f"invalid workflow json: {exc.msg}"
        if not isinstance(prompt, dict) or not prompt:
            return "workflow prompt is empty"
        if any(not isinstance(node, dict) or "class_type" not in node for node in prompt.values()):
            return "workflow prompt must use ComfyUI API export format"
        missing_nodes = [
            node_type
            for node_type in (*MINIMAX_REQUIRED_NODE_TYPES, *MINIMAX_ROI_REQUIRED_NODE_TYPES)
            if _find_node_inputs(prompt, node_type) is None
        ]
        if missing_nodes:
            return f"workflow missing required nodes: {', '.join(missing_nodes)}"
        return None

    def _build_prompt(
        self,
        job: JobRecord,
        mask_video_path: Path,
        *,
        rois: list[tuple[int, int, int, int]] | None = None,
        input_path: Path | None = None,
    ) -> tuple[dict[str, dict[str, object]], str]:
        workflow_issue = self._workflow_issue()
        if workflow_issue:
            raise AppError("PROVIDER_NOT_AVAILABLE", workflow_issue, 503)
        profile, profile_config = self._quality_profile(job)
        workflow_path = self.settings.comfyui_minimax_workflow
        if rois:
            workflow_path = self.settings.comfyui_minimax_roi_workflow
        template = json.loads(workflow_path.read_text(encoding="utf-8"))
        if rois:
            template = expand_roi_chain(
                template,
                [
                    {"__ROI_X__": x, "__ROI_Y__": y, "__ROI_W__": w, "__ROI_H__": h}
                    for x, y, w, h in rois
                ],
            )
        replacements: dict[str, object] = {
            "__INPUT_VIDEO__": str(input_path or job.input_path),
            "__MASK_VIDEO__": str(mask_video_path),
            "__OUTPUT_PREFIX__": f"video/{job.job_id}",
            "__SEED__": _job_seed(job.job_id),
            "__WATERMARK_PROMPT__": self.settings.watermark_detect_prompt,
            "__COMPOSITE_EXPAND__": profile_config.get("composite_expand", 6),
            "__COMPOSITE_FEATHER__": profile_config.get("composite_feather", 5),
        }
        prompt = _replace_placeholders(copy.deepcopy(template), replacements)
        removers = [
            node["inputs"]
            for node in prompt.values()
            if isinstance(node, dict) and node.get("class_type") == "MinimaxVideoRemover"
        ]
        if not removers:
            raise AppError("PROVIDER_NOT_AVAILABLE", "workflow missing required MinimaxVideoRemover node", 503)
        for remover_inputs in removers:
            if "num_inference_steps" in remover_inputs:
                remover_inputs["num_inference_steps"] = profile_config.get(
                    "minimax_steps", remover_inputs["num_inference_steps"]
                )
            if "mask_dilation_iterations" in remover_inputs:
                remover_inputs["mask_dilation_iterations"] = profile_config.get(
                    "minimax_mask_dilation", remover_inputs["mask_dilation_iterations"]
                )
        return prompt, workflow_path.name

    def _quality_profile_name(self, job: JobRecord | None = None) -> str:
        return resolve_quality_profile(self.settings, job)

    def _quality_profile(self, job: JobRecord | None = None) -> tuple[str, dict[str, int]]:
        profile = self._quality_profile_name(job)
        return profile, QUALITY_PROFILES[profile]

    def _record_run_metadata(
        self,
        job: JobRecord,
        device: str | None,
        scene_type: str | None,
        confidence_level: str | None,
        mask_ratio: float | None = None,
        *,
        workflow_name: str | None = None,
    ) -> None:
        if self.repository is None:
            return
        try:
            profile, profile_config = self._quality_profile(job)
            metadata = RunMetadataRecord(
                id=0,
                job_id=job.job_id,
                workflow_name=workflow_name or self.settings.comfyui_minimax_workflow.name,
                quality_profile=profile,
                steps=profile_config.get("minimax_steps"),
                subvideo_length=None,
                neighbor_length=None,
                mask_dilation_iter=profile_config.get("minimax_mask_dilation"),
                device=device,
                seed=_job_seed(job.job_id),
                scene_type=scene_type,
                confidence_level=confidence_level,
                mask_ratio=mask_ratio,
                created_at=datetime.now(UTC).replace(microsecond=0),
            )
            self.repository.record_run_metadata(metadata)
        except Exception:
            logger.exception("failed to record run metadata job_id=%s", job.job_id)


def plan_minimax_roi(
    bbox: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    *,
    min_side: int = 256,
    max_frame_fraction: float = 0.6,
    align: int = MINIMAX_ROI_ALIGN,
) -> tuple[int, int, int, int] | None:
    frame_w, frame_h = frame_size
    if frame_w <= 0 or frame_h <= 0:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    pad = max(MINIMAX_ROI_PAD_MIN, int(round(max(w, h) * MINIMAX_ROI_PAD_FRACTION)))
    x0, roi_w = _roi_axis(x - pad, x + w + pad, frame_w, min_side=min_side, align=align)
    y0, roi_h = _roi_axis(y - pad, y + h + pad, frame_h, min_side=min_side, align=align)
    if x0 is None or y0 is None:
        # Frame smaller than an aligned window: full-frame path handles it.
        return None
    if roi_w * roi_h >= max_frame_fraction * frame_w * frame_h:
        return None
    if x0 > x or y0 > y or x0 + roi_w < x + w or y0 + roi_h < y + h:
        return None
    return (int(x0), int(y0), int(roi_w), int(roi_h))


def plan_minimax_rois(
    boxes: list[tuple[int, int, int, int]],
    frame_size: tuple[int, int],
    *,
    min_side: int = 256,
    max_frame_fraction: float = 0.6,
    max_rois: int = MINIMAX_MAX_ROIS,
) -> list[tuple[int, int, int, int]] | None:
    """One window per overlay group; groups whose windows intersect are merged."""
    groups = [box for box in boxes if box[2] > 0 and box[3] > 0]
    if not groups:
        return None

    def window(box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        return plan_minimax_roi(
            box, frame_size, min_side=min_side, max_frame_fraction=max_frame_fraction
        )

    merged = True
    while merged:
        merged = False
        windows = [window(box) for box in groups]
        if any(item is None for item in windows):
            break
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _intersects(windows[i], windows[j]):  # type: ignore[arg-type]
                    groups[i] = _union_box(groups[i], groups[j])
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    windows = [window(box) for box in groups]
    if len(groups) > max_rois or any(item is None for item in windows):
        union = groups[0]
        for box in groups[1:]:
            union = _union_box(union, box)
        single = window(union)
        return [single] if single is not None else None
    frame_area = frame_size[0] * frame_size[1]
    if sum(w * h for _x, _y, w, h in windows) >= max_frame_fraction * frame_area:  # type: ignore[misc]
        return None
    return sorted(windows)  # type: ignore[arg-type]


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _context_or_raise(
    settings: Settings,
    job: JobRecord,
    on_detection_failure: Callable[[JobRecord, DetectionResult], None],
) -> PipelineContext:
    try:
        return build_pipeline_context(settings, job)
    except _DetectionOutcome as exc:
        on_detection_failure(job, exc.context.detection)
        raise


def _roi_axis(lo: int, hi: int, frame: int, *, min_side: int, align: int) -> tuple[int | None, int]:
    """One axis of the ROI: pad -> minimum side (centred) -> align -> keep inside the frame."""
    if hi - lo < min_side:
        extra = min_side - (hi - lo)
        lo -= extra // 2
        hi = lo + min_side
    lo = max(0, lo)
    hi = max(hi, lo + min_side)
    start = (lo // align) * align
    length = -(-(hi - start) // align) * align
    if start + length > frame:
        start = max(0, ((frame - length) // align) * align)
        length = min(length, frame - start)
    if length % align or length <= 0:
        return None, 0
    return start, length


def _find_node_inputs(prompt: dict[str, dict[str, object]], class_type: str) -> dict[str, object] | None:
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            return inputs
    return None


def _replace_placeholders(payload: object, replacements: dict[str, object]) -> object:
    if isinstance(payload, str):
        return replacements.get(payload, payload)
    if isinstance(payload, list):
        return [_replace_placeholders(item, replacements) for item in payload]
    if isinstance(payload, dict):
        return {key: _replace_placeholders(value, replacements) for key, value in payload.items()}
    return payload


def _job_seed(job_id: str) -> int:
    return int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16)


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_quality_profile(settings: Settings, job: JobRecord | None = None) -> str:
    raw = None
    if job is not None and job.quality_mode:
        raw = job.quality_mode
    if not raw:
        raw = settings.quality_mode
    profile = (raw or DEFAULT_QUALITY_PROFILE).strip().lower()
    if profile in QUALITY_PROFILES and profile in QUALITY_MODE_VALUES:
        return profile
    logger.warning("unknown quality profile=%s, fallback to %s", profile, DEFAULT_QUALITY_PROFILE)
    return DEFAULT_QUALITY_PROFILE


class _LocalFallbackProvider(_BaseProvider):
    def probe(self) -> ProviderProbeResult:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return ProviderProbeResult(
                name=self.name,
                installed=False,
                runnable=False,
                message="ffmpeg not found in PATH",
                details={"mode": self.settings.local_fallback_mode, "ffmpeg_path": None},
            )

        mode = self.settings.local_fallback_mode.strip().lower()
        if mode == "ffmpeg_copy":
            return ProviderProbeResult(
                name=self.name,
                installed=True,
                runnable=True,
                message="ffmpeg copy mode",
                details={"mode": "ffmpeg_copy", "ffmpeg_path": ffmpeg_path},
            )

        if mode == "delogo":
            if not self._has_complete_delogo_config():
                return ProviderProbeResult(
                    name=self.name,
                    installed=True,
                    runnable=False,
                    message="delogo mode requires x/y/w/h coordinates",
                    details={"mode": "delogo", "ffmpeg_path": ffmpeg_path},
                )
            if not self._ffmpeg_supports_delogo():
                return ProviderProbeResult(
                    name=self.name,
                    installed=True,
                    runnable=False,
                    message="ffmpeg delogo filter not available",
                    details={"mode": "delogo", "ffmpeg_path": ffmpeg_path},
                )
            return ProviderProbeResult(
                name=self.name,
                installed=True,
                runnable=True,
                message="ffmpeg delogo mode",
                details={
                    "mode": "delogo",
                    "ffmpeg_path": ffmpeg_path,
                    "x": self.settings.local_fallback_delogo_x,
                    "y": self.settings.local_fallback_delogo_y,
                    "w": self.settings.local_fallback_delogo_w,
                    "h": self.settings.local_fallback_delogo_h,
                },
            )

        return ProviderProbeResult(
            name=self.name,
            installed=True,
            runnable=False,
            message=f"unsupported local_fallback_mode: {self.settings.local_fallback_mode}",
            details={"mode": self.settings.local_fallback_mode, "ffmpeg_path": ffmpeg_path},
        )

    def run(self, job: JobRecord, context: PipelineContext | None = None) -> dict[str, str]:
        source_path = job.input_path
        output_path = build_output_path(job.job_id, source_path, self.settings)
        mode = self.settings.local_fallback_mode.strip().lower()
        if mode == "ffmpeg_copy":
            shutil.copyfile(source_path, output_path)
            return {"output_path": str(output_path)}

        command = self._build_ffmpeg_command(Path(source_path), output_path)
        timeout = self.settings.media_timeout_seconds if self.settings.media_timeout_seconds > 0 else None
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppError(
                "PROVIDER_RUN_FAILED",
                f"local_fallback ffmpeg timed out after {timeout:.1f}s",
                504,
            ) from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            detail = stderr.splitlines()[-1] if stderr else "ffmpeg failed"
            raise AppError("PROVIDER_RUN_FAILED", f"local_fallback ffmpeg error: {detail}", 500)
        return {"output_path": str(output_path)}

    def _build_ffmpeg_command(self, source_path: Path, output_path: Path) -> list[str]:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise AppError("PROVIDER_NOT_AVAILABLE", "ffmpeg is not available in PATH", 503)

        mode = self.settings.local_fallback_mode.strip().lower()
        base_command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        if mode == "ffmpeg_copy":
            return [
                *base_command,
                "-c",
                "copy",
                str(output_path),
            ]

        if mode == "delogo":
            if not self._has_complete_delogo_config():
                raise AppError("PROVIDER_NOT_AVAILABLE", "delogo mode requires x/y/w/h coordinates", 503)
            if not self._ffmpeg_supports_delogo():
                raise AppError("PROVIDER_NOT_AVAILABLE", "ffmpeg delogo filter not available", 503)
            vf = (
                f"delogo=x={self.settings.local_fallback_delogo_x}:"
                f"y={self.settings.local_fallback_delogo_y}:"
                f"w={self.settings.local_fallback_delogo_w}:"
                f"h={self.settings.local_fallback_delogo_h}"
            )
            return [
                *base_command,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "copy",
                str(output_path),
            ]

        raise AppError(
            "PROVIDER_NOT_AVAILABLE",
            f"unsupported local_fallback_mode: {self.settings.local_fallback_mode}",
            503,
        )

    def _has_complete_delogo_config(self) -> bool:
        return all(
            value is not None
            for value in (
                self.settings.local_fallback_delogo_x,
                self.settings.local_fallback_delogo_y,
                self.settings.local_fallback_delogo_w,
                self.settings.local_fallback_delogo_h,
            )
        )

    @staticmethod
    @lru_cache(maxsize=4)
    def _ffmpeg_supports_delogo() -> bool:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return False
        completed = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return False
        return " delogo " in completed.stdout


class ProviderRuntime:
    def __init__(self, settings: Settings, repository: JobRepository | None = None) -> None:
        self.settings = settings
        self.repository = repository
        self.registry: dict[str, _BaseProvider] = {
            "minimax_remover": _MiniMaxRemoverProvider(
                name="minimax_remover", settings=settings, repository=repository
            ),
            "comfy_diffueraser": _ComfyDiffuEraserProvider(
                name="comfy_diffueraser", settings=settings, repository=repository
            ),
            "local_fallback": _LocalFallbackProvider(name="local_fallback", settings=settings),
        }

    def probe_all(self) -> list[ProviderProbeResult]:
        cache_key = (
            str(self.settings.runtime_root),
            self.settings.comfyui_api_url,
            self.settings.local_fallback_mode,
            f"{self.settings.quality_mode}:{self.settings.comfyui_diffueraser_workflow}:{self.settings.comfyui_minimax_workflow}:{self.settings.comfyui_minimax_roi_workflow}:{self.settings.comfyui_sam3_workflow}:{self.settings.detector_mode}",
            self.settings.detector_mode,
            str(self.settings.auto_allow_copy_fallback),
        )
        now = time.monotonic()
        cached = _PROBE_CACHE.get(cache_key)
        if cached and (now - cached[0]) < max(0.0, self.settings.provider_probe_cache_seconds):
            return cached[1]

        probes = [self.registry[name].probe() for name in PROBE_ORDER]
        _PROBE_CACHE[cache_key] = (now, probes)
        return probes

    def run_with_fallback(
        self,
        job: JobRecord,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RunOutcome:
        """Probe -> normalize + detect once -> inpaint -> quality gate, per provider in the chain.

        A provider whose output fails the gate (unchanged frames, overlay still visible)
        does not end the job: the next provider in the chain gets a turn. When every
        output fails only the residual check, the first one is returned flagged
        `needs_review` (or the job fails, per DWM_QA_RESIDUAL_POLICY).
        """
        chain = self._resolve_fallback_chain(job)
        report: dict[str, object] = {"chain": chain, "attempts": []}
        attempts: list[dict[str, object]] = report["attempts"]  # type: ignore[assignment]
        runnable: list[str] = []
        failed_messages: list[str] = []
        last_provider: str | None = None
        for provider_name in chain:
            provider = self.registry.get(provider_name)
            if provider is None:
                failed_messages.append(f"{provider_name}: not registered")
                attempts.append({"provider": provider_name, "status": "not_registered"})
                continue
            last_provider = provider_name
            probe = provider.probe()
            if not probe.runnable:
                failed_messages.append(f"{provider_name}: {probe.message or 'not runnable'}")
                attempts.append({"provider": provider_name, "status": "not_runnable", "message": probe.message})
                continue
            runnable.append(provider_name)
        if not runnable:
            raise ProviderExecutionError(
                provider_selected=last_provider,
                error_code="PROVIDER_NOT_AVAILABLE",
                error_message="all providers unavailable",
                report=report,
            )

        context: PipelineContext | None = None
        try:
            if any(name in AI_INPAINT_PROVIDERS for name in runnable):
                context, early = self._prepare_context(
                    job, runnable[0], report, progress_callback=progress_callback
                )
                if early is not None:
                    return early
            return self._run_providers(
                job,
                runnable,
                context,
                report,
                failed_messages,
                progress_callback=progress_callback,
            )
        finally:
            _cleanup_job_spool_files(self.settings, job.job_id)
            if not self.settings.keep_work_files:
                shutil.rmtree(self.settings.work_dir / job.job_id, ignore_errors=True)

    def _prepare_context(
        self,
        job: JobRecord,
        first_provider: str,
        report: dict[str, object],
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[PipelineContext | None, RunOutcome | None]:
        try:
            context = build_pipeline_context(self.settings, job, progress_callback=progress_callback)
        except _DetectionOutcome as exc:
            report.update(exc.context.report)
            provider = self.registry.get(first_provider)
            recorder = getattr(provider, "_record_detection_failure", None)
            if callable(recorder):
                recorder(job, exc.context.detection)
            if exc.error_code == "NO_WATERMARK_DETECTED" and self.settings.no_watermark_policy == "passthrough":
                output = build_output_path(job.job_id, job.input_path, self.settings)
                shutil.copyfile(job.input_path, output)
                report["action"] = "passthrough"
                report["needs_review"] = False
                return None, RunOutcome(PASSTHROUGH_PROVIDER, str(output), report)
            raise ProviderExecutionError(
                provider_selected=first_provider,
                error_code=exc.error_code,
                error_message=exc.error_message,
                report=report,
            ) from exc
        except AppError as exc:
            raise ProviderExecutionError(
                provider_selected=first_provider,
                error_code=exc.error_code,
                error_message=exc.error_message,
                report=report,
            ) from exc
        report.update(context.report)
        report["plan"] = {"chunks": len(context.chunks), "chunk_frames": self.settings.chunk_frames}
        context.report["plan"] = report["plan"]
        return context, None

    def _run_providers(
        self,
        job: JobRecord,
        runnable: list[str],
        context: PipelineContext | None,
        report: dict[str, object],
        failed_messages: list[str],
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RunOutcome:
        attempts: list[dict[str, object]] = report["attempts"]  # type: ignore[assignment]
        flagged: tuple[str, Path, dict[str, object]] | None = None
        for provider_name in runnable:
            provider = self.registry[provider_name]
            attempt: dict[str, object] = {"provider": provider_name}
            attempts.append(attempt)
            if progress_callback is not None:
                progress_callback(f"provider:{provider_name}")
            try:
                if provider_name in AI_INPAINT_PROVIDERS:
                    try:
                        result = provider.run(job, context, progress_callback=progress_callback)  # type: ignore[call-arg]
                    except TypeError:
                        result = provider.run(job, context)  # type: ignore[call-arg]
                else:
                    result = provider.run(job)
                output_path = result.get("output_path")
                if not output_path:
                    raise AppError("PROVIDER_RUN_FAILED", f"{provider_name} returned empty output_path")
                attempt.update({key: value for key, value in result.items() if key != "output_path"})
            except AppError as exc:
                attempt.update({"status": "failed", "error_code": exc.error_code, "message": exc.error_message})
                failed_messages.append(f"{provider_name}: {exc.error_message}")
                continue
            except Exception as exc:  # noqa: BLE001 - one provider crashing must not end the chain
                logger.exception("provider crashed job_id=%s provider=%s", job.job_id, provider_name)
                attempt.update({"status": "failed", "error_code": "PROVIDER_RUN_FAILED", "message": str(exc)})
                failed_messages.append(f"{provider_name}: {exc}")
                continue

            if context is None or provider_name not in AI_INPAINT_PROVIDERS or not self.settings.qa_enabled:
                attempt["status"] = "succeeded"
                return self._finish(job, provider_name, Path(output_path), report, qa=None)
            if progress_callback is not None:
                progress_callback(f"quality_gate:{provider_name}")
            verdict = check_output(context.input_path, Path(output_path), context.detection)
            attempt["qa"] = verdict.as_report()
            if verdict.unchanged:
                attempt["status"] = "failed"
                attempt["error_code"] = "INPAINT_NO_EFFECT"
                failed_messages.append(f"{provider_name}: output identical to input inside the watermark mask")
                continue
            if verdict.residual:
                attempt["status"] = "residual_suspected"
                failed_messages.append(f"{provider_name}: overlay still detected after inpainting")
                if flagged is None:
                    flagged = (provider_name, Path(output_path), verdict.as_report())
                continue
            attempt["status"] = "succeeded"
            return self._finish(job, provider_name, Path(output_path), report, qa=verdict.as_report())

        if flagged is not None and self.settings.qa_residual_policy != "fail":
            provider_name, output_path, qa = flagged
            return self._finish(job, provider_name, output_path, report, qa=qa)
        failure_codes = {item.get("error_code") for item in attempts if item.get("status") == "failed"}
        if flagged is not None:
            error_code = "QA_RESIDUAL_WATERMARK"
        elif failure_codes == {"INPAINT_NO_EFFECT"}:
            error_code = "INPAINT_NO_EFFECT"
        else:
            error_code = "PROVIDER_RUN_FAILED"
        detail = "; ".join(failed_messages[-3:])
        raise ProviderExecutionError(
            provider_selected=runnable[-1],
            error_code=error_code,
            error_message=f"all providers failed: {detail}",
            report=report,
        )

    def _finish(
        self,
        job: JobRecord,
        provider_name: str,
        output_path: Path,
        report: dict[str, object],
        *,
        qa: dict[str, object] | None,
    ) -> RunOutcome:
        final = output_path
        if is_inside(output_path, self.settings.work_dir):
            final = build_output_path(job.job_id, str(output_path), self.settings)
            shutil.move(str(output_path), final)
        report["action"] = "copied" if provider_name == "local_fallback" else "inpainted"
        report["qa"] = qa or {"status": "skipped"}
        report["needs_review"] = bool(qa and qa.get("status") == "residual_suspected")
        return RunOutcome(provider_name, str(final), report)

    def _resolve_fallback_chain(self, job: JobRecord) -> list[str]:
        if job.provider_requested == "auto":
            return auto_inpaint_chain(self.settings)
        try:
            parsed = json.loads(job.fallback_chain_json)
            if isinstance(parsed, list) and parsed:
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [job.provider_requested]
