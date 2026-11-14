from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from wm_platform.comfy_jobs import ComfyJobClient
from wm_platform.config import Settings
from wm_platform.errors import AppError

DETECTOR_MODES = frozenset({"static_overlay", "sam3_text", "auto"})
# SAM 3.1 runs on ComfyUI's native nodes (comfy_extras/nodes_sam3.py, merged 2026-04-23).
SAM3_WORKFLOW_NAME = "sam3_native_api.json"
SAM3_NATIVE_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
SAM3_NATIVE_NODES_FILE = Path("comfy_extras") / "nodes_sam3.py"
SAM3_NATIVE_REQUIRED_NODE_TYPES = ("CheckpointLoaderSimple", "CLIPTextEncode")
SAM3_NATIVE_MASK_NODE_TYPES = frozenset({"SAM3_VideoTrack", "SAM3_Detect"})
SAM3_SAVE_NODE_TYPES = frozenset({"SaveVideo", "SaveImage", "VHS_VideoCombine"})
SAM3_LOADER_NODE_TYPES = frozenset({"VHS_LoadVideoPath", "VHS_LoadVideo"})

# Cap full-clip SAM3 tracking tensors in VRAM; long clips are strided across time and capped to 960p.
SAM3_MAX_SAMPLED_FRAMES = 180
SAM3_MAX_SIDE = 960

# Temporal masks: a frame-level mask is treated as "static" when every frame overlaps the
# union this much. Below it the overlay is considered moving and per-frame masks are kept.
STATIC_MASK_IOU = 0.8
TEMPORAL_PAD_FRAMES = 1
IMAGE_MASK_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


@dataclass(frozen=True)
class DetectionResult:
    scene_type: str
    confidence_level: str
    mask_ratio: float
    region_count: int
    mask_video_path: Path | None
    message: str
    bbox: tuple[int, int, int, int] | None = None
    frame_size: tuple[int, int] | None = None
    temporal: bool = False
    detector: str = "static_overlay"
    # Per-component boxes (frame pixels) so separate corners get separate ROIs.
    regions: tuple[tuple[int, int, int, int], ...] = ()
    # Union mask at frame resolution; used by the post-inpaint quality gate.
    union_mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    # Per-frame mask activity for temporal overlays so empty chunks can bypass GPU inpainting.
    active_frames: tuple[bool, ...] | None = field(default=None, compare=False, repr=False)

    @property
    def found(self) -> bool:
        return self.mask_video_path is not None and self.confidence_level in {"high", "medium"}

    @property
    def nothing_found(self) -> bool:
        """Every detector that ran came back empty (as opposed to unsure or oversized)."""
        return self.scene_type == "none"

    def has_active_window(self, start: int, count: int) -> bool:
        """Whether the mask is non-empty anywhere inside [start, start + count)."""
        if not self.temporal or self.active_frames is None or count <= 0:
            return True
        lo = max(0, int(start))
        hi = min(len(self.active_frames), lo + int(count))
        if lo >= hi:
            return True
        return any(self.active_frames[lo:hi])

    def as_report(self) -> dict[str, object]:
        return {
            "detector": self.detector,
            "scene_type": self.scene_type,
            "confidence": self.confidence_level,
            "mask_ratio": self.mask_ratio,
            "region_count": self.region_count,
            "bbox": list(self.bbox) if self.bbox else None,
            "regions": [list(region) for region in self.regions],
            "temporal": self.temporal,
            "message": self.message,
        }


def detect_static_overlay(
    source_path: Path,
    destination_mask_path: Path,
    *,
    max_frames: int = 16,
    corner_margin: float = 0.22,
    max_mask_ratio: float = 0.18,
    min_mask_ratio: float = 0.0004,
    max_static_frames: int | None = None,
) -> DetectionResult:
    frames = _sample_frames(source_path, max_frames=max_frames)
    if len(frames) < 2:
        return DetectionResult(
            scene_type="unknown",
            confidence_level="low",
            mask_ratio=0.0,
            region_count=0,
            mask_video_path=None,
            message="not enough frames to detect a static overlay",
        )

    candidate = _static_candidate(frames, corner_margin)
    return _finalize_detected_mask(
        source_path,
        destination_mask_path,
        candidate,
        max_mask_ratio=max_mask_ratio,
        min_mask_ratio=min_mask_ratio,
        detected_message="static overlay detected",
        empty_message="no stable logo-like overlay found",
        oversized_message="detected region is too large or fragmented for automatic watermark removal",
        scene_fallback="static_logo",
        corner_margin=corner_margin,
        detector="static_overlay",
        max_static_frames=max_static_frames,
    )


def _static_candidate(frames: list[np.ndarray], corner_margin: float) -> np.ndarray:
    count = float(len(frames))
    acc = np.zeros(frames[0].shape, dtype=np.float64)
    acc_sq = np.zeros(frames[0].shape, dtype=np.float64)
    for frame in frames:
        as_f64 = frame.astype(np.float64)
        acc += as_f64
        acc_sq += as_f64 * as_f64
    mean_f64 = acc / count
    mean = mean_f64.astype(np.float32)
    std = np.sqrt(np.maximum(0.0, (acc_sq / count) - (mean_f64 * mean_f64))).astype(np.float32)
    contrast_map = np.abs(cv2.Laplacian(mean.astype(np.uint8), cv2.CV_32F))
    # Path 1: opaque overlays. Pixels that never change but sit on a textured area.
    candidate = (std < 8.0) & (contrast_map > 12.0)
    # Path 2: translucent overlays over a moving background. The overlay's own edges
    # persist across frames while background edges drift.
    candidate |= _persistent_edge_candidate(frames)
    candidate = _keep_margin_bands(candidate, corner_margin)
    return _morphology(candidate)


def static_overlay_mask(
    source_path: Path, *, max_frames: int = 16, corner_margin: float = 0.22
) -> np.ndarray | None:
    """Raw static-overlay candidate mask at frame resolution (no size/shape policy)."""
    frames = _sample_frames(source_path, max_frames=max_frames)
    if len(frames) < 2:
        return None
    return _static_candidate(frames, corner_margin)


def sam3_workflow_path(settings: Settings) -> Path:
    """Configured SAM3 workflow, else the bundled native one."""
    configured = getattr(settings, "comfyui_sam3_workflow", None)
    if configured is not None:
        configured_path = Path(configured).expanduser().resolve()
        if configured_path.exists():
            return configured_path
    return (settings.comfyui_workflows_dir / SAM3_WORKFLOW_NAME).resolve()


def sam3_native_checkpoint_path(settings: Settings) -> Path:
    return settings.comfyui_models_dir / "checkpoints" / SAM3_NATIVE_CHECKPOINT


def sam3_detector_status(settings: Settings) -> dict[str, object]:
    issues: list[str] = []
    workflow = sam3_workflow_path(settings)
    if not workflow.exists():
        issues.append("SAM3 workflow missing")
    else:
        workflow_issue = _sam3_workflow_issue(workflow)
        if workflow_issue:
            issues.append(workflow_issue)
    native_nodes = settings.comfyui_dir / SAM3_NATIVE_NODES_FILE
    if not native_nodes.exists():
        issues.append(
            "SAM3 native nodes not installed (ComfyUI comfy_extras/nodes_sam3.py missing; update ComfyUI to >= v0.35)"
        )
    checkpoint = sam3_native_checkpoint_path(settings)
    if not checkpoint.is_file():
        issues.append(f"SAM3 checkpoint missing: models/checkpoints/{SAM3_NATIVE_CHECKPOINT}")
    return {
        "available": not issues,
        "issues": issues,
        "checkpoint_path": str(checkpoint),
        "workflow_path": str(workflow),
    }


def detect_watermark(
    source_path: Path,
    destination_mask_path: Path,
    settings: Settings,
    *,
    max_frames: int = 16,
    corner_margin: float = 0.22,
    max_mask_ratio: float = 0.18,
    min_mask_ratio: float = 0.0004,
    max_static_frames: int | None = None,
) -> DetectionResult:
    mode = (settings.detector_mode or "static_overlay").strip().lower()
    if mode not in DETECTOR_MODES:
        mode = "static_overlay"

    if mode == "static_overlay":
        return detect_static_overlay(
            source_path,
            destination_mask_path,
            max_frames=max_frames,
            corner_margin=corner_margin,
            max_mask_ratio=max_mask_ratio,
            min_mask_ratio=min_mask_ratio,
            max_static_frames=max_static_frames,
        )

    if mode == "sam3_text":
        return detect_sam3_text(
            source_path,
            destination_mask_path,
            settings,
            max_mask_ratio=max_mask_ratio,
            min_mask_ratio=min_mask_ratio,
            max_static_frames=max_static_frames,
        )

    static = detect_static_overlay(
        source_path,
        destination_mask_path,
        max_frames=max_frames,
        corner_margin=corner_margin,
        max_mask_ratio=max_mask_ratio,
        min_mask_ratio=min_mask_ratio,
        max_static_frames=max_static_frames,
    )
    if static.confidence_level in {"high", "medium"} and static.mask_video_path is not None:
        return static
    sam3 = detect_sam3_text(
        source_path,
        destination_mask_path,
        settings,
        max_mask_ratio=max_mask_ratio,
        min_mask_ratio=min_mask_ratio,
        max_static_frames=max_static_frames,
    )
    if sam3.found:
        return sam3
    message = static.message
    if sam3.message and sam3.message not in message:
        message = f"{static.message}; {sam3.message}"
    # "none" only when both detectors actually ran and both saw nothing. An unavailable
    # SAM3 cannot vouch for the centre of the frame or for moving overlays.
    both_empty = static.nothing_found and sam3.nothing_found
    return DetectionResult(
        scene_type="none" if both_empty else ("complex" if "complex" in {static.scene_type, sam3.scene_type} else "unknown"),
        confidence_level="low",
        mask_ratio=static.mask_ratio,
        region_count=static.region_count,
        mask_video_path=None,
        message=message,
        frame_size=static.frame_size,
        detector="auto",
    )


def detect_sam3_text(
    source_path: Path,
    destination_mask_path: Path,
    settings: Settings,
    *,
    max_mask_ratio: float = 0.18,
    min_mask_ratio: float = 0.0004,
    max_static_frames: int | None = None,
) -> DetectionResult:
    status = sam3_detector_status(settings)
    issues = [str(item) for item in status.get("issues", [])] if isinstance(status.get("issues"), list) else []
    if not bool(status.get("available")):
        return DetectionResult(
            scene_type="unknown",
            confidence_level="low",
            mask_ratio=0.0,
            region_count=0,
            mask_video_path=None,
            message="SAM3 detector is not available: " + "; ".join(issues) if issues else "SAM3 detector is not available",
            detector="sam3_text",
        )
    try:
        return _run_sam3_comfy(
            source_path,
            destination_mask_path,
            settings,
            max_mask_ratio=max_mask_ratio,
            min_mask_ratio=min_mask_ratio,
            max_static_frames=max_static_frames,
        )
    except Exception as exc:
        return DetectionResult(
            scene_type="unknown",
            confidence_level="low",
            mask_ratio=0.0,
            region_count=0,
            mask_video_path=None,
            message=f"SAM3 live detection failed: {exc}",
            detector="sam3_text",
        )


def _configure_sam3_loader_limits(
    prompt: dict[str, dict[str, object]],
    *,
    width: int,
    height: int,
    frame_count: int,
    max_sampled_frames: int = SAM3_MAX_SAMPLED_FRAMES,
    max_side: int = SAM3_MAX_SIDE,
) -> int:
    sample_stride = max(1, math.ceil(frame_count / max_sampled_frames)) if frame_count > 0 else 1
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") not in SAM3_LOADER_NODE_TYPES:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        existing_cap = int(inputs.get("frame_load_cap") or 0)
        inputs["frame_load_cap"] = (
            min(existing_cap, max_sampled_frames) if existing_cap > 0 else max_sampled_frames
        )
        inputs["select_every_nth"] = max(int(inputs.get("select_every_nth") or 1), sample_stride)
        if max(width, height) > max_side and not int(inputs.get("custom_width") or 0) and not int(inputs.get("custom_height") or 0):
            if width >= height:
                inputs["custom_width"] = max_side
                inputs["custom_height"] = 0
            else:
                inputs["custom_width"] = 0
                inputs["custom_height"] = max_side
    return sample_stride


def _cleanup_sam3_spool_files(settings: Settings, prefix_stem: str) -> None:
    spool_roots = [settings.comfyui_output_dir / "mask", settings.outbox_dir / "mask"]
    for root in spool_roots:
        if not root.is_dir():
            continue
        for candidate in root.glob(f"{prefix_stem}*"):
            if candidate.is_file():
                try:
                    candidate.unlink()
                except OSError:
                    pass


def _run_sam3_comfy(
    source_path: Path,
    destination_mask_path: Path,
    settings: Settings,
    *,
    max_mask_ratio: float,
    min_mask_ratio: float,
    max_static_frames: int | None = None,
) -> DetectionResult:
    workflow_path = sam3_workflow_path(settings)
    try:
        template = json.loads(workflow_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError("PROVIDER_NOT_AVAILABLE", f"SAM3 workflow unreadable: {exc}", 503) from exc
    if not isinstance(template, dict) or not template:
        raise AppError("PROVIDER_NOT_AVAILABLE", "SAM3 workflow prompt is empty", 503)

    fps, width, height, frame_count = _probe_source_dimensions(source_path)
    prompt_id = f"sam3-{uuid.uuid4().hex[:16]}"
    prefix_stem = f"{source_path.stem}-{prompt_id}"
    prompt = _replace_placeholders(
        template,
        {
            "__INPUT_VIDEO__": str(source_path),
            "__WATERMARK_PROMPT__": settings.watermark_detect_prompt,
            "__OUTPUT_PREFIX__": f"mask/{prefix_stem}",
            "__SAM3_CHECKPOINT__": SAM3_NATIVE_CHECKPOINT,
        },
    )
    assert isinstance(prompt, dict)
    sample_stride = _configure_sam3_loader_limits(
        prompt,
        width=width,
        height=height,
        frame_count=frame_count,
    )
    raw_path = destination_mask_path.with_name(f"{destination_mask_path.stem}.sam3-raw.mp4")
    try:
        downloaded = ComfyJobClient(settings).run_for_artifact(
            prompt,
            prompt_id,
            raw_path,
            allow_images=True,
        )
        return _finalize_streamed_mask(
            source_path,
            destination_mask_path,
            downloaded,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            sample_stride=sample_stride,
            max_mask_ratio=max_mask_ratio,
            min_mask_ratio=min_mask_ratio,
            max_static_frames=max_static_frames,
        )
    finally:
        for leftover in {raw_path, raw_path.with_suffix(".png"), raw_path.with_suffix(".jpg")}:
            if leftover.exists():
                try:
                    leftover.unlink()
                except OSError:
                    pass
        if "downloaded" in locals() and downloaded.exists() and downloaded != destination_mask_path:
            try:
                downloaded.unlink()
            except OSError:
                pass
        _cleanup_sam3_spool_files(settings, prefix_stem)


def _iter_raw_masks(path: Path, *, limit: int = 100_000) -> Iterator[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_MASK_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            yield _morphology(image > 127)
        return

    yielded = 0
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        try:
            while yielded < limit:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                yield _morphology(gray > 127)
                yielded += 1
        finally:
            capture.release()
    if yielded == 0:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            yield _morphology(image > 127)


def _iter_padded_masks(path: Path) -> Iterator[np.ndarray]:
    """Stream temporally padded (+-1 neighbour OR) masks using at most 3 frames in RAM."""
    stream = _iter_raw_masks(path)
    first = next(stream, None)
    if first is None:
        return
    second = next(stream, None)
    if second is None:
        yield first
        return
    if TEMPORAL_PAD_FRAMES <= 0:
        yield first
        yield second
        yield from stream
        return

    yield first | second
    prev = first
    curr = second
    for nxt in stream:
        yield prev | curr | nxt
        prev = curr
        curr = nxt
    yield prev | curr


def _finalize_streamed_mask(
    source_path: Path,
    destination_mask_path: Path,
    mask_media_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    sample_stride: int,
    max_mask_ratio: float,
    min_mask_ratio: float,
    corner_margin: float = 0.22,
    max_static_frames: int | None = None,
) -> DetectionResult:
    sampled_count = 0
    union: np.ndarray | None = None
    min_frame_sum: int | None = None
    max_ratio = 0.0
    max_regions = 0
    sampled_nonempty: list[bool] = []

    for padded in _iter_padded_masks(mask_media_path):
        sampled_count += 1
        union = padded.copy() if union is None else (union | padded)
        area = int(padded.sum())
        min_frame_sum = area if min_frame_sum is None else min(min_frame_sum, area)
        max_ratio = max(max_ratio, area / float(padded.size))
        max_regions = max(max_regions, _count_regions(padded))
        sampled_nonempty.append(area > 0)

    if sampled_count == 0 or union is None:
        return DetectionResult(
            scene_type="unknown",
            confidence_level="low",
            mask_ratio=0.0,
            region_count=0,
            mask_video_path=None,
            message="SAM3 produced no usable mask",
            detector="sam3_text",
        )

    mask_h, mask_w = union.shape[-2:]
    eff_width = width or mask_w
    eff_height = height or mask_h
    eff_frames = frame_count if frame_count > 0 else sampled_count
    frame_size = (eff_width, eff_height)

    union_area = float(union.sum())
    if sampled_count >= 2 and union_area > 0:
        moving = ((min_frame_sum or 0) / union_area) < STATIC_MASK_IOU
    else:
        moving = False

    if moving:
        mask_ratio = max_ratio
        region_count = max_regions
    else:
        mask_ratio = float(union.mean())
        region_count = _count_regions(union)

    if mask_ratio < min_mask_ratio or region_count == 0:
        return DetectionResult(
            scene_type="none",
            confidence_level="low",
            mask_ratio=mask_ratio,
            region_count=region_count,
            mask_video_path=None,
            message="SAM3 mask is empty",
            frame_size=frame_size,
            detector="sam3_text",
        )

    union_ratio = float(union.mean())
    if mask_ratio > max_mask_ratio or region_count > 6 or union_ratio > max(max_mask_ratio, 0.35):
        return DetectionResult(
            scene_type="complex",
            confidence_level="low",
            mask_ratio=mask_ratio,
            region_count=region_count,
            mask_video_path=None,
            message="detected region is too large or fragmented for automatic watermark removal",
            frame_size=frame_size,
            detector="sam3_text",
        )

    scene_type = "moving_overlay" if moving else ("corner" if _is_corner_dominant(union, corner_margin) else "sam3_text")
    confidence = "high" if mask_ratio < 0.05 and region_count <= 3 else "medium"
    destination_mask_path.parent.mkdir(parents=True, exist_ok=True)

    active_frames: tuple[bool, ...] | None = None
    if moving:
        active_frames = _write_streamed_temporal_mask_video(
            destination_mask_path,
            _iter_padded_masks(mask_media_path),
            sampled_count=sampled_count,
            sampled_nonempty=sampled_nonempty,
            sample_stride=max(1, sample_stride),
            fps=fps,
            width=eff_width,
            height=eff_height,
            frame_count=eff_frames,
        )
    else:
        static_frames = (
            min(max(1, eff_frames), max_static_frames)
            if (max_static_frames is not None and max_static_frames > 0)
            else max(1, eff_frames)
        )
        _write_mask_video(
            destination_mask_path,
            union,
            fps=fps,
            width=eff_width,
            height=eff_height,
            frame_count=static_frames,
        )

    return DetectionResult(
        scene_type=scene_type,
        confidence_level=confidence,
        mask_ratio=round(mask_ratio, 6),
        region_count=region_count,
        mask_video_path=destination_mask_path,
        message="SAM3 text mask detected",
        bbox=_mask_bbox(union, eff_width, eff_height),
        frame_size=frame_size,
        temporal=moving,
        detector="sam3_text",
        regions=_region_bboxes(union, eff_width, eff_height),
        union_mask=_resize_mask(union, eff_width, eff_height),
        active_frames=active_frames,
    )


def _finalize_detected_mask(
    source_path: Path,
    destination_mask_path: Path,
    mask: np.ndarray,
    *,
    max_mask_ratio: float,
    min_mask_ratio: float,
    detected_message: str,
    empty_message: str,
    oversized_message: str,
    scene_fallback: str,
    corner_margin: float = 0.22,
    detector: str = "static_overlay",
    max_static_frames: int | None = None,
) -> DetectionResult:
    fps, width, height, frame_count = _source_video_info(source_path, mask)
    frame_size = (width, height)
    per_frame: np.ndarray | None = None
    moving = False
    active_frames: tuple[bool, ...] | None = None
    if mask.ndim == 3:
        per_frame = np.stack([_morphology(frame) for frame in mask], axis=0)
        per_frame = _pad_temporal(per_frame, TEMPORAL_PAD_FRAMES)
        union = per_frame.any(axis=0)
        moving = not _is_static_sequence(per_frame, union)
        if not moving:
            per_frame = None
        active = union
        ratios = per_frame.reshape(per_frame.shape[0], -1).mean(axis=1) if per_frame is not None else None
        mask_ratio = float(ratios.max()) if ratios is not None else float(active.mean())
        region_count = (
            max(_count_regions(frame) for frame in per_frame) if per_frame is not None else _count_regions(active)
        )
    else:
        active = _morphology(mask)
        mask_ratio = float(active.mean())
        region_count = _count_regions(active)

    if mask_ratio < min_mask_ratio or region_count == 0:
        return DetectionResult(
            scene_type="none",
            confidence_level="low",
            mask_ratio=mask_ratio,
            region_count=region_count,
            mask_video_path=None,
            message=empty_message,
            frame_size=frame_size,
            detector=detector,
        )
    union_ratio = float(active.mean())
    if mask_ratio > max_mask_ratio or region_count > 6 or union_ratio > max(max_mask_ratio, 0.35):
        return DetectionResult(
            scene_type="complex",
            confidence_level="low",
            mask_ratio=mask_ratio,
            region_count=region_count,
            mask_video_path=None,
            message=oversized_message,
            frame_size=frame_size,
            detector=detector,
        )

    if moving:
        scene_type = "moving_overlay"
    else:
        scene_type = "corner" if _is_corner_dominant(active, corner_margin) else scene_fallback
    confidence = "high" if mask_ratio < 0.05 and region_count <= 3 else "medium"
    destination_mask_path.parent.mkdir(parents=True, exist_ok=True)
    target_frames = max(1, frame_count)
    if per_frame is None and max_static_frames is not None and max_static_frames > 0:
        target_frames = min(target_frames, max_static_frames)
    if per_frame is not None:
        total = max(1, frame_count)
        active_frames = tuple(bool(per_frame[min(index, per_frame.shape[0] - 1)].any()) for index in range(total))
    _write_mask_video(
        destination_mask_path,
        per_frame if per_frame is not None else active,
        fps=fps,
        width=width,
        height=height,
        frame_count=target_frames,
    )
    return DetectionResult(
        scene_type=scene_type,
        confidence_level=confidence,
        mask_ratio=round(mask_ratio, 6),
        region_count=region_count,
        mask_video_path=destination_mask_path,
        message=detected_message,
        bbox=_mask_bbox(active, width, height),
        frame_size=frame_size,
        temporal=per_frame is not None,
        detector=detector,
        regions=_region_bboxes(active, width, height),
        union_mask=_resize_mask(active, width, height),
        active_frames=active_frames,
    )


def _mask_frames_from_media(path: Path) -> np.ndarray | None:
    """Return a bool mask array: [H, W] for images, [N, H, W] for videos."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_MASK_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image > 127
    frames = _read_all_frames(path, limit=SAM3_MAX_SAMPLED_FRAMES * 4)
    if not frames:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image > 127
    stacked = np.stack(frames, axis=0) > 127
    if stacked.shape[0] == 1:
        return stacked[0]
    return stacked


def _replace_placeholders(payload: object, replacements: dict[str, object]) -> object:
    if isinstance(payload, str):
        return replacements.get(payload, payload)
    if isinstance(payload, list):
        return [_replace_placeholders(item, replacements) for item in payload]
    if isinstance(payload, dict):
        return {key: _replace_placeholders(value, replacements) for key, value in payload.items()}
    return payload


def _sam3_workflow_issue(workflow_path: Path) -> str | None:
    try:
        prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"invalid SAM3 workflow json: {exc.msg}"
    if not isinstance(prompt, dict) or not prompt:
        return "SAM3 workflow prompt is empty"
    class_types = {
        str(node.get("class_type"))
        for node in prompt.values()
        if isinstance(node, dict) and node.get("class_type")
    }
    missing = [name for name in SAM3_NATIVE_REQUIRED_NODE_TYPES if name not in class_types]
    if not class_types & SAM3_NATIVE_MASK_NODE_TYPES:
        missing.append("SAM3_VideoTrack|SAM3_Detect")
    if missing:
        return "SAM3 workflow missing required nodes: " + ", ".join(missing)
    if not (class_types & SAM3_SAVE_NODE_TYPES):
        return "SAM3 workflow missing Save/output node"
    return None


def _sample_frames(source_path: Path, max_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return _read_until(capture, max_frames)
        indexes = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
        frames: list[np.ndarray] = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return frames
    finally:
        capture.release()


def _read_until(capture: cv2.VideoCapture, max_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frames


def _read_all_frames(source_path: Path, limit: int = SAM3_MAX_SAMPLED_FRAMES * 4) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return []
    try:
        return _read_until(capture, limit)
    finally:
        capture.release()


def _persistent_edge_candidate(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    persistence: float = 0.8,
    canny_low: int = 60,
    canny_high: int = 140,
    close_kernel: int = 5,
) -> np.ndarray:
    """Outline-and-fill detector for translucent overlays over a moving background.

    Canny edges are computed per frame. A pixel that is an edge in almost every frame
    belongs to something that does not move: the overlay outline. Background edges
    drift and fail the persistence vote; smooth gradients never become edges at all.
    The persistent outline is closed and filled so the whole overlay is masked.
    """
    count = len(frames)
    if count < 3:
        shape = frames[0].shape[:2] if count > 0 else (0, 0)
        return np.zeros(shape, dtype=bool)
    frame_shape = frames[0].shape[:2]
    votes = np.zeros(frame_shape, dtype=np.int32)
    jitter = np.ones((3, 3), np.uint8)
    for frame in frames:
        u8 = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
        edges = cv2.Canny(u8, canny_low, canny_high)
        # Codec noise moves an edge by a pixel between frames; vote on a 1 px halo.
        votes += cv2.dilate(edges, jitter) > 0
    persistent = votes >= max(3, int(np.ceil(persistence * count)))
    if not persistent.any():
        return persistent
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    closed = cv2.morphologyEx(persistent.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    # The halo and the closing pushed the outline outwards by ~4 px; pull the fill back in.
    filled = cv2.erode(filled, np.ones((9, 9), np.uint8), iterations=1)
    return filled > 0


def _keep_margin_bands(mask: np.ndarray, margin: float) -> np.ndarray:
    height, width = mask.shape[:2]
    band_h = max(8, int(height * margin))
    band_w = max(8, int(width * margin))
    bands = np.zeros_like(mask, dtype=bool)
    bands[:band_h, :] = True
    bands[-band_h:, :] = True
    bands[:, :band_w] = True
    bands[:, -band_w:] = True
    return mask & bands


def _morphology(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed > 0


def _pad_temporal(frames: np.ndarray, radius: int) -> np.ndarray:
    """OR every frame mask with its +-radius neighbours so tracker drop-outs do not leave holes."""
    if radius <= 0 or frames.shape[0] < 2:
        return frames
    padded = frames.copy()
    for offset in range(1, radius + 1):
        padded[offset:] |= frames[:-offset]
        padded[:-offset] |= frames[offset:]
    return padded


def _is_static_sequence(frames: np.ndarray, union: np.ndarray) -> bool:
    union_area = float(union.sum())
    if union_area <= 0:
        return True
    for frame in frames:
        inter = float((frame & union).sum())
        if inter / union_area < STATIC_MASK_IOU:
            return False
    return True


def _count_regions(mask: np.ndarray) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return max(0, int(count) - 1) if stats is not None else 0


def _is_corner_dominant(mask: np.ndarray, margin: float) -> bool:
    height, width = mask.shape[:2]
    band_h = max(8, int(height * margin))
    band_w = max(8, int(width * margin))
    corners = np.zeros_like(mask, dtype=bool)
    corners[:band_h, :band_w] = True
    corners[:band_h, -band_w:] = True
    corners[-band_h:, :band_w] = True
    corners[-band_h:, -band_w:] = True
    corner_ratio = float((mask & corners).sum())
    total = float(mask.sum()) or 1.0
    return (corner_ratio / total) >= 0.7


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _region_bboxes(mask: np.ndarray, width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for label in range(1, int(count)):
        box = _mask_bbox(labels == label, width, height)
        if box is not None:
            boxes.append(box)
    return tuple(sorted(boxes))


def _mask_bbox(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    mask_h, mask_w = mask.shape[:2]
    scale_x = width / float(mask_w) if mask_w else 1.0
    scale_y = height / float(mask_h) if mask_h else 1.0
    x0 = int(np.floor(xs.min() * scale_x))
    y0 = int(np.floor(ys.min() * scale_y))
    x1 = int(np.ceil((xs.max() + 1) * scale_x))
    y1 = int(np.ceil((ys.max() + 1) * scale_y))
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(x0, min(x1, width))
    y1 = max(y0, min(y1, height))
    return (x0, y0, x1 - x0, y1 - y0)


def _probe_source_dimensions(source_path: Path, fallback_w: int = 0, fallback_h: int = 0) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(source_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or fallback_w)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or fallback_h)
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if declared > 0:
            return fps, width, height, declared
        counted = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            counted += 1
        return fps, width, height, counted
    finally:
        capture.release()


def _source_video_info(source_path: Path, mask: np.ndarray) -> tuple[float, int, int, int]:
    mask_h, mask_w = mask.shape[-2:]
    return _probe_source_dimensions(source_path, fallback_w=mask_w, fallback_h=mask_h)


def _write_streamed_temporal_mask_video(
    destination: Path,
    padded_stream: Iterator[np.ndarray],
    *,
    sampled_count: int,
    sampled_nonempty: list[bool],
    sample_stride: int,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
) -> tuple[bool, ...]:
    fourcc = "FFV1" if destination.suffix.lower() in {".mkv", ".avi"} else "mp4v"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to write mask video: {destination}")
    try:
        total = max(1, frame_count)
        stride = max(1, sample_stride)
        current_sample_index = -1
        current_bgr: np.ndarray | None = None
        active_flags: list[bool] = []
        for src_index in range(total):
            target_sample = min((src_index + (stride // 2)) // stride, max(0, sampled_count - 1))
            while current_sample_index < target_sample:
                nxt_mask = next(padded_stream, None)
                if nxt_mask is None:
                    break
                current_sample_index += 1
                current_bgr = _mask_to_bgr(nxt_mask, width, height)
            if current_bgr is None:
                current_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(current_bgr)
            active_idx = max(0, min(current_sample_index, len(sampled_nonempty) - 1))
            active_flags.append(bool(sampled_nonempty[active_idx]) if sampled_nonempty else False)
        return tuple(active_flags)
    finally:
        writer.release()


def _write_mask_video(
    destination: Path,
    mask: np.ndarray,
    *,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    if mask.ndim == 2 and destination.suffix.lower() in IMAGE_MASK_SUFFIXES:
        resized = cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(destination), resized):
            raise RuntimeError(f"failed to write mask image: {destination}")
        return
    # Masks are binary; a lossy codec smears their edges. FFV1 in Matroska is lossless and
    # readable by both OpenCV and VideoHelperSuite.
    fourcc = "FFV1" if destination.suffix.lower() in {".mkv", ".avi"} else "mp4v"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to write mask video: {destination}")
    try:
        total = max(1, frame_count)
        if mask.ndim == 2:
            rgb = _mask_to_bgr(mask, width, height)
            for _ in range(total):
                writer.write(rgb)
            return
        # Temporal mask: one frame per source frame, last mask repeated if the tracker
        # returned fewer frames, extra frames dropped if it returned more.
        last: np.ndarray | None = None
        for index in range(total):
            frame_mask = mask[min(index, mask.shape[0] - 1)]
            if last is None or index < mask.shape[0]:
                last = _mask_to_bgr(frame_mask, width, height)
            writer.write(last)
    finally:
        writer.release()


def _mask_to_bgr(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
