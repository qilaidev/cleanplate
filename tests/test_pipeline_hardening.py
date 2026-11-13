"""Ingest normalization, chunking, multi-ROI, quality gate and the no-watermark outcome."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wm_platform.comfy_graph import expand_roi_chain, set_frame_window
from wm_platform.media import Chunk, assemble_output, normalize_input, plan_chunks, probe_video
from wm_platform.models import JobCreate, JobResponse
from wm_platform.provider_runtime import ProviderRuntime, plan_minimax_rois
from wm_platform.quality_gate import check_output
from wm_platform.repository import JobRepository
from wm_platform.watermark_detect import detect_static_overlay, detect_watermark
from wm_platform.worker_service import WorkerService

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

LOGO = (12, 10, 60, 24)


def _frames(count: int, size=(320, 240), logo: tuple[int, int, int, int] | None = LOGO, seed: int = 0):
    width, height = size
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(height + 64, width + 64, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    frames = []
    for index in range(count):
        frame = base[index % 32 : index % 32 + height, index % 48 : index % 48 + width].copy()
        if logo is not None:
            x, y, w, h = logo
            frame[y : y + h, x : x + w] = (245, 245, 245)
            frame[y + 4 : y + h - 4, x + 4 : x + w - 4] = (20, 160, 250)
        frames.append(frame)
    return frames


def _write(path: Path, frames, fps: int = 10, fourcc: str = "FFV1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _create_job(job_repo, settings, source: Path, provider: str = "auto"):
    return job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested=provider,  # type: ignore[arg-type]
            fallback_chain_json=JobRepository.default_fallback_chain(provider),
            input_path=str(source),
        )
    )


# --- chunk planning -----------------------------------------------------------------


def test_plan_chunks_uses_4k_plus_1_windows_and_slides_the_tail():
    chunks = plan_chunks(200, 81)
    assert [c.count for c in chunks] == [81, 81, 81]
    assert chunks[-1].start == 200 - 81
    # Every output frame is produced exactly once after overlap trimming.
    kept = sum(c.count - c.keep_from for c in chunks)
    assert kept == 200
    assert all((c.count - 1) % 4 == 0 for c in chunks)


def test_plan_chunks_short_clip_and_disabled():
    assert plan_chunks(81, 81) == [Chunk(0, 81, 0)]
    short = plan_chunks(50, 81)
    assert [c.count for c in short] == [49, 49]
    assert sum(c.count - c.keep_from for c in short) == 50
    whole = plan_chunks(300, 0)
    assert [c.count for c in whole] == [297, 297]
    assert plan_chunks(0, 81) == []


# --- assembly and normalization ------------------------------------------------------


def test_assemble_output_trims_overlap_and_muxes_audio(tmp_path):
    frames = _frames(13)
    source = tmp_path / "with-audio.mkv"
    silent = _write(tmp_path / "silent.mkv", frames)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-f", "lavfi", "-i", "sine=duration=1.3",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(source)],
        check=True,
    )
    chunks = plan_chunks(13, 9)
    segments = []
    for index, chunk in enumerate(chunks):
        segments.append((_write(tmp_path / f"seg{index}.mkv", frames[chunk.start : chunk.start + chunk.count]), chunk))
    output = assemble_output(segments, source, tmp_path / "out.mp4", fps=10)
    info = probe_video(output)
    assert info.frame_count == 13
    assert info.has_audio


def test_normalize_input_bakes_rotation_and_odd_size(tmp_path):
    odd = _write(tmp_path / "odd.mkv", [f[:, :-1] for f in _frames(6)])
    encoded = tmp_path / "odd-h264.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(odd), "-c:v", "libx264", "-pix_fmt", "yuv444p", str(encoded)],
        check=True,
    )
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-display_rotation", "90", "-i", str(encoded), "-c", "copy", str(rotated)],
        check=True,
    )
    normalized = normalize_input(rotated, tmp_path / "work")
    assert normalized.normalized
    assert any(reason.startswith("rotation") for reason in normalized.reasons)
    assert normalized.info.rotation == 0
    assert normalized.info.width % 2 == 0 and normalized.info.height % 2 == 0
    assert normalized.info.width < normalized.info.height  # rotated into portrait
    clean = normalize_input(_write(tmp_path / "clean.mp4", _frames(4), fourcc="mp4v"), tmp_path / "w2")
    assert clean.path.name == "clean.mp4"


# --- ComfyUI graph transforms ------------------------------------------------------


def test_set_frame_window_updates_every_loader(settings):
    prompt = json.loads((settings.repo_root / "workflows" / "minimax_remover_api.json").read_text())
    set_frame_window(prompt, 81, 81)
    loaders = [n["inputs"] for n in prompt.values() if n["class_type"] == "VHS_LoadVideoPath"]
    assert len(loaders) == 2
    assert all(item["skip_first_frames"] == 81 and item["frame_load_cap"] == 81 for item in loaders)


def test_expand_roi_chain_serialises_sections(settings):
    template = json.loads((settings.repo_root / "workflows" / "minimax_remover_roi_api.json").read_text())
    rois = [(0, 0, 256, 256), (384, 0, 256, 256)]
    prompt = expand_roi_chain(
        template, [{"__ROI_X__": x, "__ROI_Y__": y, "__ROI_W__": w, "__ROI_H__": h} for x, y, w, h in rois]
    )
    composites = {k: v for k, v in prompt.items() if v["class_type"] == "ImageCompositeMasked"}
    crops = {k: v for k, v in prompt.items() if v["class_type"] == "ImageCrop"}
    assert len(composites) == 2 and len(crops) == 2
    first, second = sorted(composites)
    assert composites[first]["inputs"]["destination"] == ["1", 0]
    assert composites[second]["inputs"]["destination"] == [first, 0]
    assert crops["5_roi1"]["inputs"]["image"] == [first, 0]
    assert crops["5_roi1"]["inputs"]["x"] == 384
    create = next(v for v in prompt.values() if v["class_type"] == "CreateVideo")
    assert create["inputs"]["images"] == [second, 0]
    # model loader is shared, not duplicated
    assert sum(1 for v in prompt.values() if v["class_type"] == "MinimaxModelLoader") == 1


def test_plan_minimax_rois_keeps_opposite_corners_apart():
    rois = plan_minimax_rois([(20, 20, 120, 40), (1140, 660, 120, 40)], (1280, 720))
    assert rois is not None and len(rois) == 2
    merged = plan_minimax_rois([(20, 20, 60, 30), (100, 20, 60, 30)], (1280, 720))
    assert merged is not None and len(merged) == 1


# --- detection outcomes ------------------------------------------------------------


def test_static_detector_reports_none_for_clean_video(tmp_path):
    clean = _write(tmp_path / "clean.mkv", _frames(12, logo=None))
    result = detect_static_overlay(clean, tmp_path / "mask.mkv")
    assert result.nothing_found
    assert not result.found


def test_auto_detector_without_sam3_does_not_claim_none(tmp_path, settings):
    clean = _write(tmp_path / "clean.mkv", _frames(12, logo=None))
    result = detect_watermark(clean, tmp_path / "mask.mkv", replace(settings, detector_mode="auto"))
    assert not result.found
    assert not result.nothing_found  # SAM3 did not run, so the centre was never checked


def test_detector_regions_and_lossless_mask(tmp_path):
    frames = _frames(12, logo=None)
    for frame in frames:
        frame[10:34, 12:72] = (245, 245, 245)
        frame[200:224, 240:300] = (245, 245, 245)
    source = _write(tmp_path / "two.mkv", frames)
    result = detect_static_overlay(source, tmp_path / "mask.mkv")
    assert result.found
    assert len(result.regions) == 2
    capture = cv2.VideoCapture(str(result.mask_video_path))
    ok, mask_frame = capture.read()
    capture.release()
    assert ok and set(np.unique(mask_frame)) <= {0, 255}


# --- quality gate ---------------------------------------------------------------------


def _inpainted_copy(frames, logo=LOGO):
    fixed = [frame.copy() for frame in frames]
    x, y, w, h = logo
    for frame in fixed:
        frame[y : y + h, x : x + w] = cv2.blur(frame[y + h : y + 2 * h, x : x + w], (5, 5))
    return fixed


def test_quality_gate_flags_unchanged_output(tmp_path):
    frames = _frames(12)
    source = _write(tmp_path / "src.mkv", frames)
    detection = detect_static_overlay(source, tmp_path / "mask.mkv")
    assert detection.found
    assert check_output(source, source, detection).unchanged
    fixed = _write(tmp_path / "fixed.mkv", _inpainted_copy(frames))
    verdict = check_output(source, fixed, detection)
    assert verdict.status == "pass", verdict


# --- orchestration ------------------------------------------------------------------


def _runnable(runtime, *names):
    for name in names:
        runtime.registry[name].probe = lambda: SimpleNamespace(runnable=True, message="ready")


def test_chain_moves_on_when_inpainter_returns_input_unchanged(job_repo, settings, monkeypatch, tmp_path):
    frames = _frames(12)
    source = _write(settings.inbox_dir / "chain.mkv", frames)
    fixed = _write(tmp_path / "fixed.mkv", _inpainted_copy(frames))
    job = _create_job(job_repo, settings, source)
    runtime = ProviderRuntime(replace(settings, detector_mode="static_overlay"), repository=job_repo)
    _runnable(runtime, "minimax_remover", "comfy_diffueraser")
    calls = []

    def _noop(job_record, context=None):
        calls.append("minimax")
        out = context.work_dir / "noop.mkv"
        shutil.copyfile(context.input_path, out)
        return {"output_path": str(out)}

    def _good(job_record, context=None):
        calls.append("diffu")
        out = context.work_dir / "good.mkv"
        shutil.copyfile(fixed, out)
        return {"output_path": str(out)}

    monkeypatch.setattr(runtime.registry["minimax_remover"], "run", _noop)
    monkeypatch.setattr(runtime.registry["comfy_diffueraser"], "run", _good)
    service = WorkerService(settings=settings, repository=job_repo, providers=runtime)
    assert service.run_once() is True
    updated = job_repo.get_job(job.job_id)
    assert updated.status == "succeeded", updated.error_message
    assert updated.provider_selected == "comfy_diffueraser"
    assert calls == ["minimax", "diffu"]
    response = JobResponse.from_record(updated)
    assert response.report["attempts"][0]["error_code"] == "INPAINT_NO_EFFECT"
    assert response.report["qa"]["status"] == "pass"
    assert response.report["detection"]["scene_type"] in {"corner", "static_logo"}
    assert response.needs_review is False
    assert Path(updated.output_path).parent == settings.outbox_dir
    assert not (settings.work_dir / job.job_id).exists()


def test_all_unchanged_outputs_fail_with_inpaint_no_effect(job_repo, settings, monkeypatch):
    source = _write(settings.inbox_dir / "noop.mkv", _frames(12))
    job = _create_job(job_repo, settings, source, provider="minimax_remover")
    runtime = ProviderRuntime(replace(settings, detector_mode="static_overlay"), repository=job_repo)
    _runnable(runtime, "minimax_remover")

    def _noop(job_record, context=None):
        out = context.work_dir / "noop.mkv"
        shutil.copyfile(context.input_path, out)
        return {"output_path": str(out)}

    monkeypatch.setattr(runtime.registry["minimax_remover"], "run", _noop)
    WorkerService(settings=settings, repository=job_repo, providers=runtime).run_once()
    updated = job_repo.get_job(job.job_id)
    assert updated.status == "failed"
    assert updated.error_code == "INPAINT_NO_EFFECT"
    assert updated.report["attempts"][0]["qa"]["status"] == "unchanged"


@pytest.mark.parametrize("policy", ["fail", "passthrough"])
def test_no_watermark_policy(job_repo, settings, monkeypatch, policy):
    source = _write(settings.inbox_dir / f"clean-{policy}.mkv", _frames(12, logo=None))
    job = _create_job(job_repo, settings, source)
    runtime = ProviderRuntime(
        replace(settings, detector_mode="static_overlay", no_watermark_policy=policy), repository=job_repo
    )
    _runnable(runtime, "minimax_remover", "comfy_diffueraser")
    WorkerService(settings=settings, repository=job_repo, providers=runtime).run_once()
    updated = job_repo.get_job(job.job_id)
    if policy == "fail":
        assert updated.status == "failed"
        assert updated.error_code == "NO_WATERMARK_DETECTED"
    else:
        assert updated.status == "succeeded"
        assert updated.provider_selected == "passthrough"
        assert updated.report["action"] == "passthrough"
        assert Path(updated.output_path).read_bytes() == source.read_bytes()


def test_doctor_lists_non_commercial_weights(settings):
    from wm_platform.doctor import license_report

    report = license_report(settings)
    names = {item["name"] for item in report["non_commercial_models"]}
    assert {"minimax-remover", "propainter"} <= names
    assert report["commercial_ready"] is False
