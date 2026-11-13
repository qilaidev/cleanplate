"""Tests for the 2026-09 upgrades: native SAM 3.1 detection, temporal masks, ROI inpainting,
persistent-edge (translucent) detection and clean-plate eval metrics."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.test_runtime_eval_detector import _fake_comfy_client, _write_mask_png, fake_video_bytes
from wm_platform.eval_fixtures import generate_eval_fixtures
from wm_platform.eval_judge import merge_judgments_csv
from wm_platform.eval_metrics import auto_judgment, evaluate_output, load_region
from wm_platform.eval_runner import load_eval_cases, run_eval
from wm_platform.models import JobCreate
from wm_platform.provider_runtime import ProviderRuntime, plan_minimax_roi
from wm_platform.repository import JobRepository
from wm_platform.watermark_detect import (
    SAM3_NATIVE_CHECKPOINT,
    SAM3_WORKFLOW_NAME,
    detect_static_overlay,
    detect_watermark,
    sam3_detector_status,
)


def _write_logo_video(path: Path, *, size: tuple[int, int], logo: tuple[int, int, int, int], frames: int = 8) -> Path:
    width, height = size
    x, y, w, h = logo
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.full((height, width, 3), 30, dtype=np.uint8)
            # moving background element so the frame is not entirely static
            cv2.circle(frame, (width // 2 + index * 7, height // 2), 25, (90, 160, 200), -1)
            frame[y : y + h, x : x + w] = (240, 240, 240)
            frame[y + 3 : y + h - 3, x + 3 : x + w - 3] = (20, 180, 255)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_mask_video(path: Path, boxes: list[tuple[int, int, int, int]], *, size=(160, 120)) -> bytes:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))
    assert writer.isOpened()
    try:
        for x, y, w, h in boxes:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[y : y + h, x : x + w] = 255
            writer.write(frame)
    finally:
        writer.release()
    return path.read_bytes()


# --- ROI planning -----------------------------------------------------------------


def test_plan_minimax_roi_pads_aligns_and_contains_bbox():
    roi = plan_minimax_roi((20, 30, 60, 24), (1280, 720))
    assert roi is not None
    x, y, w, h = roi
    assert x % 16 == 0 and y % 16 == 0 and w % 16 == 0 and h % 16 == 0
    assert w >= 256 and h >= 256
    assert x <= 20 and y <= 30 and x + w >= 80 and y + h >= 54
    assert w * h < 0.6 * 1280 * 720


def test_plan_minimax_roi_bottom_right_stays_inside_frame():
    roi = plan_minimax_roi((1180, 660, 80, 40), (1280, 720))
    assert roi is not None
    x, y, w, h = roi
    assert x + w <= 1280 and y + h <= 720
    assert x <= 1180 and y <= 660


def test_plan_minimax_roi_declines_small_frames_and_large_boxes():
    assert plan_minimax_roi((4, 4, 44, 24), (160, 120)) is None
    assert plan_minimax_roi((100, 100, 1000, 500), (1280, 720)) is None


# --- MiniMax provider: ROI workflow ------------------------------------------------


def _minimax_runtime(tmp_path: Path, settings, **overrides):
    nodes = tmp_path / "custom_nodes" / "ComfyUI-MiniMax-Remover"
    nodes.mkdir(parents=True)
    venv = tmp_path / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("", encoding="utf-8")
    models = tmp_path / "models" / "MiniMax-Remover"
    for name in ("vae", "transformer", "scheduler"):
        (models / name).mkdir(parents=True)
    return replace(
        settings,
        comfyui_custom_nodes_dir=tmp_path / "custom_nodes",
        comfyui_venv_dir=tmp_path / ".venv",
        comfyui_models_dir=tmp_path / "models",
        comfyui_minimax_workflow=settings.repo_root / "workflows" / "minimax_remover_api.json",
        comfyui_minimax_roi_workflow=settings.repo_root / "workflows" / "minimax_remover_roi_api.json",
        quality_mode="balanced",
        auto_start_comfyui=False,
        **overrides,
    )


def _run_minimax(job_repo, isolated, source: Path, monkeypatch):
    job = job_repo.create_job(
        JobCreate(
            tenant_id=isolated.default_tenant_id,
            media_type="video",
            provider_requested="minimax_remover",
            fallback_chain_json=JobRepository.default_fallback_chain("minimax_remover"),
            input_path=str(source),
            quality_mode="quality",
        )
    )
    captured: dict[str, object] = {}
    fake = _fake_comfy_client("video_00001_.mp4", fake_video_bytes(source.parent), captured)
    monkeypatch.setattr("wm_platform.provider_runtime.httpx.Client", fake)
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", fake)
    provider = ProviderRuntime(isolated, repository=job_repo).registry["minimax_remover"]
    result = provider.run(job)
    return job, result, captured["prompt"]


def test_minimax_provider_uses_roi_workflow_for_corner_logo(job_repo, settings, monkeypatch, tmp_path):
    isolated = _minimax_runtime(tmp_path, settings)
    source = _write_logo_video(settings.inbox_dir / "roi-logo.mp4", size=(640, 360), logo=(24, 20, 120, 40))
    job, result, prompt = _run_minimax(job_repo, isolated, source, monkeypatch)

    assert Path(result["output_path"]).stat().st_size > 0
    assert result["workflow_name"] == "minimax_remover_roi_api.json"
    roi = tuple(int(item) for item in result["roi"].split(","))
    assert len(roi) == 4
    by_type = {node["class_type"]: node["inputs"] for node in prompt.values()}
    assert by_type["ImageCrop"]["x"] == roi[0] and by_type["ImageCrop"]["y"] == roi[1]
    assert by_type["ImageCrop"]["width"] == roi[2] and by_type["ImageCrop"]["height"] == roi[3]
    assert by_type["CropMask"]["width"] == roi[2]
    assert by_type["ImageCompositeMasked"]["x"] == roi[0]
    assert by_type["ImageCompositeMasked"]["resize_source"] is False
    assert isinstance(by_type["GrowMask"]["expand"], int) and by_type["GrowMask"]["expand"] == 8
    assert by_type["ImageBlur"]["blur_radius"] == 7
    assert by_type["MinimaxVideoRemover"]["num_inference_steps"] == 12
    assert by_type["SaveVideo"]["filename_prefix"].startswith(f"video/{job.job_id}-")
    # no placeholder survives
    assert "__" not in json.dumps(prompt)
    metadata = job_repo.get_run_metadata(job.job_id)
    assert metadata is not None
    assert metadata.workflow_name == "minimax_remover_roi_api.json"
    # ROI contains the logo and stays on the 16 px grid
    assert roi[0] <= 24 and roi[1] <= 20 and roi[0] + roi[2] >= 144 and roi[1] + roi[3] >= 60
    assert all(value % 16 == 0 for value in roi)


def test_minimax_provider_full_frame_when_roi_disabled(job_repo, settings, monkeypatch, tmp_path):
    isolated = _minimax_runtime(tmp_path, settings, minimax_roi_enabled=False)
    source = _write_logo_video(settings.inbox_dir / "roi-off-logo.mp4", size=(640, 360), logo=(24, 20, 120, 40))
    _job, result, prompt = _run_minimax(job_repo, isolated, source, monkeypatch)
    assert "roi" not in result
    assert result["workflow_name"] == "minimax_remover_api.json"
    assert not any(node["class_type"] == "ImageCrop" for node in prompt.values())


def test_minimax_probe_reports_roi_workflow(settings, tmp_path):
    isolated = _minimax_runtime(tmp_path, settings)
    probe = ProviderRuntime(isolated).registry["minimax_remover"].probe()
    assert probe.details["roi_enabled"] is True
    assert probe.details["roi_workflow_ready"] is True
    assert probe.details["roi_workflow_name"] == "minimax_remover_roi_api.json"
    assert "sam3_available" in probe.details


# --- SAM 3.1 native stack ---------------------------------------------------------


def _native_sam3_runtime(tmp_path: Path, settings):
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    models = comfy / "models"
    (comfy / "comfy_extras").mkdir()
    (comfy / "comfy_extras" / "nodes_sam3.py").write_text("# stub", encoding="utf-8")
    (models / "checkpoints").mkdir(parents=True)
    (models / "checkpoints" / SAM3_NATIVE_CHECKPOINT).write_bytes(b"stub")
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    shutil.copy(settings.repo_root / "workflows" / SAM3_WORKFLOW_NAME, workflows / SAM3_WORKFLOW_NAME)
    return replace(
        settings,
        detector_mode="sam3_text",
        comfyui_dir=comfy,
        comfyui_custom_nodes_dir=comfy / "custom_nodes",
        comfyui_models_dir=models,
        comfyui_workflows_dir=workflows,
        comfyui_sam3_workflow=workflows / SAM3_WORKFLOW_NAME,
        auto_start_comfyui=False,
    )


def test_sam3_status_native_stack_available(tmp_path, settings):
    isolated = _native_sam3_runtime(tmp_path, settings)
    status = sam3_detector_status(isolated)
    assert status["available"] is True, status
    assert status["workflow_path"].endswith(SAM3_WORKFLOW_NAME)


def test_sam3_status_native_missing_checkpoint_is_reported(tmp_path, settings):
    isolated = _native_sam3_runtime(tmp_path, settings)
    (isolated.comfyui_models_dir / "checkpoints" / SAM3_NATIVE_CHECKPOINT).unlink()
    status = sam3_detector_status(isolated)
    assert status["available"] is False
    assert any("checkpoint" in str(item) for item in status["issues"])



def test_sam3_native_mock_comfy_injects_checkpoint_and_prompt(tmp_path, settings, monkeypatch):
    isolated = _native_sam3_runtime(tmp_path, settings)
    captured: dict[str, object] = {}
    mask_bytes = _write_mask_png(tmp_path / "corner-mask.png")
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", _fake_comfy_client("mask.png", mask_bytes, captured))
    source = _write_logo_video(tmp_path / "logo.mp4", size=(160, 120), logo=(4, 4, 44, 24))
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level != "low", result
    assert result.detector == "sam3_text"
    assert result.temporal is False
    prompt = captured["prompt"]
    by_type = {node["class_type"]: node["inputs"] for node in prompt.values()}
    assert by_type["CheckpointLoaderSimple"]["ckpt_name"] == SAM3_NATIVE_CHECKPOINT
    assert by_type["CLIPTextEncode"]["text"] == isolated.watermark_detect_prompt
    assert "SAM3_VideoTrack" in by_type and "SAM3_TrackToMask" in by_type
    assert by_type["SaveVideo"]["filename_prefix"].startswith("mask/")
    assert "__" not in json.dumps(prompt)


def test_sam3_moving_mask_video_keeps_per_frame_masks(tmp_path, settings, monkeypatch):
    isolated = _native_sam3_runtime(tmp_path, settings)
    boxes = [(4 + index * 12, 6, 44, 22) for index in range(8)]
    mask_bytes = _write_mask_video(tmp_path / "moving-mask.mp4", boxes)
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", _fake_comfy_client("mask_00001_.mp4", mask_bytes))
    source = _write_logo_video(tmp_path / "drift.mp4", size=(160, 120), logo=(4, 6, 44, 22))
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level != "low", result
    assert result.temporal is True
    assert result.scene_type == "moving_overlay"
    assert result.mask_video_path is not None and result.mask_video_path.exists()
    # union bbox spans the whole drift
    assert result.bbox is not None
    assert result.bbox[0] <= 4 and result.bbox[0] + result.bbox[2] >= 4 + 7 * 12 + 44 - 2
    # written mask really varies over time
    capture = cv2.VideoCapture(str(result.mask_video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127)
    capture.release()
    assert len(frames) == 8
    assert frames[0].sum() < frames[0].size * 0.2
    assert not np.array_equal(frames[0], frames[-1])


def test_sam3_static_mask_video_collapses_to_union(tmp_path, settings, monkeypatch):
    isolated = _native_sam3_runtime(tmp_path, settings)
    boxes = [(4, 4, 44, 24)] * 8
    mask_bytes = _write_mask_video(tmp_path / "static-mask.mp4", boxes)
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", _fake_comfy_client("mask_00001_.mp4", mask_bytes))
    source = _write_logo_video(tmp_path / "logo.mp4", size=(160, 120), logo=(4, 4, 44, 24))
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level != "low", result
    assert result.temporal is False
    assert result.scene_type in {"corner", "sam3_text"}


# --- translucent overlay detection -------------------------------------------------


def test_static_detector_finds_translucent_overlay_on_moving_background(tmp_path):
    written = generate_eval_fixtures(tmp_path)
    by_name = {path.name: path for path in written}
    result = detect_static_overlay(by_name["medium_translucent.mp4"], tmp_path / "translucent.mask.mp4")
    assert result.confidence_level in {"high", "medium"}
    assert result.bbox is not None
    x, y, w, h = result.bbox
    # the overlay lives at rows 6..32, cols 6..56
    assert x <= 8 and y <= 8 and x + w >= 52 and y + h >= 28
    assert result.mask_ratio < 0.12


def test_static_detector_keeps_rejecting_hard_fixtures(tmp_path):
    written = generate_eval_fixtures(tmp_path)
    by_name = {path.name: path for path in written}
    for name in ("hard_large.mp4", "hard_multi.mp4", "hard_center.mp4", "medium_drift.mp4"):
        result = detect_static_overlay(by_name[name], tmp_path / f"{name}.mask.mp4")
        assert result.confidence_level == "low", name
        assert result.mask_video_path is None


# --- clean-plate metrics -------------------------------------------------------------


def test_eval_metrics_clean_output_passes_and_copy_fails(tmp_path):
    written = generate_eval_fixtures(tmp_path)
    dirty = next(path for path in written if path.name == "simple_top_left.mp4")
    clean = tmp_path / "clean" / "simple_top_left.mkv"
    region_path = tmp_path / "clean" / "simple_top_left.region.png"
    region = load_region(region_path)
    assert region is not None and 0.01 < region.mean() < 0.1
    assert region[4:28, 4:48].all() and not region[60:, 80:].any()
    perfect = evaluate_output(clean, dirty, clean, region_path=region_path)
    assert perfect["psnr_region"] >= 60
    assert perfect["region_source"] == "ground_truth"
    assert perfect["residual_detected"] is False
    assert auto_judgment(perfect) == "pass"
    untouched = evaluate_output(dirty, dirty, clean, region_path=region_path)
    assert untouched["psnr_region"] < 22
    assert untouched["residual_detected"] is True
    assert auto_judgment(untouched) == "fail"


def test_eval_runner_records_metrics_but_never_auto_passes_local_fallback(tmp_path, settings, job_repo):
    generate_eval_fixtures(tmp_path)
    cases = load_eval_cases(tmp_path, "simple")
    assert cases and all(case.clean_path is not None and case.region_path is not None for case in cases)
    payload = run_eval(
        dataset_root=tmp_path,
        split="simple",
        providers=["local_fallback"],
        settings=settings,
        repository=job_repo,
        results_dir=tmp_path / "results",
    )
    rows = payload["results"]
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "succeeded"
        assert row["judgment"] == "pending"
        assert row["judgment_source"] == "manual"
        assert isinstance(row["metrics"], dict)
        assert row["metrics"]["frames_compared"] > 0
        assert row["metrics"]["region_source"] == "ground_truth"
        assert row["metrics"]["residual_detected"] is True


def test_manual_judgment_csv_overrides_auto_judgment(tmp_path):
    csv_path = tmp_path / "judgments.csv"
    csv_path.write_text("id,judgment,provider\nsimple-tl,fail,\n", encoding="utf-8")
    rows = [
        {"id": "simple-tl", "provider_requested": "minimax_remover", "judgment": "pass", "judgment_source": "auto"},
        {"id": "simple-br", "provider_requested": "minimax_remover", "judgment": "pass", "judgment_source": "manual"},
    ]
    merged = merge_judgments_csv(rows, csv_path)
    assert merged[0]["judgment"] == "fail" and merged[0]["judgment_source"] == "manual"
    assert merged[1]["judgment"] == "pass"


@pytest.mark.parametrize("name", ["sam3_native_api.json", "minimax_remover_roi_api.json"])
def test_new_workflow_files_are_api_prompts(settings, name):
    prompt = json.loads((settings.repo_root / "workflows" / name).read_text(encoding="utf-8"))
    assert prompt and all("class_type" in node and "inputs" in node for node in prompt.values())
    class_types = {node["class_type"] for node in prompt.values()}
    assert "SaveVideo" in class_types
