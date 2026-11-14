"""Governance & reliability regression tests: OOM caps, spool isolation, A/V & color fidelity,
watchdog/timeout recovery, ComfyUI orphan cleanup, SHA-256 HTTP model installer, and chunk bypass."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pytest

from wm_platform.api_app import create_app
from wm_platform.comfy_graph import set_frame_window
from wm_platform.comfy_jobs import ComfyJobClient
from wm_platform.comfy_runtime import build_comfyui_command
from wm_platform.db import db_connection
from wm_platform.errors import AppError
from wm_platform.eval_runner import build_eval_settings
from wm_platform.job_locks import JobFileLock
from wm_platform.maintenance import run_file_cleanup
from wm_platform.media import (
    Chunk,
    VideoInfo,
    assemble_output,
    normalize_input,
    plan_chunks,
    probe_video,
    sdr_color_args,
    sdr_setparams_filter,
)
from wm_platform.media import (
    _run as run_media_cmd,
)
from wm_platform.models import JobCreate
from wm_platform.provider_runtime import PipelineContext, run_comfy_chunks
from wm_platform.quality_gate import check_output
from wm_platform.repository import JobRepository
from wm_platform.runtime_contract import expected_model_entries
from wm_platform.runtime_installer import RuntimeInstaller
from wm_platform.watermark_detect import (
    SAM3_MAX_SAMPLED_FRAMES,
    SAM3_MAX_SIDE,
    DetectionResult,
    _configure_sam3_loader_limits,
    _finalize_streamed_mask,
    _iter_padded_masks,
    _pad_temporal,
    detect_static_overlay,
)
from wm_platform.worker_service import WorkerService


def _write_textured_video(
    path: Path,
    *,
    frames: int = 16,
    size: tuple[int, int] = (160, 120),
    fps: int = 10,
    logo_box: tuple[int, int, int, int] | None = (8, 8, 40, 20),
    active_range: tuple[int, int] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    rng = np.random.default_rng(42)
    base = rng.integers(45, 195, size=(height + 32, width + 32, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    fourcc = "FFV1" if path.suffix.lower() == ".mkv" else "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            frame = base[idx % 24 : idx % 24 + height, idx % 24 : idx % 24 + width].copy()
            show_logo = logo_box is not None and (
                active_range is None or (active_range[0] <= idx < active_range[1])
            )
            if show_logo and logo_box is not None:
                x, y, w, h = logo_box
                frame[y : y + h, x : x + w] = (245, 245, 245)
                frame[y + 4 : y + h - 4, x + 4 : x + w - 4] = (20, 170, 250)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _count_readable_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            count += 1
    finally:
        capture.release()
    return count


# --- 1. VRAM & RAM OOM Protection --------------------------------------------------


def test_sam3_workflow_caps_frames_and_downscales_4k_strided(settings):
    workflow_path = settings.repo_root / "workflows" / "sam3_native_api.json"
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert prompt["1"]["inputs"]["frame_load_cap"] == SAM3_MAX_SAMPLED_FRAMES

    stride = _configure_sam3_loader_limits(prompt, width=3840, height=2160, frame_count=540)
    assert stride == 3
    loader = prompt["1"]["inputs"]
    assert loader["frame_load_cap"] == SAM3_MAX_SAMPLED_FRAMES
    assert loader["select_every_nth"] == 3
    assert loader["custom_width"] == SAM3_MAX_SIDE
    assert loader["custom_height"] == 0


def test_sam3_streaming_temporal_mask_matches_padded_array_and_expands_stride(tmp_path):
    mask_src = tmp_path / "sampled_mask.mkv"
    width, height = 160, 120
    writer = cv2.VideoWriter(str(mask_src), cv2.VideoWriter_fourcc(*"FFV1"), 10, (width, height))
    assert writer.isOpened()
    raw_bools = []
    try:
        for idx in range(6):
            canvas = np.zeros((height, width), dtype=np.uint8)
            if idx < 3:
                x = 6 + idx * 18
                canvas[10:30, x : x + 32] = 255
            raw_bools.append(canvas > 127)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()

    streamed = list(_iter_padded_masks(mask_src))
    stacked_ref = _pad_temporal(np.stack(raw_bools, axis=0), 1)
    assert len(streamed) == 6
    for got, expected in zip(streamed, stacked_ref, strict=True):
        assert np.array_equal(got, expected)

    source = _write_textured_video(tmp_path / "source.mp4", frames=12, size=(width, height), logo_box=None)
    dest_mask = tmp_path / "out_mask.mkv"
    result = _finalize_streamed_mask(
        source,
        dest_mask,
        mask_src,
        fps=10.0,
        width=width,
        height=height,
        frame_count=12,
        sample_stride=2,
        max_mask_ratio=0.25,
        min_mask_ratio=0.0004,
    )
    assert result.found
    assert result.temporal is True
    assert result.scene_type == "moving_overlay"
    assert _count_readable_frames(dest_mask) == 12
    assert result.active_frames is not None
    assert len(result.active_frames) == 12
    assert any(result.active_frames[:6])
    assert not any(result.active_frames[-2:])


def test_static_mask_frame_cap_png_and_loader_window_pinning(tmp_path, settings):
    source = _write_textured_video(tmp_path / "static20.mp4", frames=20)
    capped = detect_static_overlay(source, tmp_path / "capped_mask.mkv", max_static_frames=5)
    assert capped.found
    assert _count_readable_frames(tmp_path / "capped_mask.mkv") == 5

    png_res = detect_static_overlay(source, tmp_path / "single_mask.png")
    assert png_res.found
    png_img = cv2.imread(str(tmp_path / "single_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert png_img is not None and png_img.shape == (120, 160)

    prompt = json.loads((settings.repo_root / "workflows" / "minimax_remover_api.json").read_text(encoding="utf-8"))
    prompt["1"]["inputs"]["video"] = "/tmp/source.mp4"
    prompt["3"]["inputs"]["video"] = "/tmp/mask.mkv"
    set_frame_window(prompt, 81, 81, static_mask_video="/tmp/mask.mkv")
    assert prompt["1"]["inputs"]["skip_first_frames"] == 81
    assert prompt["1"]["inputs"]["frame_load_cap"] == 81
    assert prompt["3"]["inputs"]["skip_first_frames"] == 0
    assert prompt["3"]["inputs"]["frame_load_cap"] == 81


# --- 2. Disk Spool Isolation & Recursive Maintenance Cleanup -----------------------


def test_comfyui_command_uses_isolated_spool_and_download_cleans_spool(settings, tmp_path):
    command = build_comfyui_command(settings)
    out_idx = command.index("--output-directory")
    assert Path(command[out_idx + 1]) == settings.comfyui_output_dir
    assert settings.comfyui_output_dir != settings.outbox_dir

    spool_video = settings.comfyui_output_dir / "video"
    spool_video.mkdir(parents=True, exist_ok=True)
    spool_file = spool_video / "seg_0001.mp4"
    spool_file.write_bytes(b"segment-bytes")

    dest = tmp_path / "work" / "0000.mp4"
    client = ComfyJobClient(settings)
    with pytest.MonkeyPatch.context():
        import httpx

        with httpx.Client() as real_http:
            moved = client.download_to(
                real_http,
                {"filename": "seg_0001.mp4", "subfolder": "video", "type": "output"},
                dest,
            )
    assert moved == dest
    assert dest.read_bytes() == b"segment-bytes"
    assert not spool_file.exists()


def test_maintenance_recursively_cleans_subdirectories_and_prunes_empty_dirs(job_repo, settings):
    cutoff = datetime.now(UTC) - timedelta(days=settings.file_retention_days + 2)
    old_ts = cutoff.timestamp()

    nested_video = settings.outbox_dir / "video" / "leaked_chunk.mp4"
    nested_mask = settings.outbox_dir / "mask" / "leaked_mask.mp4"
    spool_orphan = settings.comfyui_output_dir / "video" / "orphan_spool.mp4"
    protected_nested = settings.inbox_dir / "nested" / "active.mp4"
    for item in (nested_video, nested_mask, spool_orphan, protected_nested):
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_bytes(b"payload")
        os.utime(item, (old_ts, old_ts))

    active_job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="local_fallback",
            fallback_chain_json=JobRepository.default_fallback_chain("local_fallback"),
            input_path=str(protected_nested),
        )
    )
    with db_connection(settings) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', updated_at = ? WHERE job_id = ?",
            (datetime.now(UTC).isoformat(), active_job.job_id),
        )

    dry = run_file_cleanup(settings=settings, repository=job_repo, execute=False)
    assert int(dry["candidate_files"]) == 3
    assert nested_video.exists()

    executed = run_file_cleanup(settings=settings, repository=job_repo, execute=True)
    assert int(executed["deleted_files"]) == 3
    assert not nested_video.exists()
    assert not nested_mask.exists()
    assert not spool_orphan.exists()
    assert not (settings.outbox_dir / "video").exists()
    assert not (settings.outbox_dir / "mask").exists()
    assert settings.outbox_dir.exists()
    assert settings.comfyui_output_dir.exists()
    assert protected_nested.exists()


# --- 3. A/V Sync, Lossless Audio Copy & SDR Color Fidelity -------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_assemble_output_copies_aac_and_preserves_tail_frames_with_short_audio(tmp_path):
    silent = _write_textured_video(tmp_path / "silent20.mkv", frames=20, fps=10)
    short_audio_src = tmp_path / "short_audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.7",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(short_audio_src),
        ],
        check=True,
    )
    assembled = assemble_output(
        [(silent, Chunk(start=0, count=20, keep_from=0))],
        short_audio_src,
        tmp_path / "assembled_short.mp4",
        fps=10.0,
    )
    info = probe_video(assembled)
    assert info.frame_count == 20
    assert info.has_audio is True
    assert info.audio_codec == "aac"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_assemble_output_clamps_overlong_audio_and_transcodes_pcm_for_mp4(tmp_path):
    silent = _write_textured_video(tmp_path / "silent20.mkv", frames=20, fps=10)
    long_pcm_wav = tmp_path / "long.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4.0",
            "-c:a",
            "pcm_s16le",
            str(long_pcm_wav),
        ],
        check=True,
    )
    assembled = assemble_output(
        [(silent, Chunk(start=0, count=20, keep_from=0))],
        long_pcm_wav,
        tmp_path / "assembled_pcm.mp4",
        fps=10.0,
    )
    info = probe_video(assembled)
    assert info.frame_count == 20
    assert info.has_audio is True
    assert info.audio_codec == "aac"
    assert info.duration is not None and info.duration <= 2.3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_normalize_and_assemble_preserve_bt709_and_skip_hdr_metadata(tmp_path):
    bt709_odd = tmp_path / "bt709_odd.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=161x121:r=10:d=1.0",
            "-vf",
            "format=yuv444p,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-c:v",
            "libx264",
            str(bt709_odd),
        ],
        check=True,
    )
    normalized = normalize_input(bt709_odd, tmp_path / "work")
    assert normalized.normalized
    assert normalized.info.color_space == "bt709"
    assert normalized.info.color_primaries == "bt709"
    assert normalized.info.color_transfer == "bt709"

    out = assemble_output(
        [(normalized.path, Chunk(start=0, count=10, keep_from=0))],
        None,
        tmp_path / "final_bt709.mp4",
        fps=10.0,
        color_info=normalized.source_info,
    )
    out_info = probe_video(out)
    assert out_info.color_space == "bt709"
    assert out_info.color_primaries == "bt709"
    assert out_info.color_transfer == "bt709"

    hdr_info = VideoInfo(
        width=1920,
        height=1080,
        fps=24.0,
        frame_count=24,
        duration=1.0,
        has_audio=False,
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="smpte2084",
    )
    assert hdr_info.is_hdr is True
    assert sdr_color_args(hdr_info) == []
    assert sdr_setparams_filter(hdr_info) == ""


# --- 4. Timeouts, Watchdog, Queue Interrupt, Installer SHA-256 & Chunk Bypass ------


def test_media_run_converts_subprocess_timeout_to_app_error(monkeypatch):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1.0))

    monkeypatch.setattr("wm_platform.media.subprocess.run", _timeout)
    with pytest.raises(AppError) as exc:
        run_media_cmd(["ffmpeg", "-version"], "probe timeout check", timeout=1.5)
    assert exc.value.error_code == "MEDIA_PROCESSING_FAILED"
    assert exc.value.status_code == 504
    assert "timed out" in exc.value.error_message


def test_worker_watchdog_stops_heartbeat_and_releases_file_lock(job_repo, settings):
    source = _write_textured_video(settings.inbox_dir / "watchdog.mp4", frames=4)
    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="local_fallback",
            fallback_chain_json=JobRepository.default_fallback_chain("local_fallback"),
            input_path=str(source),
        )
    )
    claimed = job_repo.claim_next_job(lock_owner="hung-worker")
    assert claimed is not None

    watchdog_settings = replace(
        settings,
        job_claim_heartbeat_seconds=0.02,
        job_claim_timeout_seconds=1,
        job_watchdog_timeout_seconds=0.05,
    )
    service = WorkerService(settings=watchdog_settings, repository=job_repo, providers=None)  # type: ignore[arg-type]
    service.lock_owner = "hung-worker"

    job_lock = JobFileLock(watchdog_settings, job.job_id)
    assert job_lock.acquire() is True

    progress_lock = threading.Lock()
    progress_state: dict[str, float | str] = {
        "last_progress": time.monotonic() - 1.0,
        "stage": "comfy_diffueraser:chunk:0/2",
    }
    stop_event = threading.Event()
    try:
        service._heartbeat_job_claim(job.job_id, stop_event, progress_lock, progress_state, job_lock)
        stale_at = datetime.now(UTC) - timedelta(seconds=10)
        with db_connection(watchdog_settings) as conn:
            conn.execute(
                "UPDATE jobs SET claimed_at = ?, updated_at = ? WHERE job_id = ?",
                (stale_at.isoformat(), stale_at.isoformat(), job.job_id),
            )
        # Because the watchdog released job_lock and exited heartbeat renewal, stale claim recovery succeeds!
        assert job_repo.reset_stale_claims(watchdog_settings.job_claim_timeout_seconds) == 1
        reclaimed = job_repo.get_job(job.job_id)
        assert reclaimed is not None and reclaimed.status == "queued"
    finally:
        stop_event.set()
        job_lock.release()


def test_comfy_wait_timeout_cancels_queue_and_interrupts_remote_prompt(settings, monkeypatch):
    posted: list[tuple[str, object]] = []

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {}

    class _Client:
        def get(self, url, timeout=None, params=None):
            return _Resp()

        def post(self, url, json=None):
            posted.append((urlparse(url).path, json))
            return _Resp()

    clock = {"now": 100.0}

    def _monotonic():
        clock["now"] += 40.0
        return clock["now"]

    monkeypatch.setattr("wm_platform.comfy_jobs.time.monotonic", _monotonic)
    monkeypatch.setattr("wm_platform.comfy_jobs.time.sleep", lambda _s: None)
    fast_timeout = replace(settings, comfyui_prompt_timeout_seconds=1.0)
    with pytest.raises(AppError, match="timed out"):
        ComfyJobClient(fast_timeout).wait_for_artifact(_Client(), "orphan-prompt-42")

    assert ("/queue", {"delete": ["orphan-prompt-42"]}) in posted
    assert ("/interrupt", {"prompt_id": "orphan-prompt-42"}) in posted


def test_runtime_installer_downloads_propainter_http_bundle_with_sha256_check(tmp_path, settings, monkeypatch):
    entries = expected_model_entries(settings)
    propainter_entry = next(item for item in entries if item.get("name") == "propainter")
    assert len(propainter_entry.get("files") or []) == 3
    assert set(propainter_entry.get("must_contain") or []) == {
        "ProPainter.pth",
        "raft-things.pth",
        "recurrent_flow_completion.pth",
    }

    payloads = {
        "ProPainter.pth": b"propainter-weights-v0.1.0",
        "raft-things.pth": b"raft-things-weights-v0.1.0",
        "recurrent_flow_completion.pth": b"recurrent-flow-weights-v0.1.0",
    }
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    manifest_lines = [
        "models:",
        "  - name: propainter",
        "    required: true",
        "    providers: [comfy_diffueraser]",
        "    expected_path: .runtime/ComfyUI/models/DiffuEraser/propainter",
        "    source: https://github.com/sczhou/ProPainter/releases/tag/v0.1.0",
        "    must_contain: [ProPainter.pth, raft-things.pth, recurrent_flow_completion.pth]",
        "    files:",
    ]
    for fname, content in payloads.items():
        digest = hashlib.sha256(content).hexdigest()
        manifest_lines.extend(
            [
                f"      - filename: {fname}",
                f"        url: https://example.invalid/releases/{fname}",
                f"        sha256: {digest}",
            ]
        )
    (runtime / "models" / "manifest.yaml").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    corrupt = {"enabled": False}

    class _FakeStream:
        def __init__(self, data: bytes):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size: int = 1024):
            yield self._data

    class _FakeHttpClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method: str, url: str):
            name = url.rsplit("/", 1)[-1]
            body = payloads[name]
            if corrupt["enabled"] and name == "raft-things.pth":
                body = b"corrupted-bytes"
            return _FakeStream(body)

    monkeypatch.setattr("wm_platform.runtime_installer.httpx.Client", _FakeHttpClient)
    isolated = replace(
        settings,
        repo_root=repo,
        runtime_root=runtime,
        comfyui_dir=runtime / "ComfyUI",
        comfyui_models_dir=runtime / "ComfyUI" / "models",
    )
    installer = RuntimeInstaller(isolated)

    corrupt["enabled"] = True
    bad_messages = installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert any("sha256 mismatch" in msg for msg in bad_messages)
    dest_dir = runtime / "ComfyUI" / "models" / "DiffuEraser" / "propainter"
    assert not list(dest_dir.glob("*.tmp"))

    corrupt["enabled"] = False
    good_messages = installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert any("installed model propainter" in msg for msg in good_messages)
    for fname, expected in payloads.items():
        assert (dest_dir / fname).read_bytes() == expected


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_zero_gpu_chunk_bypass_skips_empty_windows_and_passes_active_frame_qa(
    job_repo, settings, tmp_path, monkeypatch
):
    # 17 frames @ chunk_frames=9 -> Chunk(0, 9, 0) and Chunk(8, 9, 1).
    # Put the moving overlay ONLY in frames 0..5 so Chunk(8, 9, 1) has zero active mask frames!
    source_path = _write_textured_video(
        settings.inbox_dir / "intermittent.mp4",
        frames=17,
        size=(160, 120),
        fps=10,
        logo_box=(10, 10, 44, 24),
        active_range=(0, 5),
    )
    clean_chunk0 = _write_textured_video(
        tmp_path / "clean_chunk0.mp4",
        frames=9,
        size=(160, 120),
        fps=10,
        logo_box=None,
    )
    clean_chunk0_bytes = clean_chunk0.read_bytes()

    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="minimax_remover",
            fallback_chain_json=JobRepository.default_fallback_chain("minimax_remover"),
            input_path=str(source_path),
        )
    )
    work_dir = settings.work_dir / job.job_id
    normalized = normalize_input(source_path, work_dir)
    union_mask = np.zeros((120, 160), dtype=bool)
    union_mask[10:34, 10:54] = True
    active_flags = tuple(idx < 5 for idx in range(17))
    mask_path = work_dir / "mask.mkv"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_bytes(b"stub")

    detection = DetectionResult(
        scene_type="moving_overlay",
        confidence_level="high",
        mask_ratio=round(float(union_mask.mean()), 6),
        region_count=1,
        mask_video_path=mask_path,
        message="SAM3 text mask detected",
        bbox=(10, 10, 44, 24),
        frame_size=(160, 120),
        temporal=True,
        detector="sam3_text",
        regions=((10, 10, 44, 24),),
        union_mask=union_mask,
        active_frames=active_flags,
    )
    chunks = plan_chunks(17, 9)
    assert len(chunks) == 2
    context = PipelineContext(job=job, work_dir=work_dir, source=normalized, detection=detection, chunks=chunks)

    prompt_calls = {"count": 0}

    class _Resp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class _MockComfyClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout=None, params=None):
            if url.endswith("/system_stats"):
                return _Resp({"devices": [{"type": "cuda"}]})
            prompt_id = url.rsplit("/", 1)[-1]
            return _Resp(
                {
                    prompt_id: {
                        "outputs": {
                            "9": {"videos": [{"filename": "inpainted_0001.mp4", "subfolder": "video", "type": "output"}]}
                        },
                        "status": {"status_str": "success", "completed": True, "messages": []},
                    }
                }
            )

        def post(self, url, json=None):
            assert url.endswith("/prompt")
            prompt_calls["count"] += 1
            return _Resp({"prompt_id": json["prompt_id"]})

        def stream(self, method, url, params=None):
            class _Stream:
                status_code = 200
                text = ""

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def iter_bytes(self):
                    yield clean_chunk0_bytes

            return _Stream()

    monkeypatch.setattr("wm_platform.provider_runtime.httpx.Client", _MockComfyClient)
    output_path, device = run_comfy_chunks(
        replace(settings, chunk_frames=9),
        context,
        "minimax_remover",
        lambda _dev: {
            "1": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": str(source_path)}},
            "9": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/out"}},
        },
    )
    assert device == "cuda"
    assert prompt_calls["count"] == 1
    assert context.report["plan"] == {
        "chunks": 2,
        "chunk_frames": 9,
        "inpainted_chunks": 1,
        "bypassed_chunks": 1,
    }
    assert probe_video(output_path).frame_count == 17
    verdict = check_output(normalized.path, output_path, detection)
    assert verdict.status == "pass", verdict.as_report()
    shutil.rmtree(work_dir, ignore_errors=True)


def test_fastapi_version_and_eval_settings_isolation(settings):
    app = create_app()
    assert app.version == "0.3.0"

    eval_settings = build_eval_settings(settings)
    try:
        assert eval_settings.work_dir != settings.work_dir
        assert eval_settings.comfyui_output_dir != settings.comfyui_output_dir
        assert eval_settings.work_dir.is_relative_to(eval_settings.storage_root)
        assert eval_settings.comfyui_output_dir.is_relative_to(eval_settings.storage_root)
    finally:
        shutil.rmtree(eval_settings.storage_root, ignore_errors=True)
