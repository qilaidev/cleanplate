from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wm_platform.comfy_jobs import ComfyJobClient
from wm_platform.doctor import production_safety_report
from wm_platform.errors import AppError
from wm_platform.eval_fixtures import generate_eval_fixtures
from wm_platform.eval_judge import apply_judgment
from wm_platform.eval_runner import build_eval_report, load_eval_cases, run_eval, write_synthetic_logo_video
from wm_platform.models import JobCreate, JobResponse
from wm_platform.provider_runtime import ProviderRuntime
from wm_platform.repository import JobRepository
from wm_platform.runtime_contract import (
    expected_model_entries,
    huggingface_download_filename,
    model_destination,
    parse_huggingface_source,
)
from wm_platform.runtime_installer import RuntimeInstaller
from wm_platform.watermark_detect import (
    detect_static_overlay,
    detect_watermark,
    sam3_detector_status,
    sam3_workflow_path,
)
from wm_platform.worker_service import WorkerService


def _write_subject_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (160, 120))
    assert writer.isOpened()
    try:
        for index in range(8):
            frame = np.full((120, 160, 3), 30, dtype=np.uint8)
            x = 20 + index * 8
            cv2.rectangle(frame, (x, 10), (x + 90, 110), (180, 140, 120), -1)
            cv2.circle(frame, (x + 30, 40), 12, (40, 40, 200), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_huggingface_download_filename_uses_blob_and_explicit():
    dest = Path("sd-vae-ft-mse.safetensors")
    blob = parse_huggingface_source(
        "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors"
    )
    assert blob is not None
    assert huggingface_download_filename({}, blob, dest) == "diffusion_pytorch_model.safetensors"
    tree = parse_huggingface_source("https://huggingface.co/wangfuyun/PCM_Weights/tree/main/sd15")
    assert tree is not None
    assert (
        huggingface_download_filename({}, tree, Path("pcm_sd15_smallcfg_2step_converted.safetensors"))
        == "sd15/pcm_sd15_smallcfg_2step_converted.safetensors"
    )
    assert (
        huggingface_download_filename(
            {"filename": "diffusion_pytorch_model.safetensors"},
            parse_huggingface_source("https://huggingface.co/stabilityai/sd-vae-ft-mse"),
            dest,
        )
        == "diffusion_pytorch_model.safetensors"
    )
    assert huggingface_download_filename({"filename": "keep-me.safetensors"}, tree, dest) == "sd15/keep-me.safetensors"


def test_parse_huggingface_source_accepts_url_and_repo_id():
    parsed = parse_huggingface_source("https://huggingface.co/zibojia/minimax-remover")
    assert parsed == {"repo_id": "zibojia/minimax-remover", "revision": None, "subpath": None}
    tree = parse_huggingface_source("https://huggingface.co/wangfuyun/PCM_Weights/tree/main/sd15")
    assert tree == {"repo_id": "wangfuyun/PCM_Weights", "revision": "main", "subpath": "sd15"}
    assert parse_huggingface_source("zibojia/minimax-remover")["repo_id"] == "zibojia/minimax-remover"
    assert parse_huggingface_source("https://github.com/sczhou/ProPainter/releases/tag/v0.1.0") is None


def test_runtime_installer_skips_models_by_default(settings, monkeypatch):
    installer = RuntimeInstaller(settings)
    monkeypatch.setattr(installer, "_ensure_runtime_venv", lambda: ["venv"])
    monkeypatch.setattr(installer, "_ensure_repo", lambda name, spec: [f"repo {name}"])
    monkeypatch.setattr(installer, "_install_python_packages", lambda: ["pkgs"])
    called = {"models": False}

    def _models() -> list[str]:
        called["models"] = True
        return ["downloaded"]

    monkeypatch.setattr(installer, "_ensure_models", _models)
    messages = installer.install()
    assert called["models"] is False
    assert any("skipped model download" in item for item in messages)


def test_runtime_installer_downloads_huggingface_targets(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    (runtime / "models" / "manifest.yaml").write_text(
        """
models:
  - name: demo-dir
    required: true
    expected_path: .runtime/ComfyUI/models/MiniMax-Remover
    source: https://huggingface.co/zibojia/minimax-remover
  - name: demo-file
    required: false
    expected_path: .runtime/ComfyUI/models/vae/demo.safetensors
    source: org/weights
  - name: github-only
    required: false
    expected_path: .runtime/ComfyUI/models/DiffuEraser/propainter
    source: https://github.com/sczhou/ProPainter/releases/tag/v0.1.0
""",
        encoding="utf-8",
    )
    isolated = replace(settings, repo_root=repo, runtime_root=runtime)
    installer = RuntimeInstaller(isolated)

    def _snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "vae").mkdir()
        (dest / "transformer").mkdir()
        (dest / "scheduler").mkdir()
        return str(dest)

    def _file(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        path = local_dir / Path(str(kwargs["filename"])).name
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr(installer, "_huggingface_fns", lambda: (_file, _snapshot))
    messages = installer.install(include_repos=False, include_python_packages=False, include_models=True)
    dest_dir = repo / ".runtime/ComfyUI/models/MiniMax-Remover"
    dest_file = repo / ".runtime/ComfyUI/models/vae/demo.safetensors"
    assert dest_dir.joinpath("vae").exists()
    assert dest_file.is_file()
    assert any("installed model demo-dir" in item for item in messages)
    assert any("github-only" in item for item in messages)


def test_runtime_installer_required_failure_is_visible(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    (runtime / "models" / "manifest.yaml").write_text(
        """
models:
  - name: must-have
    required: true
    expected_path: .runtime/ComfyUI/models/MiniMax-Remover
    source: https://huggingface.co/zibojia/minimax-remover
""",
        encoding="utf-8",
    )
    installer = RuntimeInstaller(replace(settings, repo_root=repo, runtime_root=runtime))

    def _missing_hub():
        raise RuntimeError("huggingface_hub is required to download models")

    monkeypatch.setattr(installer, "_huggingface_fns", _missing_hub)
    with pytest.raises(RuntimeError, match="required models not installed") as exc:
        installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert "must-have" in str(exc.value)
    assert not (repo / ".runtime/ComfyUI/models/MiniMax-Remover").exists()


def test_runtime_paths_follow_remapped_runtime_root(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    bundled = repo / ".runtime"
    (bundled / "models").mkdir(parents=True)
    (bundled / "lock.yaml").write_text("runtime:\n  python: '3.12'\n", encoding="utf-8")
    (bundled / "models" / "manifest.yaml").write_text(
        """
models:
  - name: minimax-remover
    required: true
    providers: [minimax_remover]
    expected_path: .runtime/ComfyUI/models/MiniMax-Remover
    source: https://huggingface.co/zibojia/minimax-remover
""",
        encoding="utf-8",
    )
    remapped = tmp_path / "data" / "runtime"
    remapped.mkdir(parents=True)
    isolated = replace(
        settings,
        repo_root=repo,
        runtime_root=remapped,
        comfyui_dir=remapped / "ComfyUI",
        comfyui_models_dir=remapped / "ComfyUI" / "models",
    )
    dest = model_destination(isolated, expected_model_entries(isolated)[0])
    assert dest == (remapped / "ComfyUI" / "models" / "MiniMax-Remover").resolve()

    def _snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "vae").mkdir()
        (target / "transformer").mkdir()
        (target / "scheduler").mkdir()
        return str(target)

    installer = RuntimeInstaller(isolated)
    monkeypatch.setattr(installer, "_huggingface_fns", lambda: (_unused_file_download, _snapshot))
    installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert dest.joinpath("vae").exists()
    assert (remapped / "models" / "manifest.yaml").is_file()
    assert dest.is_relative_to(remapped.resolve())


def _unused_file_download(**kwargs):
    raise AssertionError(f"unexpected file download: {kwargs}")


def test_models_install_continues_when_propainter_github_fails(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    (runtime / "models" / "manifest.yaml").write_text(
        """
models:
  - name: minimax-remover
    required: true
    providers: [minimax_remover]
    expected_path: .runtime/ComfyUI/models/MiniMax-Remover
    source: https://huggingface.co/zibojia/minimax-remover
  - name: propainter
    required: true
    providers: [comfy_diffueraser]
    expected_path: .runtime/ComfyUI/models/DiffuEraser/propainter
    source: https://github.com/sczhou/ProPainter/releases/tag/v0.1.0
""",
        encoding="utf-8",
    )
    isolated = replace(settings, repo_root=repo, runtime_root=runtime)
    installer = RuntimeInstaller(isolated)

    def _snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "vae").mkdir()
        (target / "transformer").mkdir()
        (target / "scheduler").mkdir()
        return str(target)

    monkeypatch.setattr(installer, "_huggingface_fns", lambda: (_unused_file_download, _snapshot))
    messages = installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert (repo / ".runtime/ComfyUI/models/MiniMax-Remover/vae").exists()
    assert any("propainter" in item for item in messages)
    assert any("non-blocking required model propainter" in item for item in messages)


def test_dest_missing_after_download_is_required_failure(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    (runtime / "models" / "manifest.yaml").write_text(
        """
models:
  - name: must-have
    required: true
    providers: [minimax_remover]
    expected_path: .runtime/ComfyUI/models/MiniMax-Remover
    source: https://huggingface.co/zibojia/minimax-remover
""",
        encoding="utf-8",
    )
    installer = RuntimeInstaller(replace(settings, repo_root=repo, runtime_root=runtime))

    def _snapshot(**kwargs):
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["local_dir"]) / ".cache").mkdir()
        return str(kwargs["local_dir"])

    monkeypatch.setattr(installer, "_huggingface_fns", lambda: (_unused_file_download, _snapshot))
    with pytest.raises(RuntimeError, match="required models not installed") as exc:
        installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert "dest missing" in str(exc.value)
    assert any("dest missing after download" in item for item in getattr(exc.value, "messages", []))


def test_unsafe_huggingface_subpath_is_rejected(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    (runtime / "models" / "manifest.yaml").write_text(
        """
models:
  - name: sneaky
    required: false
    expected_path: .runtime/ComfyUI/models/vae/demo.safetensors
    source: https://huggingface.co/org/repo/resolve/main/../../../etc/passwd
""",
        encoding="utf-8",
    )
    installer = RuntimeInstaller(replace(settings, repo_root=repo, runtime_root=runtime))
    monkeypatch.setattr(installer, "_huggingface_fns", lambda: (_unused_file_download, _unused_file_download))
    messages = installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert any("unsafe huggingface" in item for item in messages)
    assert not (repo / ".runtime/ComfyUI/models/vae/demo.safetensors").exists()


def test_eval_runner_empty_split_writes_empty_results(tmp_path, settings, job_repo):
    (tmp_path / "simple").mkdir()
    payload = run_eval(
        dataset_root=tmp_path,
        split="simple",
        providers=["auto"],
        settings=settings,
        repository=job_repo,
        results_dir=tmp_path / "results",
    )
    assert payload["results"] == []
    assert "no videos found" in str(payload["message"])
    written = list((tmp_path / "results").glob("*-auto.json"))
    assert len(written) == 1
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert body["results"] == []


def test_eval_runner_records_local_fallback_case(tmp_path, settings, job_repo):
    simple = tmp_path / "simple"
    simple.mkdir()
    write_synthetic_logo_video(simple / "corner_logo_fixture.mp4")
    payload = run_eval(
        dataset_root=tmp_path,
        split="simple",
        providers=["local_fallback"],
        settings=settings,
        repository=job_repo,
        results_dir=tmp_path / "results",
    )
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    assert row["status"] == "succeeded"
    assert row["provider_requested"] == "local_fallback"
    assert row["provider_selected"] == "local_fallback"
    assert row["job_id"]
    assert row["judgment"] == "pending"
    assert "mask_ratio" in row
    assert row["mask_ratio"] is None


def test_eval_manifest_skips_missing_videos(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "manifest.csv").write_text(
        "id,split,filename,watermark_type,position,moving,notes,expected\n"
        "missing,simple,nope.mp4,static_logo,top_left,false,placeholder,pass\n",
        encoding="utf-8",
    )
    assert load_eval_cases(tmp_path, "simple") == []


def test_eval_unions_manifest_and_directory_videos(tmp_path):
    simple = tmp_path / "simple"
    simple.mkdir()
    extra = write_synthetic_logo_video(simple / "dropped.mp4")
    (tmp_path / "manifest.csv").write_text(
        "id,split,filename,watermark_type,position,moving,notes,expected\n"
        "missing,simple,nope.mp4,static_logo,top_left,false,placeholder,pass\n",
        encoding="utf-8",
    )
    cases = load_eval_cases(tmp_path, "simple")
    assert [case.path.resolve() for case in cases] == [extra.resolve()]


def test_static_overlay_detector_rejects_full_frame_subject(tmp_path):
    source = _write_subject_video(tmp_path / "subject.mp4")
    result = detect_static_overlay(source, tmp_path / "mask.mp4")
    assert result.confidence_level == "low"
    assert result.mask_video_path is None


def test_detector_auto_uses_static_when_confident(tmp_path, settings):
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", replace(settings, detector_mode="auto"))
    assert result.confidence_level in {"high", "medium"}
    assert result.mask_video_path is not None


def test_detector_sam3_text_unavailable_without_runtime(tmp_path, settings):
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    isolated = replace(
        settings,
        detector_mode="sam3_text",
        comfyui_custom_nodes_dir=tmp_path / "custom_nodes",
        comfyui_models_dir=tmp_path / "models",
        comfyui_workflows_dir=tmp_path / "workflows",
    )
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level == "low"
    assert result.mask_video_path is None
    assert "not available" in result.message.lower()


def test_detector_auto_does_not_invent_mask(tmp_path, settings):
    source = _write_subject_video(tmp_path / "subject.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", replace(settings, detector_mode="auto"))
    assert result.confidence_level == "low"
    assert result.mask_video_path is None


def test_minimax_run_uses_detector_policy(job_repo, settings, tmp_path, monkeypatch):
    called = {"detect": False}

    def _fake_detect(*args, **kwargs):
        called["detect"] = True
        from wm_platform.watermark_detect import DetectionResult

        return DetectionResult(
            scene_type="unknown",
            confidence_level="low",
            mask_ratio=0.0,
            region_count=0,
            mask_video_path=None,
            message="policy rejected",
        )

    monkeypatch.setattr("wm_platform.provider_runtime.detect_watermark", _fake_detect)
    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="minimax_remover",
            fallback_chain_json=JobRepository.default_fallback_chain("minimax_remover"),
            input_path=str(write_synthetic_logo_video(settings.inbox_dir / "policy.mp4")),
        )
    )
    provider = ProviderRuntime(settings, repository=job_repo).registry["minimax_remover"]
    with pytest.raises(AppError) as exc:
        provider.run(job)
    assert exc.value.error_code == "DETECTOR_LOW_CONFIDENCE"
    assert called["detect"] is True



def test_callback_payload_includes_degraded(job_repo, settings):
    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="local_fallback",
            fallback_chain_json=JobRepository.default_fallback_chain("local_fallback"),
            input_path=str(write_synthetic_logo_video(settings.inbox_dir / "callback-degraded.mp4")),
            callback_url="http://callback.local/notify",
        )
    )
    assert WorkerService(settings=settings, repository=job_repo, providers=ProviderRuntime(settings, repository=job_repo)).run_once()
    outbox = job_repo.get_callback_outbox(job.job_id)
    assert outbox
    payload = json.loads(outbox[0].payload_json)
    assert payload["provider"] == "local_fallback"
    assert payload["degraded"] is True


def test_production_safety_reports_low_disk(settings, monkeypatch):
    class _Usage:
        def __init__(self) -> None:
            self.total = 100 * 1024 * 1024 * 1024
            self.free = 1 * 1024 * 1024 * 1024

    monkeypatch.setattr("wm_platform.doctor.shutil.disk_usage", lambda path: _Usage())
    report = production_safety_report(settings)
    assert any("low on space" in item for item in report["findings"])
    assert report["ready"] is False


def test_sam3_files_present_does_not_invent_mask(tmp_path, settings):
    isolated = replace(_prepare_sam3_runtime(tmp_path, settings), comfyui_api_url="http://127.0.0.1:1")
    status = sam3_detector_status(isolated)
    assert status["available"] is True
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level == "low"
    assert result.mask_video_path is None
    auto = detect_watermark(
        _write_subject_video(tmp_path / "subject.mp4"),
        tmp_path / "auto-mask.mp4",
        replace(isolated, detector_mode="auto"),
    )
    assert auto.confidence_level == "low"
    assert auto.mask_video_path is None


def test_auto_detector_low_confidence_does_not_run_diffueraser(job_repo, settings, monkeypatch):
    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="auto",
            fallback_chain_json=JobRepository.default_fallback_chain("auto"),
            input_path=str(_write_subject_video(settings.inbox_dir / "auto-subject.mp4")),
        )
    )
    runtime = ProviderRuntime(settings, repository=job_repo)
    monkeypatch.setattr(
        runtime.registry["minimax_remover"],
        "probe",
        lambda: SimpleNamespace(runnable=True, message="ready"),
    )
    monkeypatch.setattr(
        runtime.registry["comfy_diffueraser"],
        "probe",
        lambda: SimpleNamespace(runnable=True, message="ready"),
    )
    called = {"diffu": False}

    def _diffu_run(job_record):
        called["diffu"] = True
        return {"output_path": str(settings.outbox_dir / "should-not.mp4")}

    monkeypatch.setattr(runtime.registry["comfy_diffueraser"], "run", _diffu_run)
    service = WorkerService(settings=settings, repository=job_repo, providers=runtime)
    assert service.run_once() is True
    updated = job_repo.get_job(job.job_id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error_code == "DETECTOR_LOW_CONFIDENCE"
    assert JobResponse.from_record(updated).degraded is False
    assert called["diffu"] is False


def test_sam3_status_reports_missing_node(settings, tmp_path):
    status = sam3_detector_status(
        replace(
            settings,
            comfyui_custom_nodes_dir=tmp_path / "custom_nodes",
            comfyui_models_dir=tmp_path / "models",
        )
    )
    assert status["available"] is False
    assert any("native nodes" in str(item) for item in status["issues"])
    assert any("checkpoint" in str(item) for item in status["issues"])


def _write_mask_png(path: Path, *, full: bool = False) -> bytes:
    image = np.full((120, 160), 255, dtype=np.uint8) if full else np.zeros((120, 160), dtype=np.uint8)
    if not full:
        image[4:28, 4:48] = 255
    assert cv2.imwrite(str(path), image)
    return path.read_bytes()


def fake_video_bytes(tmp_path: Path) -> bytes:
    return write_synthetic_logo_video(tmp_path / "fake-comfy-output.mp4").read_bytes()


def _fake_comfy_client(history_filename: str, content: bytes, captured: dict[str, object] | None = None):
    class _Response:
        def __init__(self, status_code=200, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.text or f"http {self.status_code}")

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout=None, params=None):
            if str(url).endswith("/system_stats"):
                return _Response(payload={"devices": [{"type": "cpu"}]})
            if "/history/" in str(url):
                prompt_id = str(url).rstrip("/").split("/")[-1]
                return _Response(
                    payload={
                        prompt_id: {
                            "outputs": {
                                "7": {
                                    "images": [
                                        {
                                            "filename": history_filename,
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            },
                            "status": {"status_str": "success", "completed": True, "messages": []},
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def post(self, url, json=None):
            assert str(url).endswith("/prompt")
            if captured is not None:
                captured.update(json or {})
            return _Response(payload={"prompt_id": (json or {}).get("prompt_id", "prompt")})

        def stream(self, method, url, params=None):
            assert method == "GET"
            assert str(url).endswith("/view")

            class _StreamResponse:
                status_code = 200
                text = ""

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def iter_bytes(self):
                    yield content

            return _StreamResponse()

    return _FakeClient


def _prepare_sam3_runtime(tmp_path: Path, settings):
    """Native SAM 3.1 stack: ComfyUI with comfy_extras/nodes_sam3.py plus the multiplex checkpoint."""
    comfy = tmp_path / "ComfyUI"
    (comfy / "comfy_extras").mkdir(parents=True)
    (comfy / "comfy_extras" / "nodes_sam3.py").write_text("# stub", encoding="utf-8")
    (comfy / "custom_nodes").mkdir()
    checkpoints = comfy / "models" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "sam3.1_multiplex_fp16.safetensors").write_bytes(b"stub")
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    shutil.copy(settings.repo_root / "workflows" / "sam3_native_api.json", workflows / "sam3_native_api.json")
    return replace(
        settings,
        detector_mode="sam3_text",
        comfyui_dir=comfy,
        comfyui_custom_nodes_dir=comfy / "custom_nodes",
        comfyui_models_dir=comfy / "models",
        comfyui_workflows_dir=workflows,
        comfyui_sam3_workflow=workflows / "sam3_native_api.json",
        auto_start_comfyui=False,
    )


def test_sam3_mock_comfy_writes_corner_mask(tmp_path, settings, monkeypatch):
    isolated = _prepare_sam3_runtime(tmp_path, settings)
    mask_bytes = _write_mask_png(tmp_path / "corner-mask.png")
    captured: dict[str, object] = {}
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", _fake_comfy_client("mask.png", mask_bytes, captured))
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level != "low"
    assert result.mask_video_path is not None
    assert result.mask_video_path.exists()
    prompt = captured["prompt"]
    assert prompt["1"]["inputs"]["video"] == str(source)
    assert prompt["4"]["inputs"]["text"] == isolated.watermark_detect_prompt
    assert prompt["3"]["inputs"]["ckpt_name"] == "sam3.1_multiplex_fp16.safetensors"
    assert prompt["9"]["class_type"] == "SaveVideo"


def test_sam3_unreachable_comfy_does_not_invent_mask(tmp_path, settings):
    isolated = replace(_prepare_sam3_runtime(tmp_path, settings), comfyui_api_url="http://127.0.0.1:1")
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level == "low"
    assert result.mask_video_path is None


def test_sam3_oversized_mask_is_rejected(tmp_path, settings, monkeypatch):
    isolated = _prepare_sam3_runtime(tmp_path, settings)
    mask_bytes = _write_mask_png(tmp_path / "full-mask.png", full=True)
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", _fake_comfy_client("mask.png", mask_bytes))
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level == "low"
    assert result.scene_type == "complex"
    assert result.mask_video_path is None


def test_eval_fixtures_write_splits_and_manifest(tmp_path):
    written = generate_eval_fixtures(tmp_path)
    assert len(written) == 7
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    cases = load_eval_cases(tmp_path, "all")
    assert {case.split for case in cases} == {"simple", "medium", "hard"}
    assert len(cases) == 7
    by_name = {path.name: path for path in written}
    for name in ("simple_top_left.mp4", "simple_bottom_right.mp4", "medium_translucent.mp4"):
        result = detect_static_overlay(by_name[name], tmp_path / f"{name}.mask.mp4")
        assert result.confidence_level in {"high", "medium"}, name
        assert result.mask_video_path is not None
        assert result.bbox is not None
    for name in ("medium_drift.mp4",):
        result = detect_static_overlay(by_name[name], tmp_path / f"{name}.mask.mp4")
        assert result.confidence_level == "low"
        assert result.mask_video_path is None
    clean_dir = tmp_path / "clean"
    assert sorted(path.name for path in clean_dir.glob("*.mkv")) == sorted(
        Path(path.name).with_suffix(".mkv").name for path in written
    )
    assert len(list(clean_dir.glob("*.region.png"))) == len(written)
    assert all(case.clean_path is not None and case.clean_path.is_file() for case in cases)
    assert all(case.region_path is not None and case.region_path.is_file() for case in cases)
    for name in ("hard_large.mp4", "hard_multi.mp4", "hard_center.mp4"):
        result = detect_static_overlay(by_name[name], tmp_path / f"{name}.mask.mp4")
        assert result.confidence_level == "low"
        assert result.mask_video_path is None


def test_sam3_queues_configured_workflow(tmp_path, settings, monkeypatch):
    isolated = _prepare_sam3_runtime(tmp_path, settings)
    default_workflow = isolated.comfyui_workflows_dir / "sam3_native_api.json"
    custom = tmp_path / "custom_sam3.json"
    template = json.loads(default_workflow.read_text(encoding="utf-8"))
    template["5"]["inputs"]["detection_threshold"] = 0.91
    custom.write_text(json.dumps(template), encoding="utf-8")
    isolated = replace(isolated, comfyui_sam3_workflow=custom)
    assert sam3_workflow_path(isolated) == custom.resolve()
    captured: dict[str, object] = {}
    mask_bytes = _write_mask_png(tmp_path / "custom-mask.png")
    monkeypatch.setattr("wm_platform.comfy_jobs.httpx.Client", _fake_comfy_client("mask.png", mask_bytes, captured))
    source = write_synthetic_logo_video(tmp_path / "logo.mp4")
    result = detect_watermark(source, tmp_path / "mask.mp4", isolated)
    assert result.confidence_level != "low"
    assert captured["prompt"]["5"]["inputs"]["detection_threshold"] == 0.91


def test_mask_artifact_waits_until_completed(settings, monkeypatch):
    calls = {"n": 0}

    class _Response:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    class _Client:
        def get(self, url, timeout=None, params=None):
            calls["n"] += 1
            images = [{"filename": "preview.png", "subfolder": "", "type": "output"}]
            completed = calls["n"] > 1
            return _Response(
                {
                    "sam3-wait": {
                        "outputs": {"7": {"images": images}},
                        "status": {"status_str": "success", "completed": completed, "messages": []},
                    }
                }
            )

    monkeypatch.setattr("wm_platform.comfy_jobs.time.sleep", lambda _seconds: None)
    artifact = ComfyJobClient(settings).wait_for_artifact(_Client(), "sam3-wait", allow_images=True)
    assert calls["n"] == 2
    assert artifact["filename"] == "preview.png"


def test_mask_artifact_rejects_completed_error_preview(settings, monkeypatch):
    class _Response:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    class _Client:
        def get(self, url, timeout=None, params=None):
            return _Response(
                {
                    "sam3-error": {
                        "outputs": {
                            "7": {
                                "images": [
                                    {"filename": "preview.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                        "status": {
                            "status_str": "error",
                            "completed": True,
                            "messages": ["node failed"],
                        },
                    }
                }
            )

    monkeypatch.setattr("wm_platform.comfy_jobs.time.sleep", lambda _seconds: None)
    with pytest.raises(AppError) as exc:
        ComfyJobClient(settings).wait_for_artifact(_Client(), "sam3-error", allow_images=True)
    assert exc.value.error_code == "PROVIDER_RUN_FAILED"
    assert "failed" in exc.value.error_message.lower()


def test_mask_video_matches_source_frame_count(tmp_path):
    source = write_synthetic_logo_video(tmp_path / "logo.mp4", frames=12, fps=8)
    result = detect_static_overlay(source, tmp_path / "mask.mp4")
    assert result.mask_video_path is not None
    source_frames = 0
    capture = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_frames += 1
    finally:
        capture.release()
    mask_frames = 0
    capture = cv2.VideoCapture(str(result.mask_video_path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            mask_frames += 1
    finally:
        capture.release()
    assert source_frames == 12
    assert mask_frames == source_frames


def test_eval_report_separates_local_fallback_from_ai_success(tmp_path, settings, job_repo):
    generate_eval_fixtures(tmp_path)
    payload = run_eval(
        dataset_root=tmp_path,
        split="simple",
        providers=["local_fallback"],
        settings=settings,
        repository=job_repo,
        results_dir=tmp_path / "results",
    )
    report = build_eval_report(payload)
    summary = report["providers"]["local_fallback"]
    assert summary["succeeded"] >= 1
    assert summary["ai_success"] == 0
    assert summary["degraded"] == summary["succeeded"]
    assert summary["local_fallback"] == summary["succeeded"]


def test_eval_runner_writes_judgment_overlay(tmp_path, settings, job_repo):
    simple = tmp_path / "simple"
    simple.mkdir()
    write_synthetic_logo_video(simple / "corner.mp4")
    judgments = tmp_path / "judgments.csv"
    judgments.write_text("id,judgment\nsimple-01,degraded\n", encoding="utf-8")
    payload = run_eval(
        dataset_root=tmp_path,
        split="simple",
        providers=["local_fallback"],
        settings=settings,
        repository=job_repo,
        results_dir=tmp_path / "results",
        judgments=judgments,
    )
    assert payload["results"][0]["judgment"] == "degraded"
    written = json.loads(Path(payload["written"][0]).read_text(encoding="utf-8"))
    assert written["results"][0]["judgment"] == "degraded"


def test_eval_judge_updates_report_counts(tmp_path, settings, job_repo):
    simple = tmp_path / "simple"
    simple.mkdir()
    write_synthetic_logo_video(simple / "corner_logo_fixture.mp4")
    payload = run_eval(
        dataset_root=tmp_path,
        split="simple",
        providers=["local_fallback"],
        settings=settings,
        repository=job_repo,
        results_dir=tmp_path / "results",
    )
    before = build_eval_report(payload)
    assert before["overall"]["judgments"]["pending"] == 1
    result_file = next((tmp_path / "results").glob("*-local_fallback.json"))
    apply_judgment(result_file, payload["results"][0]["id"], "degraded")
    updated = json.loads(result_file.read_text(encoding="utf-8"))
    after = build_eval_report(updated)
    assert after["overall"]["judgments"].get("degraded") == 1
    assert after["overall"]["judgments"].get("pending", 0) == 0


def test_eval_report_does_not_count_missing_ai_as_success():
    payload = {
        "date": "2026-08-17",
        "split": "simple",
        "results": [
            {
                "id": "simple-tl",
                "provider_requested": "minimax_remover",
                "provider_selected": None,
                "status": "failed",
                "error_code": "PROVIDER_NOT_AVAILABLE",
                "judgment": "pending",
            }
        ],
    }
    report = build_eval_report(payload)
    assert report["overall"]["ai_success"] == 0
    assert report["overall"]["succeeded"] == 0
    assert report["overall"]["failed"] == 1
    assert report["overall"]["error_codes"]["PROVIDER_NOT_AVAILABLE"] == 1


def test_diffueraser_without_mask_is_low_confidence(job_repo, settings):
    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="comfy_diffueraser",
            fallback_chain_json=JobRepository.default_fallback_chain("comfy_diffueraser"),
            input_path=str(_write_subject_video(settings.inbox_dir / "no-mask.mp4")),
        )
    )
    provider = ProviderRuntime(settings, repository=job_repo).registry["comfy_diffueraser"]
    with pytest.raises(AppError) as exc:
        provider.run(job)
    assert exc.value.error_code == "DETECTOR_LOW_CONFIDENCE"


def test_minimax_provider_runs_with_api_prompt(job_repo, settings, monkeypatch, tmp_path):
    nodes = tmp_path / "custom_nodes" / "ComfyUI-MiniMax-Remover"
    nodes.mkdir(parents=True)
    venv = tmp_path / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("", encoding="utf-8")
    models = tmp_path / "models" / "MiniMax-Remover"
    for name in ("vae", "transformer", "scheduler"):
        (models / name).mkdir(parents=True)
    isolated = replace(
        settings,
        comfyui_custom_nodes_dir=tmp_path / "custom_nodes",
        comfyui_venv_dir=tmp_path / ".venv",
        comfyui_models_dir=tmp_path / "models",
        comfyui_minimax_workflow=settings.repo_root / "workflows" / "minimax_remover_api.json",
        quality_mode="balanced",
        auto_start_comfyui=False,
    )
    source = write_synthetic_logo_video(settings.inbox_dir / "minimax-logo.mp4")
    job = job_repo.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested="minimax_remover",
            fallback_chain_json=JobRepository.default_fallback_chain("minimax_remover"),
            input_path=str(source),
            quality_mode="quality",
        )
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "wm_platform.provider_runtime.httpx.Client",
        _fake_comfy_client("video_00001_.mp4", fake_video_bytes(tmp_path), captured),
    )
    monkeypatch.setattr(
        "wm_platform.comfy_jobs.httpx.Client",
        _fake_comfy_client("video_00001_.mp4", fake_video_bytes(tmp_path), captured),
    )
    provider = ProviderRuntime(isolated, repository=job_repo).registry["minimax_remover"]
    result = provider.run(job)
    assert Path(result["output_path"]).stat().st_size > 0
    prompt = captured["prompt"]
    assert prompt["1"]["inputs"]["video"] == job.input_path
    assert Path(prompt["3"]["inputs"]["video"]).name == "mask.mkv"
    assert prompt["6"]["inputs"]["num_inference_steps"] == 12
    assert prompt["6"]["inputs"]["mask_dilation_iterations"] == 2
    assert prompt["8"]["class_type"] == "SaveVideo"
    assert prompt["8"]["inputs"]["filename_prefix"].startswith(f"video/{job.job_id}-minimax_remover-")


def test_secondary_hf_failure_does_not_abort(tmp_path, settings, monkeypatch):
    repo = tmp_path / "repo"
    runtime = repo / ".runtime"
    (runtime / "models").mkdir(parents=True)
    (runtime / "models" / "manifest.yaml").write_text(
        """
models:
  - name: minimax-remover
    required: true
    providers: [minimax_remover]
    expected_path: .runtime/ComfyUI/models/MiniMax-Remover
    source: https://huggingface.co/zibojia/minimax-remover
  - name: sd-vae-ft-mse
    required: true
    providers: [comfy_diffueraser]
    expected_path: .runtime/ComfyUI/models/vae/sd-vae-ft-mse.safetensors
    filename: diffusion_pytorch_model.safetensors
    source: https://huggingface.co/stabilityai/sd-vae-ft-mse
""",
        encoding="utf-8",
    )
    installer = RuntimeInstaller(replace(settings, repo_root=repo, runtime_root=runtime))

    def _snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "vae").mkdir()
        (target / "transformer").mkdir()
        (target / "scheduler").mkdir()
        return str(target)

    def _file(**kwargs):
        raise RuntimeError("hub unavailable")

    monkeypatch.setattr(installer, "_huggingface_fns", lambda: (_file, _snapshot))
    messages = installer.install(include_repos=False, include_python_packages=False, include_models=True)
    assert (repo / ".runtime/ComfyUI/models/MiniMax-Remover/vae").exists()
    assert any("sd-vae-ft-mse" in item for item in messages)
    assert any("non-blocking required model sd-vae-ft-mse" in item for item in messages)
