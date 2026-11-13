from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

from wm_platform.comfy_runtime import start_comfyui, wait_for_comfyui
from wm_platform.config import Settings
from wm_platform.errors import AppError
from wm_platform.models import JobRecord
from wm_platform.storage import build_output_path


class ComfyJobClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ensure_ready(self, client: httpx.Client) -> str:
        stats = self.system_stats(client)
        if stats is None and self.settings.auto_start_comfyui:
            start_comfyui(self.settings)
            health = wait_for_comfyui(self.settings)
            if not bool(health.get("ok")):
                raise AppError("PROVIDER_NOT_AVAILABLE", "ComfyUI API is unavailable after auto-start", 503)
            payload = health.get("payload")
            if isinstance(payload, dict):
                stats = payload
        if stats is None:
            raise AppError("PROVIDER_NOT_AVAILABLE", "ComfyUI API is unavailable", 503)

        devices = stats.get("devices")
        if isinstance(devices, list) and devices:
            device_type = str(devices[0].get("type", "")).lower()
            if device_type in {"cuda", "mps", "cpu"}:
                return device_type
        return "cpu"

    def system_stats(self, client: httpx.Client) -> dict[str, object] | None:
        try:
            response = client.get(urljoin(self.settings.comfyui_api_url, "/system_stats"), timeout=2.0)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def queue_prompt(self, client: httpx.Client, prompt: dict[str, dict[str, object]], prompt_id: str) -> str:
        response = client.post(
            urljoin(self.settings.comfyui_api_url, "/prompt"),
            json={"prompt": prompt, "prompt_id": prompt_id, "client_id": prompt_id},
        )
        if response.status_code != 200:
            detail = response.text[:1000] if response.text else "queue prompt failed"
            raise AppError("PROVIDER_RUN_FAILED", f"ComfyUI prompt rejected: {detail}", 500)
        payload = response.json()
        queued_prompt_id = str(payload.get("prompt_id", prompt_id))
        if not queued_prompt_id:
            raise AppError("PROVIDER_RUN_FAILED", "ComfyUI prompt response missing prompt_id", 500)
        return queued_prompt_id

    def wait_for_artifact(
        self,
        client: httpx.Client,
        prompt_id: str,
        *,
        allow_images: bool = False,
    ) -> dict[str, str]:
        # Covers queueing behind other workers' prompts on a shared ComfyUI as well as the run.
        deadline = time.monotonic() + max(60.0, float(self.settings.comfyui_prompt_timeout_seconds))
        extractor = extract_mask_artifact if allow_images else extract_video_artifact
        empty_detail = (
            "ComfyUI execution completed without mask output"
            if allow_images
            else "ComfyUI execution completed without video output"
        )
        while time.monotonic() < deadline:
            response = client.get(urljoin(self.settings.comfyui_api_url, f"/history/{prompt_id}"))
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict) and prompt_id in payload:
                    history_item = payload[prompt_id]
                    artifact = extractor(history_item)
                    status = history_item.get("status", {})
                    completed = isinstance(status, dict) and bool(status.get("completed"))
                    if allow_images and completed and isinstance(status, dict) and status.get("status_str") == "error":
                        messages = status.get("messages") or []
                        detail = "; ".join(str(item) for item in messages) or empty_detail
                        raise AppError("PROVIDER_RUN_FAILED", f"ComfyUI execution failed: {detail}", 500)
                    if artifact is not None and (completed or not allow_images):
                        return artifact
                    if completed:
                        messages = status.get("messages") or []
                        detail = "; ".join(str(item) for item in messages) or empty_detail
                        if status.get("status_str") == "error":
                            raise AppError("PROVIDER_RUN_FAILED", f"ComfyUI execution failed: {detail}", 500)
                        raise AppError("PROVIDER_RUN_FAILED", detail, 500)
            time.sleep(1.0)
        raise AppError("PROVIDER_RUN_FAILED", "ComfyUI execution timed out", 500)

    def download_to(self, client: httpx.Client, artifact: dict[str, str], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f"{destination.name}.tmp")
        try:
            with client.stream("GET", urljoin(self.settings.comfyui_api_url, "/view"), params=artifact) as response:
                if response.status_code != 200:
                    detail = response.text[:1000] if response.text else "artifact download failed"
                    raise AppError("PROVIDER_RUN_FAILED", f"failed to fetch ComfyUI artifact: {detail}", 500)
                with temp_path.open("wb") as output:
                    for chunk in response.iter_bytes():
                        if chunk:
                            output.write(chunk)
            temp_path.replace(destination)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        return destination

    def download_artifact(self, client: httpx.Client, artifact: dict[str, str], job: JobRecord) -> Path:
        output_path = build_output_path(job.job_id, artifact["filename"], self.settings)
        return self.download_to(client, artifact, output_path)

    def run_for_artifact(
        self,
        prompt: dict[str, dict[str, object]],
        prompt_id: str,
        destination: Path,
        *,
        allow_images: bool = False,
    ) -> Path:
        with httpx.Client(timeout=None) as client:
            self.ensure_ready(client)
            queued_id = self.queue_prompt(client, prompt, prompt_id)
            artifact = self.wait_for_artifact(client, queued_id, allow_images=allow_images)
            suffix = Path(artifact.get("filename", "")).suffix
            if suffix and destination.suffix.lower() != suffix.lower():
                destination = destination.with_suffix(suffix)
            return self.download_to(client, artifact, destination)


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _artifact_from_entry(entry: object) -> dict[str, str] | None:
    if not isinstance(entry, dict):
        return None
    filename = str(entry.get("filename", ""))
    if not filename:
        return None
    return {
        "filename": filename,
        "subfolder": str(entry.get("subfolder", "")),
        "type": str(entry.get("type", "output")),
    }


def _iter_output_entries(history_item: object):
    if not isinstance(history_item, dict):
        return
    outputs = history_item.get("outputs", {})
    if not isinstance(outputs, dict):
        return
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in ("images", "gifs", "videos"):
            entries = node_output.get(key, [])
            if not isinstance(entries, list):
                continue
            yield from entries


def extract_video_artifact(history_item: object) -> dict[str, str] | None:
    for entry in _iter_output_entries(history_item):
        artifact = _artifact_from_entry(entry)
        if artifact is None:
            continue
        if Path(artifact["filename"]).suffix.lower() in VIDEO_SUFFIXES:
            return artifact
    return None


def extract_mask_artifact(history_item: object) -> dict[str, str] | None:
    video = extract_video_artifact(history_item)
    if video is not None:
        return video
    for entry in _iter_output_entries(history_item):
        artifact = _artifact_from_entry(entry)
        if artifact is None:
            continue
        if Path(artifact["filename"]).suffix.lower() in IMAGE_SUFFIXES:
            return artifact
    return None
