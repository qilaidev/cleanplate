from __future__ import annotations

import fcntl
import ipaddress
import logging
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from wm_platform.config import Settings

logger = logging.getLogger(__name__)


def is_loopback_url(api_url: str) -> bool:
    """Return True when `api_url` targets localhost or an IPv4/IPv6 loopback literal."""
    host = (urlparse(api_url).hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    return bool(addr.is_loopback or (mapped is not None and mapped.is_loopback))


def open_comfy_http_client(client_cls: Any, api_url: str, *, timeout: float | None = None):
    """Instantiate an HTTP client, bypassing HTTP_PROXY/HTTPS_PROXY on loopback targets."""
    if client_cls is httpx.Client:
        return client_cls(timeout=timeout, trust_env=not is_loopback_url(api_url))
    return client_cls(timeout=timeout)


def comfy_http_get(get_fn: Any, url: str, *, timeout: float = 2.0):
    """Perform a GET probe against ComfyUI, bypassing env proxies on loopback targets."""
    if get_fn is httpx.get:
        return get_fn(url, timeout=timeout, trust_env=not is_loopback_url(url))
    return get_fn(url, timeout=timeout)


def comfy_python(settings: Settings) -> Path:
    return settings.comfyui_venv_dir / "bin" / "python"


def comfy_main(settings: Settings) -> Path:
    return settings.comfyui_dir / "main.py"


def build_comfyui_command(settings: Settings) -> list[str]:
    return [
        str(comfy_python(settings)),
        str(comfy_main(settings)),
        "--listen",
        "127.0.0.1",
        "--port",
        str(_port_from_api_url(settings.comfyui_api_url)),
        "--output-directory",
        str(settings.comfyui_output_dir),
        "--disable-auto-launch",
        "--disable-api-nodes",
    ]


def comfyui_health(settings: Settings) -> dict[str, object]:
    url = f"{settings.comfyui_api_url.rstrip('/')}/system_stats"
    try:
        response = comfy_http_get(httpx.get, url, timeout=2.0)
        response.raise_for_status()
        payload = response.json()
        return {
            "ok": True,
            "url": url,
            "status_code": response.status_code,
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error": str(exc),
        }


def wait_for_comfyui(settings: Settings, timeout_seconds: float = 60.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        health = comfyui_health(settings)
        if bool(health.get("ok")):
            return health
        time.sleep(1.0)
    return comfyui_health(settings)


def start_comfyui(settings: Settings) -> subprocess.Popen[str]:
    """Start ComfyUI with file lock to prevent race conditions.

    When multiple workers try to start ComfyUI simultaneously, only one
    will actually start it. Others will wait for the first one to finish.
    """
    lock_dir = settings.runtime_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "comfyui_startup.lock"

    with open(lock_file, "w") as f:
        try:
            # Try to acquire exclusive lock without blocking
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process is already starting ComfyUI
            logger.info("Another process is starting ComfyUI, waiting...")
            # Wait for the other process to finish
            result = wait_for_comfyui(settings, timeout_seconds=120.0)
            if result.get("ok"):
                logger.info("ComfyUI started by another process")
                # Return a dummy Popen object since ComfyUI is already running
                return subprocess.Popen(["sleep", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                raise RuntimeError(f"ComfyUI failed to start by another process: {result.get('error')}") from None

        process: subprocess.Popen[str] | None = None
        try:
            health = comfyui_health(settings)
            if health.get("ok"):
                logger.info("ComfyUI already running")
                return subprocess.Popen(["sleep", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            settings.comfyui_output_dir.mkdir(parents=True, exist_ok=True)
            command = build_comfyui_command(settings)
            logger.info("Starting ComfyUI: %s", " ".join(command))
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
            ready = wait_for_comfyui(settings, timeout_seconds=120.0)
            if not bool(ready.get("ok")):
                process.terminate()
                raise RuntimeError(f"ComfyUI failed to become healthy: {ready.get('error')}")
            return process
        except Exception:
            if process is not None and process.poll() is None:
                process.terminate()
            raise
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _port_from_api_url(api_url: str) -> int:
    parsed = urlparse(api_url.strip())
    try:
        if parsed.port is not None:
            return int(parsed.port)
    except ValueError:
        pass
    stripped = api_url.rstrip("/").rsplit(":", 1)
    if len(stripped) == 2 and stripped[1].isdigit():
        return int(stripped[1])
    return 8188
