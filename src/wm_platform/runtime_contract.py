from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from wm_platform.config import Settings

_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}
_MODEL_FILE_SUFFIXES = {".safetensors", ".pth", ".pt", ".ckpt", ".bin", ".onnx", ".pkl", ".npz"}
_RUNTIME_PREFIX = ".runtime"
_MINIMAX_CHILDREN = ("vae", "transformer", "scheduler")
PRIMARY_MODEL_PROVIDERS = frozenset({"minimax_remover"})


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def bundled_runtime_root(settings: Settings) -> Path:
    return (settings.repo_root / _RUNTIME_PREFIX).resolve()


def runtime_contract_path(settings: Settings, relative: str) -> Path:
    primary = settings.runtime_root / relative
    if primary.is_file():
        return primary
    bundled = bundled_runtime_root(settings) / relative
    if bundled.is_file() and bundled.resolve() != primary.resolve():
        return bundled
    return primary


def copy_runtime_contract_files(settings: Settings) -> list[str]:
    messages: list[str] = []
    bundled = bundled_runtime_root(settings)
    if settings.runtime_root.resolve() == bundled:
        return messages
    for relative in ("lock.yaml", "models/manifest.yaml"):
        source = bundled / relative
        destination = settings.runtime_root / relative
        if destination.exists() or not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        messages.append(f"copied {relative} into runtime_root")
    return messages


def resolve_runtime_relative(settings: Settings, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == _RUNTIME_PREFIX:
        return settings.runtime_root.joinpath(*parts[1:]).resolve()
    return (settings.repo_root / path).resolve()


def load_runtime_lock(settings: Settings) -> dict[str, Any]:
    return _read_yaml(runtime_contract_path(settings, "lock.yaml"))


def load_model_manifest(settings: Settings) -> dict[str, Any]:
    return _read_yaml(runtime_contract_path(settings, "models/manifest.yaml"))


def expected_repo_paths(settings: Settings) -> list[Path]:
    lock = load_runtime_lock(settings)
    repositories = lock.get("repositories", {})
    paths: list[Path] = []
    for spec in repositories.values():
        if isinstance(spec, dict) and spec.get("target"):
            paths.append(resolve_runtime_relative(settings, str(spec["target"])))
    return paths


def expected_model_entries(settings: Settings, provider: str | None = None) -> list[dict[str, Any]]:
    manifest = load_model_manifest(settings)
    entries = manifest.get("models", [])
    if not isinstance(entries, list):
        return []
    models = [item for item in entries if isinstance(item, dict)]
    if provider is None:
        return models
    filtered: list[dict[str, Any]] = []
    for item in models:
        declared = item.get("providers")
        if not declared or provider in declared:
            filtered.append(item)
    return filtered


def model_destination(settings: Settings, entry: dict[str, Any]) -> Path:
    raw = str(entry.get("expected_path", "")).strip()
    if not raw:
        return (settings.runtime_root / "models" / "unknown").resolve()
    return resolve_runtime_relative(settings, raw)


def is_model_file_target(entry: dict[str, Any]) -> bool:
    return Path(str(entry.get("expected_path", ""))).suffix.lower() in _MODEL_FILE_SUFFIXES


def is_weight_filename(value: str) -> bool:
    return Path(value).suffix.lower() in _MODEL_FILE_SUFFIXES


def model_required_children(entry: dict[str, Any]) -> tuple[str, ...]:
    declared = entry.get("must_contain") or entry.get("children")
    if isinstance(declared, list) and declared:
        return tuple(str(item) for item in declared if str(item).strip())
    dest = str(entry.get("expected_path", "")).rstrip("/")
    if dest.endswith("MiniMax-Remover"):
        return _MINIMAX_CHILDREN
    return ()


def model_is_present(entry: dict[str, Any], destination: Path | None = None) -> bool:
    dest = destination if destination is not None else Path(str(entry.get("expected_path", "")))
    if is_model_file_target(entry):
        return dest.is_file()
    if not dest.exists() or not dest.is_dir():
        return False
    children = model_required_children(entry)
    if children:
        return all((dest / name).exists() for name in children)
    return _directory_has_weights(dest)


def _directory_has_weights(dest: Path) -> bool:
    try:
        entries = list(dest.iterdir())
    except OSError:
        return False
    for item in entries:
        if item.name.startswith("."):
            continue
        if item.is_file() and item.suffix.lower() in _MODEL_FILE_SUFFIXES:
            return True
        if not item.is_dir():
            continue
        try:
            nested = list(item.iterdir())
        except OSError:
            continue
        if any(
            child.is_file() and not child.name.startswith(".") and child.suffix.lower() in _MODEL_FILE_SUFFIXES
            for child in nested
        ):
            return True
    return False


def is_fatal_required_model(entry: dict[str, Any]) -> bool:
    if not bool(entry.get("required", False)):
        return False
    declared = entry.get("providers")
    if isinstance(declared, list) and declared and not any(item in PRIMARY_MODEL_PROVIDERS for item in declared):
        return False
    return parse_huggingface_source(str(entry.get("source") or "")) is not None


def validate_hf_component(value: str | None, label: str) -> str | None:
    if value is None or value == "":
        return value
    if Path(value).is_absolute() or value.startswith("/") or value.startswith("\\"):
        raise RuntimeError(f"unsafe huggingface {label}: {value}")
    if ".." in Path(value).as_posix().split("/"):
        raise RuntimeError(f"unsafe huggingface {label}: {value}")
    return value


def huggingface_download_filename(entry: dict[str, Any], parsed: dict[str, str | None], dest: Path) -> str:
    explicit = str(entry.get("filename") or "").strip()
    subpath = (parsed.get("subpath") or "").strip().strip("/")
    if explicit:
        if subpath and not is_weight_filename(subpath) and not explicit.startswith(f"{subpath}/"):
            return f"{subpath}/{explicit}"
        return explicit
    if subpath and is_weight_filename(subpath):
        return subpath
    if subpath:
        return f"{subpath}/{dest.name}"
    return dest.name


def _normalize_sha256(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    digest = raw.strip().lower()
    if not digest:
        return None
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError(f"invalid sha256 digest: {raw!r}")
    return digest


def extract_http_download_specs(entry: dict[str, Any], dest: Path) -> list[dict[str, str | None]] | None:
    """Return validated HTTP file download specs for non-HF release assets (e.g. GitHub Releases)."""
    declared_files = entry.get("files")
    if isinstance(declared_files, list) and declared_files:
        specs: list[dict[str, str | None]] = []
        for item in declared_files:
            if not isinstance(item, dict):
                raise RuntimeError("model manifest 'files' entries must be mappings")
            raw_url = str(item.get("url") or "").strip()
            parsed_url = urlparse(raw_url)
            if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
                raise RuntimeError(f"invalid model file download url: {raw_url!r}")
            filename = validate_hf_component(
                str(item.get("filename") or Path(parsed_url.path).name).strip(),
                "filename",
            )
            if not filename:
                raise RuntimeError(f"missing filename for model download url: {raw_url}")
            specs.append(
                {
                    "url": raw_url,
                    "filename": filename,
                    "sha256": _normalize_sha256(item.get("sha256")),
                }
            )
        return specs

    source = str(entry.get("source") or "").strip()
    if not source or parse_huggingface_source(source) is not None:
        return None
    parsed_url = urlparse(source)
    if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
        return None
    url_name = Path(parsed_url.path).name
    if not is_weight_filename(url_name):
        return None
    filename = dest.name if is_model_file_target(entry) else str(entry.get("filename") or url_name).strip()
    validated = validate_hf_component(filename, "filename")
    if not validated:
        return None
    return [
        {
            "url": source,
            "filename": validated,
            "sha256": _normalize_sha256(entry.get("sha256")),
        }
    ]


def parse_huggingface_source(source: str | None) -> dict[str, str | None] | None:
    raw = (source or "").strip()
    if not raw:
        return None

    if "://" not in raw and "/" in raw and not raw.startswith("huggingface.co/") and not raw.startswith("hf.co/"):
        parts = [part for part in raw.split("/") if part]
        if len(parts) < 2:
            return None
        return {
            "repo_id": f"{parts[0]}/{parts[1]}",
            "revision": None,
            "subpath": "/".join(parts[2:]) or None,
        }

    normalized = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").split(":")[0].lower()
    if host not in _HF_HOSTS:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    repo_id = f"{parts[0]}/{parts[1]}"
    revision: str | None = None
    subpath: str | None = None
    rest = parts[2:]
    if rest and rest[0] in {"tree", "blob", "resolve"}:
        if len(rest) >= 2:
            revision = rest[1]
            subpath = "/".join(rest[2:]) or None
    elif rest:
        subpath = "/".join(rest)
    return {"repo_id": repo_id, "revision": revision, "subpath": subpath}
