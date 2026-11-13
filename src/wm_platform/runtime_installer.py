from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from wm_platform.config import Settings
from wm_platform.runtime_contract import (
    copy_runtime_contract_files,
    expected_model_entries,
    huggingface_download_filename,
    is_fatal_required_model,
    is_model_file_target,
    load_runtime_lock,
    model_destination,
    model_is_present,
    parse_huggingface_source,
    resolve_runtime_relative,
    validate_hf_component,
)


class RuntimeInstaller:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = load_runtime_lock(settings)

    def _run(self, command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
        )

    def _ensure_repo(self, name: str, spec: dict[str, Any]) -> list[str]:
        target = resolve_runtime_relative(self.settings, str(spec["target"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        messages: list[str] = []
        git_dir = target / ".git"
        if target.exists() and not git_dir.exists():
            if any(target.iterdir()):
                raise RuntimeError(f"target exists and is not a git repo: {target}")
            target.rmdir()
        if not target.exists():
            result = self._run(["git", "clone", str(spec["url"]), str(target)])
            if result.returncode != 0:
                raise RuntimeError(f"clone {name} failed: {result.stderr.strip()}")
            messages.append(f"cloned {name}")
        fetch = self._run(["git", "fetch", "--all", "--tags"], cwd=target)
        if fetch.returncode != 0:
            raise RuntimeError(f"fetch {name} failed: {fetch.stderr.strip()}")
        checkout = self._run(["git", "checkout", str(spec["ref"])], cwd=target)
        if checkout.returncode != 0:
            raise RuntimeError(f"checkout {name} failed: {checkout.stderr.strip()}")
        messages.append(f"pinned {name} to {str(spec['ref'])[:12]}")
        return messages

    def _ensure_runtime_venv(self) -> list[str]:
        venv_dir = self.settings.comfyui_venv_dir
        python_version = str(self.lock.get("runtime", {}).get("python", "3.12"))
        messages: list[str] = []
        if shutil.which("uv") is None:
            raise RuntimeError("uv not found")
        if not venv_dir.exists():
            install_python = self._run(["uv", "python", "install", python_version])
            if install_python.returncode != 0:
                raise RuntimeError(f"install python {python_version} failed: {install_python.stderr.strip()}")
            create_venv = self._run(["uv", "venv", str(venv_dir), "--python", python_version])
            if create_venv.returncode != 0:
                raise RuntimeError(f"create venv failed: {create_venv.stderr.strip()}")
            messages.append(f"created runtime venv {venv_dir}")
        return messages

    def _install_python_packages(self) -> list[str]:
        venv_python = self.settings.comfyui_venv_dir / "bin" / "python"
        comfy_requirements = self.settings.comfyui_dir / "requirements.txt"
        messages: list[str] = []
        if shutil.which("uv") is None:
            raise RuntimeError("uv not found")
        if venv_python.exists() and comfy_requirements.exists():
            install = self._run(["uv", "pip", "install", "--python", str(venv_python), "-r", str(comfy_requirements)])
            if install.returncode != 0:
                raise RuntimeError(f"install comfy requirements failed: {install.stderr.strip()}")
            messages.append("installed ComfyUI requirements")
        for requirement in self.settings.comfyui_dir.glob("custom_nodes/*/requirements.txt"):
            install = self._run(["uv", "pip", "install", "--python", str(venv_python), "-r", str(requirement)])
            if install.returncode != 0:
                raise RuntimeError(f"install node requirements failed {requirement.parent.name}: {install.stderr.strip()}")
            messages.append(f"installed node requirements {requirement.parent.name}")
        compat = self._run(["uv", "pip", "install", "--python", str(venv_python), "numpy<2"])
        if compat.returncode != 0:
            raise RuntimeError(f"pin numpy<2 failed: {compat.stderr.strip()}")
        messages.append("pinned numpy<2")
        diffueraser = self._run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "diffusers>=0.35,<0.36",
                "transformers>=4.56,<5",
                "peft>=0.18",
                "accelerate>=1.13",
                "huggingface-hub<1",
                "matplotlib",
            ]
        )
        if diffueraser.returncode != 0:
            raise RuntimeError(f"install diffueraser compatible deps failed: {diffueraser.stderr.strip()}")
        messages.append("pinned diffueraser compatible deps and matplotlib")
        return messages

    def plan(self) -> dict[str, object]:
        models = expected_model_entries(self.settings)
        return {
            "python": str(self.lock.get("runtime", {}).get("python", "3.12")),
            "runtime_root": str(self.settings.runtime_root),
            "repositories": self.lock.get("repositories", {}),
            "venv_dir": str(self.settings.comfyui_venv_dir),
            "models": models,
            "model_destinations": [str(model_destination(self.settings, entry)) for entry in models],
        }

    def install(
        self,
        *,
        include_python_packages: bool = True,
        include_models: bool = False,
        include_repos: bool = True,
    ) -> list[str]:
        messages: list[str] = []
        messages.extend(copy_runtime_contract_files(self.settings))
        if include_repos:
            messages.extend(self._ensure_runtime_venv())
            repositories = self.lock.get("repositories", {})
            if "comfyui" in repositories:
                messages.extend(self._ensure_repo("comfyui", repositories["comfyui"]))
            self.settings.comfyui_custom_nodes_dir.mkdir(parents=True, exist_ok=True)
            for name, spec in repositories.items():
                if name == "comfyui":
                    continue
                if isinstance(spec, dict):
                    messages.extend(self._ensure_repo(str(name), spec))
        else:
            messages.append("skipped repository installation")
        if include_python_packages:
            messages.extend(self._install_python_packages())
        else:
            messages.append("skipped python package installation")
        if include_models:
            messages.extend(self._ensure_models())
        else:
            messages.append("skipped model download")
        return messages

    def _huggingface_fns(self):
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to download models") from exc
        return hf_hub_download, snapshot_download

    def _ensure_models(self) -> list[str]:
        messages: list[str] = []
        required_failures: list[str] = []
        entries = expected_model_entries(self.settings)
        if not entries:
            messages.append("model manifest missing or empty")
            return messages

        for entry in entries:
            name = str(entry.get("name") or "unknown")
            required = bool(entry.get("required", False))
            dest = model_destination(self.settings, entry)
            if model_is_present(entry, dest):
                messages.append(f"model {name} already present at {dest}")
                continue
            try:
                self._download_model_entry(entry, dest)
            except Exception as exc:
                detail = f"model {name} failed: {exc}"
                messages.append(detail)
                if is_fatal_required_model(entry):
                    required_failures.append(f"{name} -> {dest}: {exc}")
                elif required:
                    messages.append(f"non-blocking required model {name} not installed")
                else:
                    messages.append(f"optional model {name} skipped")
                continue
            if not model_is_present(entry, dest):
                detail = f"model {name} dest missing after download: {dest}"
                messages.append(detail)
                if is_fatal_required_model(entry):
                    required_failures.append(f"{name} -> {dest}: dest missing")
                elif required:
                    messages.append(f"non-blocking required model {name} dest missing")
                continue
            messages.append(f"installed model {name} -> {dest}")

        if required_failures:
            error = RuntimeError("required models not installed: " + "; ".join(required_failures))
            error.messages = messages  # type: ignore[attr-defined]
            raise error
        return messages

    def _download_model_entry(self, entry: dict[str, Any], dest: Path) -> None:
        source = str(entry.get("source") or "").strip()
        parsed = parse_huggingface_source(source)
        if parsed is None:
            raise RuntimeError(f"source is not a Hugging Face repo: {source or '(empty)'}; place files at {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        repo_id = validate_hf_component(parsed.get("repo_id"), "repo_id")
        revision = validate_hf_component(parsed.get("revision"), "revision")
        subpath = validate_hf_component(parsed.get("subpath"), "subpath")
        if not repo_id:
            raise RuntimeError("huggingface repo_id is empty")
        parsed = {**parsed, "repo_id": repo_id, "revision": revision, "subpath": subpath}
        hf_hub_download, snapshot_download = self._huggingface_fns()
        if is_model_file_target(entry):
            filename = validate_hf_component(huggingface_download_filename(entry, parsed, dest), "filename")
            if not filename:
                raise RuntimeError(f"huggingface filename is empty for {dest}")
            kwargs: dict[str, Any] = {"repo_id": repo_id, "filename": filename, "local_dir": str(dest.parent)}
            if revision:
                kwargs["revision"] = revision
            downloaded = Path(str(hf_hub_download(**kwargs))).resolve()
            dest_root = dest.parent.resolve()
            if not downloaded.is_relative_to(dest_root):
                raise RuntimeError(f"download escaped destination root: {downloaded}")
            dest_resolved = dest.resolve()
            if not dest_resolved.is_relative_to(dest_root):
                raise RuntimeError(f"model dest escaped destination root: {dest}")
            if downloaded != dest_resolved:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, dest)
            return

        dest.mkdir(parents=True, exist_ok=True)
        kwargs = {"repo_id": repo_id, "local_dir": str(dest)}
        if revision:
            kwargs["revision"] = revision
        if subpath:
            prefix = subpath.rstrip("/")
            kwargs["allow_patterns"] = [prefix, f"{prefix}/*", f"{prefix}/**"]
        snapshot_download(**kwargs)
        dest_resolved = dest.resolve()
        if dest.exists() and dest.is_dir():
            for child in dest.rglob("*"):
                if not child.resolve().is_relative_to(dest_resolved):
                    raise RuntimeError(f"download escaped destination root: {child}")
