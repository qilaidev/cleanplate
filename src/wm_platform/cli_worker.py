from __future__ import annotations

import argparse
import json
import logging

from wm_platform.bootstrap import bootstrap
from wm_platform.comfy_runtime import build_comfyui_command, comfyui_health, start_comfyui, wait_for_comfyui
from wm_platform.doctor import provider_doctor_report
from wm_platform.maintenance import run_file_cleanup
from wm_platform.provider_runtime import ProviderRuntime
from wm_platform.runtime_installer import RuntimeInstaller
from wm_platform.worker_service import WorkerService


def _install_mode(*, include_repos: bool, include_python_packages: bool, include_models: bool) -> str:
    if include_models and not include_repos:
        return "models_only"
    if include_repos and not include_python_packages and include_models:
        return "repos_and_models"
    if include_repos and not include_python_packages:
        return "repos_only"
    if include_models:
        return "full_framework_with_models"
    return "full_framework"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanplate worker process")
    parser.add_argument("--once", action="store_true", help="process at most one polling cycle, then exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--doctor", action="store_true", help="print provider/runtime readiness summary and exit")
    parser.add_argument("--runtime-plan", action="store_true", help="print local AI runtime bootstrap plan and exit")
    parser.add_argument("--install-runtime", action="store_true", help="install local AI runtime skeleton and exit")
    parser.add_argument("--repos-only", action="store_true", help="with --install-runtime: skip python package installs")
    parser.add_argument("--models", action="store_true", help="with --install-runtime: download models from the runtime manifest")
    parser.add_argument("--models-only", action="store_true", help="download models from the runtime manifest and exit")
    parser.add_argument("--comfyui-plan", action="store_true", help="print ComfyUI startup command and exit")
    parser.add_argument("--comfyui-health", action="store_true", help="check ComfyUI /system_stats and exit")
    parser.add_argument("--start-comfyui", action="store_true", help="start ComfyUI and wait for health, then exit")
    parser.add_argument("--cleanup", action="store_true", help="run file cleanup and exit")
    parser.add_argument("--execute-cleanup", action="store_true", help="with --cleanup: delete candidate files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings, repository = bootstrap()
    if args.doctor:
        report = provider_doctor_report(settings)
        logging.info("doctor report=%s", report)
        print(json.dumps(report, ensure_ascii=True))
        return
    if args.runtime_plan:
        plan = RuntimeInstaller(settings).plan()
        logging.info("runtime plan=%s", plan)
        print(json.dumps(plan, ensure_ascii=True))
        return
    if args.install_runtime or args.models_only:
        include_repos = not args.models_only
        include_python_packages = include_repos and not args.repos_only
        include_models = args.models or args.models_only
        mode = _install_mode(include_repos=include_repos, include_python_packages=include_python_packages, include_models=include_models)
        try:
            messages = RuntimeInstaller(settings).install(
                include_python_packages=include_python_packages,
                include_models=include_models,
                include_repos=include_repos,
            )
            payload = {"ok": True, "mode": mode, "messages": messages}
        except RuntimeError as exc:
            messages = getattr(exc, "messages", None)
            payload = {"ok": False, "mode": mode, "messages": messages if isinstance(messages, list) else [str(exc)]}
            logging.error("runtime install failed=%s", payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(1) from exc
        logging.info("runtime install=%s", payload)
        print(json.dumps(payload, ensure_ascii=True))
        return
    if args.comfyui_plan:
        payload = {"command": build_comfyui_command(settings)}
        logging.info("comfyui plan=%s", payload)
        print(json.dumps(payload, ensure_ascii=True))
        return
    if args.comfyui_health:
        payload = comfyui_health(settings)
        logging.info("comfyui health=%s", payload)
        print(json.dumps(payload, ensure_ascii=True))
        return
    if args.start_comfyui:
        process = start_comfyui(settings)
        payload = {
            "pid": process.pid,
            "command": build_comfyui_command(settings),
            "health": wait_for_comfyui(settings),
        }
        logging.info("comfyui started=%s", payload)
        print(json.dumps(payload, ensure_ascii=True))
        return
    if args.cleanup:
        report = run_file_cleanup(settings=settings, repository=repository, execute=args.execute_cleanup)
        logging.info("cleanup report=%s", report)
        print(json.dumps(report, ensure_ascii=True))
        return

    providers = ProviderRuntime(settings, repository=repository)
    service = WorkerService(settings=settings, repository=repository, providers=providers)

    if args.once:
        service.run_once()
        service.callback_service.run_once()
        return

    try:
        service.run_forever()
    except KeyboardInterrupt:
        service.stop()
