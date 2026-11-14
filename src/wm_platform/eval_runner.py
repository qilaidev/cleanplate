from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from wm_platform.config import Settings, load_settings
from wm_platform.db import init_db
from wm_platform.eval_judge import merge_judgments_csv
from wm_platform.eval_metrics import auto_judgment, evaluate_output
from wm_platform.models import JobCreate
from wm_platform.pipeline import AI_INPAINT_PROVIDERS
from wm_platform.provider_runtime import ProviderRuntime
from wm_platform.repository import JobRepository
from wm_platform.storage import ensure_storage_dirs
from wm_platform.worker_service import WorkerService

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}
SPLITS = ("simple", "medium", "hard")
DEFAULT_PROVIDERS = "auto,minimax_remover,comfy_diffueraser"
AI_SUCCESS_PROVIDERS = frozenset(AI_INPAINT_PROVIDERS)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    split: str
    path: Path
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    clean_path: Path | None = None
    region_path: Path | None = None


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "eval" / "dataset"


def write_synthetic_logo_video(
    path: Path,
    *,
    frames: int = 8,
    size: tuple[int, int] = (160, 120),
    fps: int = 8,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to write synthetic video: {path}")
    try:
        for _ in range(max(2, frames)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            frame[4:28, 4:48] = (240, 240, 240)
            frame[6:26, 6:46] = (20, 180, 255)
            writer.write(frame)
    finally:
        writer.release()
    return path


def load_eval_cases(dataset_root: Path, split: str) -> list[EvalCase]:
    selected = SPLITS if split == "all" else (split,)
    cases: list[EvalCase] = []
    seen: set[tuple[str, Path]] = set()
    manifest_path = dataset_root / "manifest.csv"
    if manifest_path.exists():
        for case in _cases_from_manifest(manifest_path, dataset_root, selected):
            key = (case.split, case.path.resolve())
            if key in seen:
                continue
            seen.add(key)
            cases.append(case)
    for case in _cases_from_directories(dataset_root, selected):
        key = (case.split, case.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        cases.append(case)
    return cases


def _cases_from_manifest(manifest_path: Path, dataset_root: Path, selected: tuple[str, ...]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            split = (row.get("split") or "").strip()
            filename = (row.get("filename") or "").strip()
            if split not in selected or not filename:
                continue
            path = (dataset_root / split / filename).resolve()
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            case_id = (row.get("id") or "").strip() or f"{split}-{index:02d}"
            clean_path = _dataset_file(dataset_root, row.get("clean_filename"))
            region_path = _dataset_file(dataset_root, row.get("region_filename"))
            cases.append(
                EvalCase(
                    case_id=case_id,
                    split=split,
                    path=path,
                    notes=(row.get("notes") or "").strip(),
                    extra={key: (value or "") for key, value in row.items() if value is not None},
                    clean_path=clean_path,
                    region_path=region_path,
                )
            )
    return cases


def _dataset_file(dataset_root: Path, raw: str | None) -> Path | None:
    name = (raw or "").strip()
    if not name:
        return None
    candidate = (dataset_root / name).resolve()
    if candidate.is_file() and candidate.is_relative_to(dataset_root.resolve()):
        return candidate
    return None


def _cases_from_directories(dataset_root: Path, selected: tuple[str, ...]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for split in selected:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        videos = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        for index, path in enumerate(videos, start=1):
            cases.append(EvalCase(case_id=f"{split}-{index:02d}", split=split, path=path.resolve()))
    return cases


def build_eval_settings(base: Settings | None = None) -> Settings:
    from dataclasses import replace

    resolved = base or load_settings()
    work = Path(tempfile.mkdtemp(prefix="cleanplate-eval-"))
    return replace(
        resolved,
        storage_root=work,
        inbox_dir=work / "inbox",
        outbox_dir=work / "outbox",
        work_dir=work / "work",
        comfyui_output_dir=work / "comfy_spool",
        db_path=work / "eval.db",
    )


def run_eval(
    *,
    dataset_root: Path,
    split: str,
    providers: list[str],
    settings: Settings,
    repository: JobRepository,
    results_dir: Path | None = None,
    judgments: Path | None = None,
) -> dict[str, object]:
    output_dir = results_dir or (dataset_root / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_eval_cases(dataset_root, split)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    providers = [item.strip() for item in providers if item.strip()] or ["auto"]
    all_results: list[dict[str, object]] = []
    empty_message = None if cases else f"no videos found for split '{split}'"

    worker = WorkerService(
        settings=settings,
        repository=repository,
        providers=ProviderRuntime(settings, repository=repository),
    )
    written: list[str] = []

    for provider in providers:
        provider_results: list[dict[str, object]] = []
        for case in cases:
            provider_results.append(_run_case(worker, repository, settings, case, provider))
        payload = {
            "date": date_stamp,
            "split": split,
            "provider": provider,
            "dataset_root": str(dataset_root),
            "message": empty_message,
            "results": provider_results,
        }
        _apply_judgment_overlays(payload, dataset_root, judgments)
        destination = output_dir / f"{date_stamp}-{provider}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        written.append(str(destination))
        all_results.extend(provider_results)

    return {
        "date": date_stamp,
        "split": split,
        "providers": providers,
        "dataset_root": str(dataset_root),
        "message": empty_message,
        "results": all_results,
        "written": written,
    }


def _run_case(
    worker: WorkerService,
    repository: JobRepository,
    settings: Settings,
    case: EvalCase,
    provider: str,
) -> dict[str, object]:
    job = repository.create_job(
        JobCreate(
            tenant_id=settings.default_tenant_id,
            media_type="video",
            provider_requested=provider,  # type: ignore[arg-type]
            fallback_chain_json=JobRepository.default_fallback_chain(provider),
            input_path=str(case.path),
            idempotency_key=f"eval-{case.case_id}-{provider}-{job_token(case, provider)}",
        )
    )
    worker.run_once()
    updated = repository.get_job(job.job_id) or job
    metadata = repository.get_run_metadata(updated.job_id)
    ai_success = _is_ai_success(updated.status, updated.provider_selected)
    metrics = _case_metrics(case, updated.output_path, updated.status)
    judgment = "pending"
    judgment_source = "manual"
    if metrics is not None and ai_success:
        judgment = auto_judgment(metrics)
        judgment_source = "auto" if judgment != "pending" else "manual"
    return {
        "id": case.case_id,
        "split": case.split,
        "filename": case.path.name,
        "job_id": updated.job_id,
        "provider_requested": provider,
        "provider_selected": updated.provider_selected,
        "status": updated.status,
        "error_code": updated.error_code,
        "duration_ms": updated.duration_ms,
        "scene_type": metadata.scene_type if metadata else None,
        "confidence_level": metadata.confidence_level if metadata else None,
        "mask_ratio": metadata.mask_ratio if metadata else None,
        "output_path": updated.output_path,
        "workflow_name": metadata.workflow_name if metadata else None,
        "notes": case.notes or updated.error_message or "",
        "judgment": judgment,
        "judgment_source": judgment_source,
        "metrics": metrics,
        "degraded": updated.provider_selected == "local_fallback",
        "ai_success": ai_success,
    }


def _case_metrics(case: EvalCase, output_path: str | None, status: str | None) -> dict[str, object] | None:
    if case.clean_path is None or status != "succeeded" or not output_path:
        return None
    output = Path(output_path)
    if not output.is_file():
        return None
    try:
        return evaluate_output(output, case.path, case.clean_path, region_path=case.region_path)
    except Exception as exc:  # metrics must never break the eval run
        return {"frames_compared": 0, "error": f"metrics failed: {exc}"}


def job_token(case: EvalCase, provider: str) -> str:
    return f"{case.path.stat().st_mtime_ns}-{provider}"


def _is_ai_success(status: str | None, provider_selected: str | None) -> bool:
    return status == "succeeded" and provider_selected in AI_SUCCESS_PROVIDERS


def build_eval_report(payload: dict[str, object]) -> dict[str, object]:
    results = payload.get("results")
    rows = [row for row in results if isinstance(row, dict)] if isinstance(results, list) else []
    by_provider: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        provider = str(row.get("provider_requested") or payload.get("provider") or "unknown")
        by_provider.setdefault(provider, []).append(row)
    provider_summaries = {
        name: _summarize_rows(items) for name, items in sorted(by_provider.items())
    }
    return {
        "date": payload.get("date"),
        "split": payload.get("split"),
        "dataset_root": payload.get("dataset_root"),
        "providers": provider_summaries,
        "overall": _summarize_rows(rows),
    }


def render_eval_report_markdown(report: dict[str, object]) -> str:
    lines = [
        f"# Eval report {report.get('date') or ''}".rstrip(),
        "",
        f"split: {report.get('split')}",
        "",
    ]
    providers = report.get("providers")
    if isinstance(providers, dict):
        for name, summary in providers.items():
            if not isinstance(summary, dict):
                continue
            lines.extend(_markdown_provider_section(str(name), summary))
    overall = report.get("overall")
    if isinstance(overall, dict):
        lines.extend(_markdown_provider_section("overall", overall))
    return "\n".join(lines).rstrip() + "\n"


def write_eval_report(report: dict[str, object], output_dir: Path, date_stamp: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{date_stamp}-report.json"
    md_path = output_dir / f"{date_stamp}-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(render_eval_report_markdown(report), encoding="utf-8")
    return json_path, md_path


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    succeeded = 0
    failed = 0
    degraded = 0
    ai_success = 0
    error_codes: dict[str, int] = {}
    judgments: dict[str, int] = {}
    durations: list[float] = []
    mask_ratios: list[float] = []
    metric_values: dict[str, list[float]] = {}
    residual_detected = 0
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            for key in ("psnr_region", "psnr_outside", "ssim_region", "flicker_region"):
                value = metrics.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_values.setdefault(key, []).append(float(value))
            if metrics.get("residual_detected"):
                residual_detected += 1
        status = str(row.get("status") or "")
        selected = row.get("provider_selected")
        if status == "succeeded":
            succeeded += 1
        else:
            failed += 1
        if selected == "local_fallback" and status == "succeeded":
            degraded += 1
        if _is_ai_success(status, str(selected) if selected is not None else None):
            ai_success += 1
        error_code = row.get("error_code")
        if error_code:
            key = str(error_code)
            error_codes[key] = error_codes.get(key, 0) + 1
        judgment = str(row.get("judgment") or "pending")
        judgments[judgment] = judgments.get(judgment, 0) + 1
        duration = row.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
        mask_ratio = row.get("mask_ratio")
        if isinstance(mask_ratio, (int, float)):
            mask_ratios.append(float(mask_ratio))
    return {
        "total": len(rows),
        "succeeded": succeeded,
        "failed": failed,
        "degraded": degraded,
        "local_fallback": degraded,
        "ai_success": ai_success,
        "error_codes": error_codes,
        "judgments": judgments,
        "mean_duration_ms": round(sum(durations) / len(durations), 3) if durations else None,
        "mean_mask_ratio": round(sum(mask_ratios) / len(mask_ratios), 6) if mask_ratios else None,
        "metrics": {
            f"mean_{key}": round(sum(values) / len(values), 4) for key, values in sorted(metric_values.items())
        },
        "residual_detected": residual_detected,
    }


def _markdown_provider_section(name: str, summary: dict[str, object]) -> list[str]:
    return [
        f"## {name}",
        f"- total: {summary.get('total', 0)}",
        f"- succeeded: {summary.get('succeeded', 0)}",
        f"- failed: {summary.get('failed', 0)}",
        f"- ai_success: {summary.get('ai_success', 0)}",
        f"- local_fallback / degraded: {summary.get('degraded', 0)} (not watermark-removal success)",
        f"- error_codes: {summary.get('error_codes') or {}}",
        f"- judgments: {summary.get('judgments') or {}}",
        f"- mean_duration_ms: {summary.get('mean_duration_ms')}",
        f"- mean_mask_ratio: {summary.get('mean_mask_ratio')}",
        f"- metrics (clean-plate reference): {summary.get('metrics') or {}}",
        f"- residual_detected: {summary.get('residual_detected', 0)}",
        "",
    ]


def _load_results_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"results file is not a JSON object: {path}")
    return payload


def _apply_judgment_overlays(payload: dict[str, object], dataset_root: Path, judgments: Path | None) -> None:
    results = payload.get("results")
    if not isinstance(results, list):
        return
    candidates: list[Path] = []
    if judgments is not None:
        candidates.append(judgments)
    candidates.append(dataset_root / "judgments.csv")
    for candidate in candidates:
        if candidate.exists():
            merge_judgments_csv([row for row in results if isinstance(row, dict)], candidate)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cleanplate eval cases in-process")
    parser.add_argument("--split", default="simple", choices=["simple", "medium", "hard", "all"])
    parser.add_argument("--providers", default=DEFAULT_PROVIDERS)
    parser.add_argument("--provider", default=None, help="single provider; overrides --providers when set")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--report", action="store_true", help="write a summary report next to results")
    parser.add_argument("--from-results", type=Path, default=None, help="summarize an existing results JSON")
    parser.add_argument("--judgments", type=Path, default=None, help="optional judgments.csv overlay")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_root = (args.dataset_root or default_dataset_root()).expanduser().resolve()
    if args.from_results is not None:
        result_path = args.from_results.expanduser().resolve()
        payload = _load_results_payload(result_path)
        _apply_judgment_overlays(payload, dataset_root, args.judgments)
        result_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        if args.report:
            report = build_eval_report(payload)
            date_stamp = str(payload.get("date") or datetime.now().strftime("%Y-%m-%d"))
            output_dir = args.from_results.expanduser().resolve().parent
            write_eval_report(report, output_dir, date_stamp)
            payload = {**payload, "report": report}
        print(json.dumps(payload if not args.report else payload.get("report", payload), ensure_ascii=True))
        return 0

    raw_providers = args.provider or args.providers
    providers = [item.strip() for item in str(raw_providers).split(",") if item.strip()]
    settings = build_eval_settings(load_settings())
    ensure_storage_dirs(settings)
    init_db(settings)
    repository = JobRepository(settings)
    payload = run_eval(
        dataset_root=dataset_root,
        split=args.split,
        providers=providers,
        settings=settings,
        repository=repository,
        judgments=args.judgments,
    )
    if args.report:
        report = build_eval_report(payload)
        date_stamp = str(payload.get("date") or datetime.now().strftime("%Y-%m-%d"))
        write_eval_report(report, dataset_root / "results", date_stamp)
        payload = {**payload, "report": report}
    if payload.get("message"):
        print(payload["message"])
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
