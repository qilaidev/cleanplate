from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

JUDGMENTS = frozenset({"pass", "fail", "degraded"})


def apply_judgment(
    result_path: Path,
    case_id: str,
    judgment: str,
    *,
    provider: str | None = None,
) -> int:
    if judgment not in JUDGMENTS:
        raise ValueError(f"judgment must be one of {sorted(JUDGMENTS)}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    updated = _apply_to_payload(payload, case_id, judgment, provider=provider)
    if updated == 0:
        raise KeyError(f"case not found: {case_id}")
    result_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return updated


def apply_judgments_csv(result_path: Path, csv_path: Path) -> int:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    updated = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("id") or "").strip()
            judgment = (row.get("judgment") or "").strip().lower()
            provider = (row.get("provider") or "").strip() or None
            if not case_id or judgment not in JUDGMENTS:
                continue
            updated += _apply_to_payload(payload, case_id, judgment, provider=provider)
    result_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return updated


def merge_judgments_csv(results: list[dict[str, object]], csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return results
    overlays: dict[tuple[str, str | None], str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("id") or "").strip()
            judgment = (row.get("judgment") or "").strip().lower()
            provider = (row.get("provider") or "").strip() or None
            if not case_id or judgment not in JUDGMENTS:
                continue
            overlays[(case_id, provider)] = judgment
    for row in results:
        case_id = str(row.get("id") or "")
        provider = str(row.get("provider_requested") or "") or None
        judgment = overlays.get((case_id, provider)) or overlays.get((case_id, None))
        current = str(row.get("judgment") or "pending")
        if judgment and (current in {"", "pending"} or row.get("judgment_source") == "auto"):
            row["judgment"] = judgment
            row["judgment_source"] = "manual"
    return results


def _apply_to_payload(
    payload: dict[str, object],
    case_id: str,
    judgment: str,
    *,
    provider: str | None = None,
) -> int:
    results = payload.get("results")
    if not isinstance(results, list):
        return 0
    updated = 0
    for row in results:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or "") != case_id:
            continue
        if provider and str(row.get("provider_requested") or "") != provider:
            continue
        row["judgment"] = judgment
        row["judgment_source"] = "manual"
        updated += 1
    return updated


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set eval judgment fields on a results JSON")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--id", dest="case_id", default=None)
    parser.add_argument("--judgment", choices=sorted(JUDGMENTS), default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_path = args.result.expanduser().resolve()
    if args.csv is not None:
        updated = apply_judgments_csv(result_path, args.csv.expanduser().resolve())
        print(json.dumps({"updated": updated, "result": str(result_path)}))
        return 0
    if not args.case_id or not args.judgment:
        raise SystemExit("provide --id and --judgment, or --csv")
    updated = apply_judgment(result_path, args.case_id, args.judgment, provider=args.provider)
    print(json.dumps({"updated": updated, "id": args.case_id, "judgment": args.judgment}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
