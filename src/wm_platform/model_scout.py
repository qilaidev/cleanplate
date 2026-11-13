"""Find recently published or updated models/papers/repos relevant to watermark removal.

This is step 1 of the model-scouting playbook (docs/model-scouting.md): a repeatable sweep
over Hugging Face, arXiv and GitHub restricted to a recent window (default 30 days, use
--days 15 for a quick pass). It only produces a shortlist; adoption still goes through
the license gate and the clean-plate eval (`python -m wm_platform.eval_runner --report`).

    uv run cleanplate-scout --days 30 --output eval/scouting/2026-09-30.md
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

HF_SEARCH_TERMS = (
    "watermark",
    "remover",
    "object removal",
    "video inpainting",
    "video eraser",
    "subtitle removal",
    "text removal",
    "logo detection",
)
ARXIV_QUERIES = (
    'abs:"video object removal"',
    'abs:"video inpainting"',
    'abs:"visible watermark"',
    'cat:cs.CV AND abs:watermark AND abs:removal',
    'cat:cs.CV AND abs:"subtitle removal"',
    'cat:cs.CV AND abs:"text removal" AND abs:video',
    'ti:eraser AND cat:cs.CV',
)
GITHUB_QUERIES = (
    "video inpainting",
    "video object removal",
    "watermark removal",
    "comfyui remover",
    "subtitle removal video",
)
RELEVANCE_TERMS = (
    "watermark", "logo", "subtitle", "caption", "text removal", "inpaint", "remov", "eras",
    "video", "mask-free", "overlay",
)
# Invisible-watermark / provenance / audio work shares the vocabulary but not the problem.
OFF_TOPIC_TERMS = ("invisible", "fingerprint", "provenance", "steganograph", "audio", "sound", "llm", "language model")

PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "openrail", "creativeml-openrail-m", "cc-by-4.0", "unlicense"}
NON_COMMERCIAL_MARKERS = ("nc", "non-commercial", "noncommercial", "research", "academic")
COPYLEFT_MARKERS = ("agpl", "gpl")


@dataclass
class Candidate:
    source: str  # hf-model | arxiv | github
    name: str
    url: str
    updated: datetime
    license: str | None = None
    summary: str = ""
    popularity: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def license_class(self) -> str:
        return classify_license(self.license)

    @property
    def relevance(self) -> int:
        text = f"{self.name} {self.summary} {' '.join(self.tags)}".lower()
        score = sum(1 for term in RELEVANCE_TERMS if term in text)
        return score - 3 * sum(1 for term in OFF_TOPIC_TERMS if term in self.name.lower())


def classify_license(raw: str | None) -> str:
    """permissive | non_commercial | copyleft | unknown. Unknown is not permission."""
    if not raw:
        return "unknown"
    value = raw.strip().lower()
    if any(marker in value for marker in COPYLEFT_MARKERS):
        return "copyleft"
    if value in PERMISSIVE or value.startswith(("apache", "bsd", "mit")):
        return "permissive"
    if any(f"-{marker}" in value or value.startswith(marker) or f" {marker}" in value for marker in NON_COMMERCIAL_MARKERS):
        return "non_commercial"
    return "unknown"


def _parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def search_huggingface(client: httpx.Client, since: datetime, terms=HF_SEARCH_TERMS, limit: int = 50) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    for term in terms:
        response = client.get(
            "https://huggingface.co/api/models",
            params={"search": term, "sort": "lastModified", "direction": "-1", "limit": limit, "full": "true"},
        )
        response.raise_for_status()
        for item in response.json():
            found.update({c.name: c for c in parse_hf_models([item], since)})
    return list(found.values())


def parse_hf_models(items: list[dict[str, object]], since: datetime) -> list[Candidate]:
    out: list[Candidate] = []
    for item in items:
        updated = _parse_time(str(item.get("lastModified") or item.get("createdAt") or ""))
        if updated is None or updated < since:
            continue
        tags = [str(tag) for tag in item.get("tags") or []]
        license_tag = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), None)
        name = str(item.get("id") or item.get("modelId"))
        out.append(
            Candidate(
                source="hf-model",
                name=name,
                url=f"https://huggingface.co/{name}",
                updated=updated,
                license=license_tag,
                summary=str(item.get("pipeline_tag") or ""),
                popularity=int(item.get("likes") or 0) * 10 + int(item.get("downloads") or 0) // 100,
                tags=[tag for tag in tags if not tag.startswith(("license:", "region:", "endpoints"))][:8],
            )
        )
    return out


def search_arxiv(client: httpx.Client, since: datetime, queries=ARXIV_QUERIES, limit: int = 40) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    for query in queries:
        response = client.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": limit},
        )
        response.raise_for_status()
        found.update({c.url: c for c in parse_arxiv_feed(response.text, since)})
    return list(found.values())


def parse_arxiv_feed(xml_text: str, since: datetime) -> list[Candidate]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    out: list[Candidate] = []
    for entry in root.findall("a:entry", ns):
        updated = _parse_time(entry.findtext("a:published", default="", namespaces=ns))
        if updated is None or updated < since:
            continue
        title = " ".join((entry.findtext("a:title", default="", namespaces=ns)).split())
        summary = " ".join((entry.findtext("a:summary", default="", namespaces=ns)).split())
        url = entry.findtext("a:id", default="", namespaces=ns).replace("http://", "https://")
        has_code = any(marker in summary.lower() for marker in ("github.com", "huggingface.co", "code is available", "code will be"))
        out.append(
            Candidate(
                source="arxiv",
                name=title,
                url=url,
                updated=updated,
                summary=summary[:400],
                tags=["code-link"] if has_code else [],
            )
        )
    return out


def search_github(client: httpx.Client, since: datetime, queries=GITHUB_QUERIES) -> list[Candidate]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    found: dict[str, Candidate] = {}
    for query in queries:
        response = client.get(
            "https://api.github.com/search/repositories",
            params={"q": f"{query} pushed:>={since.date().isoformat()}", "sort": "stars", "order": "desc", "per_page": 20},
            headers=headers,
        )
        response.raise_for_status()
        for item in response.json().get("items", []):
            updated = _parse_time(item.get("pushed_at"))
            if updated is None or updated < since:
                continue
            found[item["full_name"]] = Candidate(
                source="github",
                name=item["full_name"],
                url=item["html_url"],
                updated=updated,
                license=((item.get("license") or {}).get("spdx_id") or None),
                summary=item.get("description") or "",
                popularity=int(item.get("stargazers_count") or 0),
                tags=list(item.get("topics") or [])[:8],
            )
    return list(found.values())


def rank(candidates: list[Candidate], min_relevance: int = 2) -> list[Candidate]:
    kept = [c for c in candidates if c.relevance >= min_relevance]
    return sorted(kept, key=lambda c: (c.relevance, c.popularity, c.updated), reverse=True)


def render_markdown(candidates: list[Candidate], *, days: int, now: datetime, errors: list[str]) -> str:
    lines = [
        f"# Model scouting sweep {now.date().isoformat()} (last {days} days)",
        "",
        "Generated by `cleanplate-scout`. A shortlist, not a decision: every adoption goes through",
        "the license gate and the clean-plate eval in docs/model-scouting.md.",
        "",
    ]
    if errors:
        lines += ["Sources that failed (re-run or check by hand):", ""] + [f"- {item}" for item in errors] + [""]
    for source, title in (("hf-model", "Hugging Face models"), ("arxiv", "arXiv papers"), ("github", "GitHub repositories")):
        rows = [c for c in candidates if c.source == source]
        lines += [f"## {title} ({len(rows)})", ""]
        if not rows:
            lines += ["_none in window_", ""]
            continue
        lines += ["| updated | name | license | popularity | notes |", "| --- | --- | --- | --- | --- |"]
        for c in rows:
            license_text = f"{c.license or '?'} ({c.license_class})" if source != "arxiv" else "-"
            notes = (", ".join(c.tags) or c.summary)[:120].replace("|", "/")
            lines.append(f"| {c.updated.date()} | [{c.name}]({c.url}) | {license_text} | {c.popularity} | {notes} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30, help="window size in days (15 for a quick pass)")
    parser.add_argument("--sources", default="hf,arxiv,github", help="comma list of hf,arxiv,github")
    parser.add_argument("--min-relevance", type=int, default=2)
    parser.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    args = parser.parse_args(argv)

    now = datetime.now(UTC)
    since = now - timedelta(days=max(1, args.days))
    sources = {item.strip() for item in args.sources.split(",") if item.strip()}
    searches = {"hf": search_huggingface, "arxiv": search_arxiv, "github": search_github}
    candidates: list[Candidate] = []
    errors: list[str] = []
    with httpx.Client(timeout=30.0, follow_redirects=True, headers={"User-Agent": "cleanplate-scout"}) as client:
        for key in ("hf", "arxiv", "github"):
            if key not in sources:
                continue
            try:
                candidates += searches[key](client, since)
            except Exception as exc:  # noqa: BLE001 - one failing source must not hide the others
                errors.append(f"{key}: {exc}")
    report = render_markdown(rank(candidates, args.min_relevance), days=args.days, now=now, errors=errors)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
