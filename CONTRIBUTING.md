# Contributing

## Scope

This repository is a self-hosted video watermark removal service used as a step in automated media workflows.

Please keep contributions aligned with the current direction:

- local-first deployment
- API + worker separation
- provider-based execution
- simple, runnable implementation over heavy abstraction

## Read First

1. `README.md` for positioning, quick start and configuration.
2. `docs/model-stack.md` for which detection / inpainting models are used and why.
3. `docs/architecture.md` and `docs/api.md` for boundaries and the HTTP contract.
4. `docs/roadmap.md` for what is done and what is next.

Retired workflows and plans are not kept in the tree; git history is the archive.

Model changes follow `docs/model-scouting.md`: scan, license gate, clean-plate eval, then decide.

## Development

```sh
uv sync --extra dev
uv run ruff check src tests apps
uv run pytest
```

Run services locally:

```sh
uv run python -m apps.api.main --host 0.0.0.0 --port 8000
uv run python -m apps.worker.main
```

## Pull Requests

- keep changes focused
- include tests for behavior changes
- avoid unrelated refactors
- document any new runtime dependency or model expectation

## Issues

When reporting a bug, include:

- input type and file size
- selected provider
- ComfyUI version / custom nodes / models status
- relevant logs from API, worker, and `--doctor`
