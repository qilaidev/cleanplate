# Cleanplate — Self-Hosted Async Video Watermark Detection, ROI Inpainting & QA Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20HTTP-009688.svg)](docs/api.md)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.38.1-FF6F00.svg)](docs/model-stack.md)
[![Architecture](https://img.shields.io/badge/Worker-Async%20Queue-6E44FF.svg)](docs/architecture.md)

[中文文档 (README.md)](README.md) · **English (README.en.md)** · [llms.txt](llms.txt) · [API Contract](docs/api.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

**Cleanplate** is a self-hosted, asynchronous video watermark removal service built on **FastAPI**, **ComfyUI**, and **FFmpeg** for automated media pipelines. It automatically detects static or moving watermarks without manual masking, plans per-region ROI crops and 81-frame chunks, executes a multi-model inpainting fallback chain, stitches audio/video seamlessly, and returns a clean video alongside a structured QA report via polling or HMAC-signed webhooks.

Named after the film post-production *clean plate* (a background frame with overlays removed). MIT-licensed service code, single-GPU ready, horizontally scalable across multiple workers.

> **Compliance & Licensing Notice**: Only process media you own or have explicit rights to modify. While Cleanplate's service code is **MIT** licensed, the default primary inpainting model weights (**MiniMax-Remover**) are **CC-BY-NC-4.0** (non-commercial) and **ProPainter** priors are **S-Lab License 1.0** (non-commercial). Review [Model Stack & Licenses](docs/model-stack.md#许可证商用前必看) before any commercial deployment.

---

## Positioning

| What it is | What it is not |
| --- | --- |
| An async processing node for automated workflows (HTTP submit -> webhook callback / polling) | A consumer-facing online watermark removal website |
| Automatic detection + ROI inpainting + automated QA with a structured JSON `report` per job | A manual-masking video editing tool |
| A swappable model stack with a standardized "scout -> license gate -> clean-plate eval -> swap" lifecycle | A single-model demo locked to one backend |
| Single-host / single-GPU friendly, scaling horizontally by adding worker processes | A multi-tenant SaaS platform with billing or cluster scheduling |

---

## 6-Stage Processing Pipeline

```text
POST /v1/jobs (multipart upload or path under DWM_STORAGE_ROOT)
  └─ worker claims job
      1. Ingest Normalization   VFR / rotation / odd dimensions / non-yuv420p -> CFR intermediate
      2. Detection (Single)     OpenCV static corner & semi-transparent edge detector;
                                falls back to ComfyUI native SAM 3.1 when uncertain
                                └─ No watermark -> NO_WATERMARK_DETECTED (or passthrough)
                                └─ Uncertain    -> DETECTOR_LOW_CONFIDENCE (refuses blind inpaint)
      3. ROI & Chunk Planning   Per-watermark ROI crop windows; 81-frame (4k+1) chunk splitting
      4. Inpainting Chain       MiniMax-Remover (primary) -> DiffuEraser (secondary) on ComfyUI
      5. Audio/Video Stitch     Trims chunk overlap frames, remuxes original audio, unifies codec
      6. Automated QA & Report  Checks mask_mean_abs_diff & residual_overlap; triggers fallback on
                                INPAINT_NO_EFFECT or flags needs_review; delivers HMAC webhook
```

Every completed or failed job writes a structured `report` detailing input normalization, detection confidence/regions, chunk plan, per-provider attempts, and QA metrics so downstream systems can accept, review, or discard outputs deterministically. See [docs/api.md](docs/api.md).

---

## Quick Start

### 1. Zero-GPU Mock Quickstart (Tests & API Smoke Run)

Prerequisites: Python 3.11 or 3.12 (`>=3.11,<3.13`), [uv](https://docs.astral.sh/uv/), and `ffmpeg` / `ffprobe` in `PATH`.

```sh
uv sync --extra dev
uv run pytest                                   # 139 tests run under mock ComfyUI — no GPU required

uv run cleanplate-api --host 127.0.0.1 --port 8000
uv run cleanplate-worker                        # in a second terminal (add --once for single-job debug)
```

Verify the full HTTP -> queue -> worker -> FFmpeg -> report pipeline without installing GPU weights using `provider=local_fallback` (copies the normalized video without AI inpainting):

```sh
ffmpeg -v quiet -f lavfi -i testsrc=duration=2:size=320x240:rate=10 -pix_fmt yuv420p /tmp/smoke.mp4 -y
curl -s -X POST http://127.0.0.1:8000/v1/jobs -H "X-API-Key: dev-secret-key" \
  -F media_type=video -F provider=local_fallback -F file=@/tmp/smoke.mp4
curl -s http://127.0.0.1:8000/v1/jobs/<job_id> -H "X-API-Key: dev-secret-key"
```

### 2. GPU Production Setup (ComfyUI + Weights)

```sh
uv run cleanplate-worker --runtime-plan               # Preview local ComfyUI runtime & model download plan
uv run cleanplate-worker --install-runtime --models   # Install per .runtime/lock.yaml & manifest (or --models-only)
uv run cleanplate-worker --start-comfyui              # Start ComfyUI and wait until healthy (--comfyui-health to probe)
uv run cleanplate-worker --doctor                     # Verify providers.runnable and commercial license status
```

Once `GET /v1/providers` reports `minimax_remover.runnable=true`, submit jobs with `provider=auto`. Run `uv run cleanplate-worker --cleanup` (dry-run) and `--cleanup --execute-cleanup` periodically to prune expired job artifacts. See [docs/operations.md](docs/operations.md) for deployment, backups, and scaling.

---

## Configuration

All settings are configured via environment variables (see [.env.example](.env.example)):

| Variable | Default | Description |
| --- | --- | --- |
| `DWM_ENV` | `development` | Rejects the default dev API key when set to `production` |
| `DWM_DEFAULT_API_KEY` | `dev-secret-key` | Required `X-API-Key` header value |
| `DWM_API_HOST` / `DWM_API_PORT` | `127.0.0.1` / `8000` | Bind address and port for `cleanplate-api` |
| `DWM_STORAGE_ROOT` | `storage` | Root directory for `inbox`, `outbox`, `work`, and `app.db` |
| `DWM_COMFYUI_API_URL` | `http://127.0.0.1:8188` | ComfyUI endpoint; each worker can target a separate GPU instance |
| `DWM_DETECTOR_MODE` | `auto` | `static_overlay` / `sam3_text` / `auto` (falls back to SAM 3.1 when uncertain) |
| `DWM_WATERMARK_DETECT_PROMPT` | `watermark, logo, on-screen text overlay, corner badge, platform stamp` | SAM 3.1 text prompt (supports `:N` instance cap) |
| `DWM_NO_WATERMARK_POLICY` | `fail` | `passthrough` returns the original video when no watermark is detected |
| `DWM_CHUNK_FRAMES` | `81` | Frames per chunk (snapped to `4k+1`); `0` disables chunking (short clips only) |
| `DWM_QA_ENABLED` | `true` | Enables post-inpaint quality gate |
| `DWM_QA_RESIDUAL_POLICY` | `flag` | On suspected residual: `flag` sets `needs_review=true`; `fail` fails the job |
| `DWM_QUALITY_MODE` | `balanced` | `fast` / `balanced` / `quality` / `corner_hq` (overridable per job) |
| `DWM_MINIMAX_ROI_ENABLED` | `true` | Crops and inpaints only ROI windows around detected watermarks |
| `DWM_MAX_VIDEO_SECONDS` | `180` | Rejects videos exceeding this duration (`0` disables limit) |
| `DWM_COMFYUI_PROMPT_TIMEOUT_SECONDS` | `1800` | Per-chunk ComfyUI execution timeout |
| `DWM_AUTO_ALLOW_COPY_FALLBACK` | `false` | Allows `auto` to fall back to copying input; keep `false` in production |
| `DWM_FILE_RETENTION_DAYS` | `7` | Retention window for `--cleanup --execute-cleanup` |
| `DWM_KEEP_WORK_FILES` | `false` | Preserves intermediate masks and chunks for debugging |

Full configuration reference is in `src/wm_platform/config.py`.

---

## Model Stack & Licenses

| Role | Component | License | Notes |
| --- | --- | --- | --- |
| Detection (Default) | OpenCV Static/Edge Detector | MIT | Scans outer 22% border bands for corner/static overlays |
| Detection (Fallback) | SAM 3.1 (ComfyUI native) | SAM License | Handles moving/central text watermarks; requires GPU |
| Primary Inpainting | MiniMax-Remover 1.3B | **CC-BY-NC-4.0** | Non-commercial weights |
| Secondary Inpainting | DiffuEraser + ProPainter | Apache-2.0 / **S-Lab 1.0 (NC)** | Requires manual placement of 5 ProPainter weights (~0.4 GB) |

`--install-runtime --models` downloads ~14 GB of weights (~18–20 GB total `.runtime/` footprint). See [docs/model-stack.md](docs/model-stack.md) and [docs/model-scouting.md](docs/model-scouting.md) (`uv run cleanplate-scout --days 30`) for commercial alternatives and model evaluation.

---

## FAQ

### 1. What GPU and VRAM are required to run Cleanplate?
Real AI inpainting requires an NVIDIA GPU running ComfyUI (typically 16 GB+ VRAM for 81-frame diffusion chunks with `MiniMax-Remover` 1.3B or `DiffuEraser`). When VRAM is tight, enable `DWM_MINIMAX_ROI_ENABLED=true` (default) to inpaint only watermark ROI crops rather than full frames, or reduce `DWM_CHUNK_FRAMES`.

### 2. Can I run and test Cleanplate without a GPU or ComfyUI?
Yes. All 139 tests in `uv run pytest` run against a mock ComfyUI backend with zero GPU required. You can also run `cleanplate-api` and `cleanplate-worker` locally and submit jobs with `provider=local_fallback` to validate HTTP submission, FFmpeg normalization, SQLite persistence, and callback delivery.

### 3. Can I use Cleanplate in a commercial product?
Cleanplate's **service code** is MIT licensed, so you can integrate, modify, and self-host it freely. However, the default **model weights** (`MiniMax-Remover` under CC-BY-NC-4.0 and `ProPainter` priors under S-Lab License 1.0) are **non-commercial**. For commercial workflows, either obtain commercial weight licenses or swap in commercially permissive models (such as Apache-2.0 `Wan2.1 VACE` or `CLEAR`, documented in [docs/model-stack.md](docs/model-stack.md)).

### 4. How does Cleanplate handle moving watermarks vs. static corner logos?
In `DWM_DETECTOR_MODE=auto` (default), Cleanplate first runs a fast CPU OpenCV detector (`static_overlay`) optimized for static corner badges and semi-transparent persistent edges. If confidence is uncertain or no static overlay is found, it escalates once to ComfyUI native `SAM 3.1` (`sam3_text`) to generate per-frame masks for moving watermarks.

### 5. Why does Cleanplate sometimes refuse to process a video (`DETECTOR_LOW_CONFIDENCE`)?
By design, Cleanplate avoids blind inpainting that would blur or corrupt clean footage. If the detected mask is overly fragmented, covers too much of the frame (`complex`), or `SAM 3.1` is unavailable when static detection is uncertain, the job fails cleanly with `DETECTOR_LOW_CONFIDENCE` so downstream pipelines can route it to human review or keep the original.

### 6. How do I scale Cleanplate horizontally across multiple GPUs?
Run one shared `cleanplate-api` instance (or shared storage/DB) and launch multiple `cleanplate-worker` processes, each configured with its own `DWM_COMFYUI_API_URL` pointing to a dedicated GPU ComfyUI instance (e.g., `http://127.0.0.1:8188`, `http://127.0.0.1:8189`). Workers coordinate job claims safely via file locks and database transactions.

---

## Repository Structure

```text
cleanplate/
├── src/wm_platform/   # Core service (API, worker, detection, chunking, QA, ComfyUI client, eval, scout)
├── apps/              # Process entrypoints for api and worker
├── workflows/         # ComfyUI API-format workflow JSON templates
├── .runtime/          # Pinned ComfyUI runtime lock.yaml and model manifest.yaml
├── eval/              # Clean-plate evaluation fixtures and model scouting snapshots
├── tests/             # pytest suite (mock ComfyUI, zero GPU required)
├── docs/              # Architecture, API, operations, model stack, scouting, roadmap
├── llms.txt           # Machine-readable project & API summary (llmstxt.org)
├── SECURITY.md        # Vulnerability reporting & security hardening notes
├── CONTRIBUTING.md    # Development workflow & PR guidelines
└── pyproject.toml     # Dependencies & CLI entrypoints (cleanplate-api / cleanplate-worker / cleanplate-scout)
```

---

## Documentation

- [docs/architecture.md](docs/architecture.md): Pipeline stages, modules, database schema, and extension points
- [docs/api.md](docs/api.md): HTTP endpoints, `report` schema, error codes, and HMAC webhooks
- [docs/operations.md](docs/operations.md): Deployment, first GPU run, health checks, scaling, backup, and rollback
- [docs/model-stack.md](docs/model-stack.md): Current models, licenses, candidates, and trade-offs
- [docs/model-scouting.md](docs/model-scouting.md): Model scouting workflow and evaluation gates
- [docs/roadmap.md](docs/roadmap.md): Completed milestones and upcoming work
- [CHANGELOG.md](CHANGELOG.md): Release history
