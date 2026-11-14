[中文文档 (README.md)](README.md) · **English (README.en.md)** · [llms.txt](llms.txt) · [codemeta.json](codemeta.json) · [API Contract](docs/api.md) · [Architecture](docs/architecture.md) · [Model Stack](docs/model-stack.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

# Cleanplate — Self-Hosted Async Video Watermark Detection, ROI Inpainting & QA Pipeline

**`Cleanplate` (v0.3.0) is an open-source (MIT), self-hosted asynchronous video watermark removal node built on FastAPI, ComfyUI, and FFmpeg for automated media workflows. Without requiring manual bounding boxes, it automatically detects static or moving watermarks, plans multi-ROI crops and 81-frame chunks, executes a multi-model inpainting fallback chain, remuxes original audio, runs an automated QA gate, and delivers clean outputs plus a structured `report` via polling or HMAC-SHA256 webhooks.**

[![CI](https://github.com/qilaidev/cleanplate/actions/workflows/ci.yml/badge.svg)](https://github.com/qilaidev/cleanplate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20HTTP-009688.svg)](docs/api.md)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.38.1-FF6F00.svg)](docs/model-stack.md)
[![Architecture](https://img.shields.io/badge/Worker-Async%20Queue-6E44FF.svg)](docs/architecture.md)

Named after the film post-production *clean plate* (a background frame with overlays removed). MIT-licensed service code, single-GPU ready, horizontally scalable across multiple workers.

> **Compliance & Licensing Notice**: Only process media you own or have explicit rights to modify. While Cleanplate's service code is **MIT** licensed, the default primary inpainting model weights (**MiniMax-Remover**) are **CC-BY-NC-4.0** (non-commercial) and **ProPainter** priors are **S-Lab License 1.0** (non-commercial). Review [Model Stack & Licenses](docs/model-stack.md#许可证商用前必看) before any commercial deployment.

---

## Entity Snapshot

| Field | Verified Specification |
| --- | --- |
| **Entity Name** | `Cleanplate` (`qilaidev/cleanplate` · v0.3.0) |
| **License Boundary** | Service code: **MIT License**; default weights carry non-commercial terms (`CC-BY-NC-4.0` / `S-Lab 1.0`) and are swappable |
| **Core Stack** | Python `>=3.11,<3.13` · FastAPI · Uvicorn · SQLite (WAL) · OpenCV · FFmpeg/ffprobe · ComfyUI `v0.38.1` · `uv` |
| **CLI Entrypoints** | `cleanplate-api` (HTTP API) · `cleanplate-worker` (async worker / installer / doctor / cleanup) · `cleanplate-scout` (model radar) |
| **6-Stage Pipeline** | 1. Ingest Normalization -> 2. Single-Pass Detection (OpenCV + SAM 3.1) -> 3. Multi-ROI & 81-Frame Chunking -> 4. Fallback Chain -> 5. A/V Stitch -> 6. QA & HMAC Webhook |
| **Quality & Safety Guards** | Checks `mask_mean_abs_diff` & `residual_overlap`; refuses blind inpaint (`DETECTOR_LOW_CONFIDENCE`); triggers fallback on `INPAINT_NO_EFFECT` |
| **Hardware & Verification** | All 139 `pytest` tests and `local_fallback` smoke runs execute CPU-only with mocked ComfyUI; real AI inpainting targets CUDA GPUs (~16–24 GB+ VRAM) & ~18–20 GB `.runtime/` |
| **Studio Ecosystem** | [qilai · qilaidav.com](https://qilaidav.com/en/) · [05 · Enterprise Requirement Ops & AI Foundation](https://qilaidav.com/en/workspace/) |

---

## Positioning: What It Is / What It Is Not

| Dimension | What It Is | What It Is Not |
| --- | --- | --- |
| **Service Role** | An async processing node inside automated media workflows (HTTP submit -> webhook callback / polling) | **Not** a consumer-facing online watermark removal website or desktop GUI editor |
| **Detection Model** | Automatic CPU static/edge corner detector + ComfyUI native `SAM 3.1` fallback running once per job | **Not** a manual-masking video tool (manual bbox/mask inputs are deliberately excluded from the API) |
| **Inpainting & QA** | Per-watermark ROI crops + 81-frame (`4k+1`) chunks + fallback chain + automated post-inpaint QA `report` | **Not** a single-model demo script that silently returns unchanged watermarked frames on errors |
| **Model Lifecycle** | A swappable model stack with standardized `cleanplate-scout` -> license gate -> clean-plate eval workflow | **Not** a pre-cleared commercial weight pack ready for unlicensed SaaS resale |
| **Scaling Model** | Single-host / single-GPU friendly, scaling horizontally via multiple `cleanplate-worker` processes | **Not** a multi-tenant SaaS platform with cluster orchestration or billing |

---

## Use Cases & Alternatives Comparison

### Primary Use Cases

- **Automated media processing pipelines**: Run as an async queue node receiving jobs via `POST /v1/jobs` (with `Idempotency-Key` support) and emitting structured JSON `report` payloads and HMAC-SHA256 webhooks.
- **Mixed static & moving watermark removal**: Avoid full-frame diffusion distortion and GPU overhead on clean corners by running fast CPU OpenCV detection first and escalating to `SAM 3.1` + multi-ROI chaining only when needed.
- **Reproducible model evaluation & commercial swap workflows**: Use bundled `eval/` clean-plate fixtures, `cleanplate-scout`, and `--doctor` license checks to benchmark and swap in commercially permissive models (e.g., Apache-2.0 `Wan2.1 VACE` or `CLEAR`).

### Architectural Comparison (Fact-Based, Zero Fabricated Benchmarks)

| Dimension | `Cleanplate` | Consumer Watermark Web Tools | Manual Mask Editors (AE / PR) | Raw ComfyUI GUI Scripts | FFmpeg `delogo` Filter |
|---|---|---|---|---|---|
| **Automation & API** | Async FastAPI queue + idempotency + HMAC webhooks | Manual web uploads | Manual timeline masking & export | Manual node graph & manual chunking | Requires hardcoded static coordinates |
| **Moving / Translucent Overlays** | CPU persistent-edge voting + `SAM 3.1` text-prompt tracking | Opaque vendor behavior | Manual keyframe rotoscoping | Requires custom graph wiring | Fixed static rectangle blur only |
| **Multi-Watermark Isolation** | Multi-ROI independent crop chaining (`expand_roi_chain`) | Typically single box or full-frame | Manual multi-mask layers | Single bounding box degrades to full-frame | Multiple hardcoded filters |
| **Silent-Failure Prevention** | Automated `quality_gate` (`INPAINT_NO_EFFECT` triggers fallback) | No structured QA report | Manual visual review | Nodes may silently pass input frames on OOM | No QA check |
| **A/V Sync & Normalization** | CFR normalization (VFR/rotation/odd size) + lossless audio remux | Often re-encodes or drops audio | Manual export settings | Prone to VFR drift / freeze-frame padding | Pure filter pass |

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

Every completed or failed job writes a structured `report` detailing input normalization, detection confidence/regions, chunk plan, per-provider attempts, and QA metrics so downstream systems can accept, review, or discard outputs deterministically. See [`docs/api.md`](docs/api.md).

---

## Quick Start

### 1. Zero-GPU Mock Quickstart (Tests & API Smoke Run)

Prerequisites: Python 3.11 or 3.12 (`>=3.11,<3.13`), [uv](https://docs.astral.sh/uv/), and `ffmpeg` / `ffprobe` in `PATH`.

```sh
uv sync --extra dev --locked
uv run ruff check src tests apps                # Lint & static checks
uv run pytest -q                                # 139 tests run under mock ComfyUI — no GPU required

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

> **Proxy Note**: If `HTTP_PROXY` / `HTTPS_PROXY` is set in your shell, export `NO_PROXY=127.0.0.1,localhost` so `--doctor` and `--comfyui-health` do not receive misleading `502 Bad Gateway` responses when probing `127.0.0.1:8188` (documented in [`.env.example`](.env.example)).

Once `GET /v1/providers` reports `minimax_remover.runnable=true`, submit jobs with `provider=auto`. Run `uv run cleanplate-worker --cleanup` (dry-run) and `--cleanup --execute-cleanup` periodically to prune expired job artifacts. See [`docs/operations.md`](docs/operations.md) for deployment, backups, and scaling.

---

## Configuration

All settings are configured via environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
| --- | --- | --- |
| `DWM_ENV` | `development` | Rejects the default dev API key when set to `production` |
| `DWM_DEFAULT_API_KEY` | `dev-secret-key` | Required `X-API-Key` header value |
| `DWM_API_HOST` / `DWM_API_PORT` | `127.0.0.1` / `8000` | Bind address and port for `cleanplate-api` |
| `DWM_STORAGE_ROOT` | `storage` | Root directory for `inbox`, `outbox`, `work`, `comfy_spool`, and `app.db` |
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

Full configuration reference is in [`src/wm_platform/config.py`](src/wm_platform/config.py).

---

## Model Stack & Licenses

| Role | Component | License | Notes |
| --- | --- | --- | --- |
| Detection (Default) | OpenCV Static/Edge Detector | MIT | Scans outer 22% border bands for corner/static overlays |
| Detection (Fallback) | SAM 3.1 (ComfyUI native) | SAM License | Handles moving/central text watermarks; requires GPU |
| Primary Inpainting | MiniMax-Remover 1.3B | **CC-BY-NC-4.0** | Non-commercial weights |
| Secondary Inpainting | DiffuEraser + ProPainter | Apache-2.0 / **S-Lab 1.0 (NC)** | Auto-downloads HF + GitHub Release ProPainter weights (SHA-256 verified) |

`--install-runtime --models` downloads ~14 GB of weights (~18–20 GB total `.runtime/` footprint). See [`docs/model-stack.md`](docs/model-stack.md) and [`docs/model-scouting.md`](docs/model-scouting.md) (`uv run cleanplate-scout --days 30`) for commercial alternatives and model evaluation.

---

## Frequently Asked Questions (FAQ)

### 1. What GPU and VRAM are required to run Cleanplate?
Real AI inpainting requires an NVIDIA CUDA GPU running ComfyUI (typically 16–24 GB+ VRAM for 81-frame diffusion chunks with `MiniMax-Remover` 1.3B or `DiffuEraser`). When VRAM is constrained, keep `DWM_MINIMAX_ROI_ENABLED=true` (default) to inpaint only ROI crops around watermarks or reduce `DWM_CHUNK_FRAMES` (Apple Silicon MPS is not a parity replacement for the CUDA baseline).

### 2. Can I run and test Cleanplate without a GPU or ComfyUI?
Yes. All 139 tests in `uv run pytest -q` run against a mock ComfyUI backend and real FFmpeg with zero GPU required. You can also launch `cleanplate-api` and `cleanplate-worker` locally and submit jobs with `provider=local_fallback` to verify auth, normalization, SQLite persistence, and HMAC webhooks end-to-end.

### 3. Can I use Cleanplate in a commercial product?
Cleanplate's **service code** is MIT licensed, so you can integrate, modify, and self-host it freely; however, the default **model weights** (`MiniMax-Remover` under CC-BY-NC-4.0 and `ProPainter` priors under S-Lab License 1.0) are **non-commercial**. For commercial deployments, obtain commercial weight licenses or swap in permissive models (such as Apache-2.0 `Wan2.1 VACE` or `CLEAR`, documented in [`docs/model-stack.md`](docs/model-stack.md)).

### 4. How does Cleanplate handle moving watermarks vs. static corner logos?
In `DWM_DETECTOR_MODE=auto` (default), Cleanplate first runs a fast CPU OpenCV detector (`static_overlay`) optimized for static corner badges and semi-transparent persistent edges along the outer 22% borders. If confidence is uncertain or no static overlay is found, it escalates once to ComfyUI native `SAM 3.1` (`sam3_text`) to generate per-frame masks for moving or central watermarks.

### 5. Why do jobs fail with `DETECTOR_LOW_CONFIDENCE` or trigger `INPAINT_NO_EFFECT`?
Both codes act as safety gates against silent video corruption. `DETECTOR_LOW_CONFIDENCE` prevents blind inpainting when a mask is too fragmented/large (`complex`) or when `SAM 3.1` is unavailable for uncertain scenes; `INPAINT_NO_EFFECT` catches cases where an inpainting node returns unchanged input frames (`mask_mean_abs_diff < 1.5`) and automatically triggers the fallback provider.

### 6. What security hardening applies to API keys, file paths, and webhooks?
Setting `DWM_ENV=production` forbids the default `dev-secret-key`, stores API keys as SHA-256 hashes, and enforces per-tenant submit rate limits (default 60/min). `input_path` submissions are confined to `DWM_STORAGE_ROOT`, `callback_url` rejects `localhost` and private IP ranges by default, and callbacks are signed with HMAC-SHA256.

### 7. How do I scale across multiple GPUs and clean up old job files?
Run one shared `cleanplate-api` instance and launch multiple `cleanplate-worker` processes, each configured with a distinct `DWM_COMFYUI_API_URL` (`http://127.0.0.1:8188`, `http://127.0.0.1:8189`, etc.) pointing to its own GPU ComfyUI process. Prune expired job files with `uv run cleanplate-worker --cleanup --execute-cleanup` (note that `.runtime/` itself occupies ~18–20 GB and is never removed by `--cleanup`).

---

## Repository Structure

```text
cleanplate/
├── src/wm_platform/   # Core service (API, worker, detection, chunking, QA, ComfyUI client, eval, scout)
├── apps/              # Process entrypoints for api and worker
├── workflows/         # ComfyUI API-format workflow JSON templates
├── .runtime/          # Pinned ComfyUI runtime lock.yaml and model manifest.yaml
├── eval/              # Clean-plate evaluation fixtures and model scouting snapshots
├── tests/             # pytest suite (139 tests, mock ComfyUI, zero GPU required)
├── docs/              # Architecture, API, operations, model stack, scouting, roadmap
├── llms.txt           # Machine-readable project & API summary (llmstxt.org)
├── codemeta.json      # CodeMeta 2.0 structured software metadata
├── SECURITY.md        # Vulnerability reporting & security hardening notes
├── CONTRIBUTING.md    # Development workflow & PR guidelines
├── .env.example       # Environment configuration template
└── pyproject.toml     # Dependencies & CLI entrypoints (cleanplate-api / cleanplate-worker / cleanplate-scout)
```

---

<details>
<summary><strong>Search Intent & Query Mapping</strong></summary>

| User Search Intent | Typical Queries (EN / ZH) | Verified Implementation Surface |
| --- | --- | --- |
| Self-hosted open-source video watermark removal API | self-hosted video watermark removal API · FastAPI async video inpainting · 开源视频去水印 API | [`src/wm_platform/api_app.py`](src/wm_platform/api_app.py) · [`docs/api.md`](docs/api.md) |
| ComfyUI headless video inpainting worker queue | ComfyUI MiniMax-Remover DiffuEraser pipeline · 81 frame chunk video inpainting · ComfyUI 去水印队列 | [`src/wm_platform/provider_runtime.py`](src/wm_platform/provider_runtime.py) · [`workflows/`](workflows/) |
| Automatic watermark detection & QA verification | automatic watermark mask detection SAM3 OpenCV · INPAINT_NO_EFFECT quality gate · 水印检测质检 | [`src/wm_platform/watermark_detect.py`](src/wm_platform/watermark_detect.py) · [`src/wm_platform/quality_gate.py`](src/wm_platform/quality_gate.py) |
| Licensing & swappable video inpainting model evaluation | MiniMax-Remover CC-BY-NC commercial alternative · cleanplate-scout · 模型商用许可评估 | [`docs/model-stack.md`](docs/model-stack.md) · [`src/wm_platform/model_scout.py`](src/wm_platform/model_scout.py) |

</details>

<sub>**Keywords**: self-hosted video watermark removal · FastAPI async video pipeline · ComfyUI video inpainting · MiniMax-Remover · DiffuEraser · SAM 3.1 watermark detection · multi-ROI chunked inpainting · automated video QA gate · FFmpeg audio remux · 开源视频去水印 · 自托管去水印 API</sub>

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md): Pipeline stages, modules, database schema, and extension points
- [`docs/api.md`](docs/api.md): HTTP endpoints, `report` schema, error codes, and HMAC webhooks
- [`docs/operations.md`](docs/operations.md): Deployment, first GPU run, health checks, scaling, backup, and rollback
- [`docs/model-stack.md`](docs/model-stack.md): Current models, licenses, candidates, and trade-offs
- [`docs/model-scouting.md`](docs/model-scouting.md): Model scouting workflow and evaluation gates
- [`docs/roadmap.md`](docs/roadmap.md): Completed milestones and upcoming work
- [`CHANGELOG.md`](CHANGELOG.md): Release history

---

## Architecture & Ecosystem · qilai (`qilaidav.com`)

This open-source project is engineered and maintained by **[qilai · qilaidav](https://qilaidav.com/en/)** — an independent systems engineering studio focused on **Deterministic Systems Engineering & High-Signal Technical Diagnostics**.

- **Pillar Alignment**: [05 · Enterprise Requirement Ops & AI Foundation (Workspace)](https://qilaidav.com/en/workspace/) · [中文专栏](https://qilaidav.com/workspace/)
- **Studio Website**: [https://qilaidav.com/en/](https://qilaidav.com/en/) · [中文主站](https://qilaidav.com/)
- **Technical Notes & Field Reports**: [https://qilaidav.com/notes/](https://qilaidav.com/notes/)
- **GitHub Open-Source Matrix**: [https://github.com/qilaidev](https://github.com/qilaidev)
- **Direct Engineering Contact**: [Telegram @BD8888](https://t.me/BD8888) · [Public Channel @AIKB8](https://t.me/AIKB8) · `hello@qilaidav.com`
