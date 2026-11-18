# Changelog

Format: Keep a Changelog. Versioning: SemVer.

## [0.4.0] - 2026-11-18

Production runtime hardening, lossless A/V & SDR color preservation, bounded SAM 3.1 streaming, OpenAPI 3.1 Agent-Ready contract, and Nov 2026 runtime/model re-survey. Verified across 156 automated tests against mocked ComfyUI plus real `ffmpeg`/`ffprobe`.

### Added

- **Clean-chunk GPU bypass** (`provider_runtime.run_comfy_chunks`, `media.extract_frame_window`): for temporal/intermittent overlays, 81-frame windows whose `active_frames` mask is empty bypass ComfyUI diffusion completely and are sliced directly via FFmpeg (`report.plan.inpainted_chunks` / `bypassed_chunks`).
- **SHA-256 verified multi-source weight installer** (`runtime_installer.py`, `.runtime/models/manifest.yaml`): `--install-runtime --models` now downloads `ProPainter.pth`, `raft-things.pth`, and `recurrent_flow_completion.pth` directly from GitHub Releases with streaming SHA-256 verification and atomic `.tmp` replacement alongside Hugging Face snapshots.
- **Bounded SAM 3.1 VRAM & O(1) RAM temporal mask streaming** (`watermark_detect.py`): caps full-clip SAM 3.1 tracking to `<=180` strided frames at `<=960p` and streams `+-1` neighbor temporal mask padding with at most 3 frames in RAM.
- **Lossless audio muxing, SDR color-tag preservation & tail-frame protection** (`media.py`): container-aware lossless stream copy (`-c:a copy` for MP4/MOV/MKV compatible codecs), demux `-t` audio clamping instead of `-shortest` so tail video frames are never truncated, SDR VUI/filtergraph `setparams` color-tag preservation (`bt709` / `smpte170m`), `+faststart` MP4 output, and duration-scaled media timeouts (`DWM_MEDIA_TIMEOUT_SECONDS`).
- **Isolated ComfyUI spool, stage watchdog & orphan prompt cancellation** (`comfy_jobs.py`, `worker_service.py`, `maintenance.py`): isolates `--output-directory` to `storage/comfy_spool` (`DWM_COMFYUI_OUTPUT_DIR`), cancels/interrupts orphaned ComfyUI prompts (`/queue` delete + `/interrupt`) on client timeout, and adds a stage-progress watchdog (`DWM_JOB_WATCHDOG_TIMEOUT_SECONDS`) so stalled worker claims are reclaimed automatically.
- **Loopback proxy bypass & `--disable-api-nodes` hardening** (`comfy_runtime.py`): local ComfyUI HTTP calls (`127.0.0.1`, `localhost`, `::1`) automatically set `trust_env=False` to prevent `HTTP_PROXY` `502 Bad Gateway` misrouting, and managed ComfyUI startups pass `--disable-api-nodes`.
- **SSRF & OpenAPI 3.1 contract hardening** (`callbacks.py`, `api_app.py`, `dependencies.py`): `validate_callback_url` blocks RFC 6598 CGNAT (`100.64.0.0/10`), benchmark (`198.18.0.0/15`), IPv4-mapped IPv6 literals, and all non-global IPs; `/openapi.json` and `/docs` expose `ApiKeyAuth` (`X-API-Key`), typed health/provider schemas, and stable `operationId`s across all 8 endpoints.
- **Python 3.13 support, PEP 735 `[dependency-groups]`, and `CITATION.cff`**.

### Changed

- Runtime lock upgraded to `Comfy-Org/ComfyUI` **v0.45.0** (`989d93d0`, 2026-11-10) and `smthemex/ComfyUI_DiffuEraser` **`9ef5c12f`** (2026-10-23, ComfyUI-native `ModelPatcher` core without legacy `diffusers` dependency conflicts); `workflows/diffueraser_mask_api.json` updated to match V3 node inputs.
- Re-surveyed model stack and benchmarks as of **2026-11-18** (`eval/scouting/2026-11-18-auto.md`, `docs/model-stack.md`).
- Docs record measured `.runtime/` footprint (~18–20 GB total; never removed by `--cleanup`), Apple Silicon MPS boundaries vs. 16–24 GB+ CUDA GPUs, and proxy troubleshooting (`NO_PROXY=127.0.0.1,localhost` in `.env.example`).

## [0.3.0] - 2026-09-30

Batch-pipeline hardening, model re-survey, repository cleanup. Verified against a mocked ComfyUI plus real ffmpeg; real-GPU evaluation is still the next milestone.

### Added

- Ingest normalization (`media.normalize_input`): VFR, rotation side data, odd dimensions or non-yuv420p inputs are re-encoded to a constant-frame-rate intermediate so detector, VideoHelperSuite and inpainter index the same frames.
- Chunked inference: `DWM_CHUNK_FRAMES` (default 81) windows of 4k+1 frames, last window slid back; overlap trimmed and source audio muxed once at assembly. MiniMax-Remover previously received the whole clip in one diffusion call.
- Multi-ROI inpainting: separate overlays get separate crop windows, chained in one prompt (`comfy_graph.expand_roi_chain`).
- Post-inpaint quality gate (`quality_gate.py`): unchanged output inside the mask (ComfyUI-MiniMax-Remover returns the input frames on any exception) fails the attempt with `INPAINT_NO_EFFECT` and the next provider runs; a residual static overlay is flagged `needs_review` or fails with `QA_RESIDUAL_WATERMARK` (`DWM_QA_RESIDUAL_POLICY`).
- `NO_WATERMARK_DETECTED` (every detector ran and saw nothing) separated from `DETECTOR_LOW_CONFIDENCE`; `DWM_NO_WATERMARK_POLICY=passthrough` returns the input unchanged.
- Per-job `report` (input, detection, plan, attempts, QA, action) on `GET /v1/jobs/{id}`, `/result` and callbacks; stored in `jobs.report_json`.
- Model licenses in the manifest (`license`, `commercial_use`) and a `licenses` section in `--doctor`.
- `cleanplate-scout`: recent-window sweep over Hugging Face, arXiv and GitHub; playbook in `docs/model-scouting.md`.
- CI (ruff + pytest on 3.11/3.12), ruff config, `.env.example`.

### Changed

- Detection runs once per job and is shared by every provider; masks are lossless FFV1 `.mkv`. Default `DWM_DETECTOR_MODE` is now `auto`.
- ComfyUI wait timeout is `DWM_COMFYUI_PROMPT_TIMEOUT_SECONDS` (per chunk, default 1800 s); callback retries back off exponentially.
- `--cleanup` also removes stale `storage/work/` directories.
- Runtime lock: ComfyUI `Comfy-Org/ComfyUI` v0.38.1 (native SAM3 nodes unchanged since v0.35).
- Workflows no longer wire audio into `CreateVideo`; audio is muxed after assembly.
- Docs consolidated into README + architecture / api / operations / model-stack / model-scouting / roadmap; positioning is "video watermark removal service inside a workflow".
- Eval fixtures moved from `plans/eval_dataset` to `eval/dataset`.

### Removed

- `archive/` (retired workflows and plans; recoverable from commit `c29d144`), `llms.txt`, `llms-full.txt`, `README.en.md`, and the overview / FAQ / requirements / index / usage-examples docs whose content was merged or outdated.

## [0.2.0] - 2026-09-11

Detection / inpainting upgrade and legacy isolation. Verified only against a mocked ComfyUI; real-GPU evaluation is the next milestone.

### Added

- SAM 3.1 detection on ComfyUI's native nodes (`workflows/sam3_native_api.json`, checkpoint `models/checkpoints/sam3.1_multiplex_fp16.safetensors`).
- Per-frame masks for moving overlays (`scene_type=moving_overlay`); static overlays collapse to one mask. `DetectionResult` gains `bbox`, `frame_size`, `temporal`.
- Persistent-edge path in the static detector: Canny edges voted across frames, closed and filled. Detects translucent overlays over a moving background.
- ROI inpainting for MiniMax-Remover (`workflows/minimax_remover_roi_api.json`): a 16 px-aligned crop around the overlay is inpainted and composited back with a feathered mask; pixels outside stay bit-exact. Env: `DWM_MINIMAX_ROI_ENABLED`, `DWM_MINIMAX_ROI_MIN_SIDE`, `DWM_MINIMAX_ROI_MAX_FRAME_FRACTION`.
- Clean-plate eval metrics (`wm_platform.eval_metrics`): fixtures ship lossless `clean/*.mkv` and `clean/*.region.png`; eval rows carry region PSNR/SSIM, flicker, outside-region PSNR and residual re-detection, with automatic judgments for AI providers (manual CSV still overrides).
- Static corner/logo detector, video duration limit, job max attempts, per-job `quality_mode`, `GET /v1/jobs/{id}/artifact` with `download_url`.
- In-repo eval fixtures (`python -m wm_platform.eval_fixtures`), `eval_runner --report`, `eval_judge`.

### Changed

- Default AI chain is `minimax_remover`; `comfy_diffueraser` is secondary and consumes the same detector mask. `provider=auto` never succeeds via `ffmpeg_copy`.
- Low-confidence detection fails with `DETECTOR_LOW_CONFIDENCE`; no mask is ever invented.
- Runtime lock: ComfyUI v0.35.0, VideoHelperSuite 2026-09-02, ComfyUI_DiffuEraser 2026-03-27. Manifest adds the `sam3.1-native` checkpoint.
- Submit validation treats blank optional fields as absent, bounds `priority`, rate-limits by tenant. Artifact download returns `409 JOB_NOT_READY` until success and refuses paths outside `DWM_STORAGE_ROOT`.
- Quality profiles use watermark-sized mask dilation and add ROI composite expand/feather.
- Project renamed from `dewatermark-platform` to `Cleanplate`; CLI entry points are `cleanplate-api` and `cleanplate-worker`.
- `docs/model-stack.md` re-surveyed (VOID, BridgeRemoval, DualEraser, Object-WIPER, CLEAR, PROVE, Wan2.2 VACE, YOLO watermark detectors).

### Removed

- RMBG/PreData segmentation path: `DWM_ALLOW_RMBG_LEGACY`, `DWM_COMFYUI_SEGMENTATION_REPO`, per-profile DiffuEraser workflows. DiffuEraser has one workflow, `diffueraser_mask_api.json` (`DWM_COMFYUI_DIFFUERASER_WORKFLOW` default); `DWM_COMFYUI_DIFFUERASER_MASK_WORKFLOW` is gone.
- Custom-node SAM3 stack and unused `sam2` runtime/model entries. Probe details drop `sam3_stack`, `segmentation_repo`, `allow_rmbg_legacy`.
- Stale planning docs and the project report page. Retired files live under `archive/2026-09-legacy/`; the runtime never reads that directory.
- Repository URLs, badges and SEO keyword sections in README and docs.

## [0.1.0] - 2026-05-19

First tagged release: the platform MVP.

- API + worker architecture (FastAPI, SQLite WAL, async worker), local-first.
- Provider routing with `provider=auto` failover: `comfy_diffueraser` (primary at the time) and `local_fallback` (`ffmpeg_copy`, `delogo`).
- Quality profiles `fast` / `balanced` / `quality` / `corner_hq`.
- Runtime contract: `doctor` / `runtime-plan` / `install-runtime` / `comfyui-plan` / `start-comfyui`.
- Job API: submit, list, status, result, cancel, providers, health; `Idempotency-Key`; per-key rate limit.
- HMAC-SHA256 callbacks with retry; private callback URLs blocked by default.
- File retention cleanup; run metadata (workflow, profile, device, seed).
- Bilingual READMEs, `llms.txt`, docs for FAQ / API / architecture / requirements / roadmap.
