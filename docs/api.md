# Cleanplate API | HTTP API 与 OpenAPI 3.1 合同

`Cleanplate`（v0.4.0）的 HTTP 接口：异步提交、状态查询、结果与处理报告、provider 探测、HMAC 回调。服务启动后自动暴露机器可读的 `/openapi.json`（含 `ApiKeyAuth` 鉴权定义与固定 `operationId`）及交互式文档 `/docs`。

## 0. 鉴权

所有 `/v1/*` 接口都需要 `X-API-Key` 请求头。

```
X-API-Key: dev-secret-key
```

API Key 在服务端以 SHA-256 哈希存储，不会明文保存。

生产部署必须设置：

```sh
export DWM_ENV=production
export DWM_DEFAULT_API_KEY="replace-with-a-long-random-secret"
```

当 `DWM_ENV=production` 且仍使用默认开发 key `dev-secret-key` 时，服务启动会失败。

## 0.1 速率限制

`POST /v1/jobs` 接口有速率限制：

- 默认: 60 次/分钟 per API key
- 可通过 `DWM_SUBMIT_RATE_LIMIT_COUNT` 和 `DWM_SUBMIT_RATE_LIMIT_WINDOW_SECONDS` 配置
- 超出限制返回 `429 RATE_LIMITED`，并带 `Retry-After`（秒，等于限流窗口）
- 计数按租户（API Key 对应的 `tenant_id`），不是按原始 key 字符串

## 1. Submit Job

`POST /v1/jobs`

用途：

- 创建去水印任务

请求字段：

- `media_type`
  - `video` (当前提交接口唯一支持的类型)
  - `image` (模型层预留，提交接口当前会返回 `MEDIA_TYPE_NOT_SUPPORTED`)
- `provider`
  - `auto` (默认。检测只跑一次；没有水印 → `NO_WATERMARK_DETECTED`（或按 `DWM_NO_WATERMARK_POLICY=passthrough` 原样返回），看不准 → `DETECTOR_LOW_CONFIDENCE`，两种情况都不会换模型重试。有水印时先走 `minimax_remover`，执行失败或质检不过再走 `comfy_diffueraser`；默认不 copy)
  - `minimax_remover`
  - `comfy_diffueraser`
  - `local_fallback`
- `callback_url`
- `callback_secret`
- `input_path` (必须位于 `DWM_STORAGE_ROOT` 下, 与 file 二选一)
- `priority` (默认 0, 越高越优先；范围 `-1000`–`1000`)
- `quality_mode` (可选, `fast` / `balanced` / `quality` / `corner_hq`；缺省用 `DWM_QUALITY_MODE`；非法值返回 `VALIDATION_ERROR`)
- 空字符串的 `callback_url` / `callback_secret` / `input_path` / `Idempotency-Key` 视为未提供

请求头：

- `X-API-Key`
- `Idempotency-Key` (可选, 用于幂等提交)

说明：

- 当前支持 `multipart/form-data` 上传文件
- 如果传 `input_path`，只允许 `DWM_STORAGE_ROOT` 下的本机文件路径
- 如果传文件上传，服务端负责落盘到 `storage/inbox/`
- 如果传 `callback_url`，默认只允许公网全局可达 `http/https` 地址；`localhost`、私网、链路本地、CGNAT（`100.64.0.0/10`）与 IPv4-mapped IPv6 地址默认全部拒绝
- 幂等: 相同 `Idempotency-Key` + 相同输入和参数 → 返回已有 job，不重复创建
- 如果幂等键已使用但参数不同 → 返回 `409 IDEMPOTENCY_CONFLICT`
- 幂等写入在数据库写事务内按租户串行化，避免并发请求用同一 key 创建多条不同 job

成功响应示例：

```json
{
  "job_id": "job_123",
  "status": "queued",
  "provider_requested": "auto",
  "created_at": "2026-01-01T00:00:00"
}
```

返回字段包括：

- `job_id`
- `status`
- `provider_requested`
- `quality_mode`
- `created_at`

## 2. Get Job

`GET /v1/jobs/{job_id}`

返回字段包括：

- `job_id`
- `status`
- `media_type`
- `provider_requested`
- `provider_selected` (实际执行的 provider)
- `degraded` (`provider_selected` 为 `local_fallback` 时为 `true`，表示 copy/delogo 保活而非 AI 修复)
- `output_path`
- `attempt_count`
- `error_code`
- `error_message`
- `created_at`
- `updated_at`
- `quality_mode` (提交时写入的质量档，缺省为当时的 `DWM_QUALITY_MODE`)
- `needs_review` (`true` 表示产物已生成但质检怀疑水印残留，建议人工复核或丢弃)
- `report` (处理报告，见下节；任务开始执行前为 `null`)

## 2.1 处理报告 `report`

worker 在任务结束（成功或失败）时写入。下游应以 `status` + `needs_review` 决定是否使用产物，`report` 用于排查和统计。

```json
{
  "chain": ["minimax_remover", "comfy_diffueraser"],
  "input": {"width": 1080, "height": 1920, "fps": 30.0, "frame_count": 450, "duration": 15.0,
            "has_audio": true, "codec": "hevc", "pix_fmt": "yuv420p10le", "rotation": 90,
            "variable_frame_rate": true, "normalized": true,
            "normalize_reasons": ["variable_frame_rate", "rotation_90", "pix_fmt_yuv420p10le"]},
  "detection": {"detector": "static_overlay", "scene_type": "corner", "confidence": "high",
                "mask_ratio": 0.012, "region_count": 2, "bbox": [24, 40, 1010, 1830],
                "regions": [[24, 40, 180, 60], [860, 1790, 174, 80]], "temporal": false,
                "message": "static overlay detected"},
  "plan": {"chunks": 6, "chunk_frames": 81, "inpainted_chunks": 6, "bypassed_chunks": 0},
  "attempts": [
    {"provider": "minimax_remover", "status": "failed", "error_code": "INPAINT_NO_EFFECT",
     "workflow_name": "minimax_remover_roi_api.json", "roi": "0,0,320,256;752,1664,320,256",
     "qa": {"status": "unchanged", "mask_mean_abs_diff": 0.21}},
    {"provider": "comfy_diffueraser", "status": "succeeded",
     "workflow_name": "diffueraser_mask_api.json",
     "qa": {"status": "pass", "mask_mean_abs_diff": 38.4, "residual_overlap": 0.02}}
  ],
  "action": "inpainted",
  "qa": {"status": "pass", "mask_mean_abs_diff": 38.4, "residual_overlap": 0.02},
  "needs_review": false
}
```

| 字段 | 取值 |
| --- | --- |
| `detection.scene_type` | `corner` / `static_logo` / `moving_overlay` / `sam3_text` / `none`（确认无水印）/ `complex`（区域过大或过碎）/ `unknown` |
| `detection.confidence` | `high` / `medium` / `low` |
| `attempts[].status` | `succeeded` / `failed` / `residual_suspected` / `not_runnable` / `not_registered` |
| `action` | `inpainted` / `passthrough`（无水印原样返回）/ `copied`（`local_fallback`） |
| `qa.status` | `pass` / `unchanged` / `residual_suspected` / `skipped` |

## 3. List Jobs

`GET /v1/jobs`

当前支持过滤：

- `status`
- `provider`
- `media_type`
- `page`，默认 `1`
- `page_size`，默认 `50`，最大 `200`

响应包括：

- `jobs`
- `page`
- `page_size`
- `has_more`

## 4. Get Job Result

`GET /v1/jobs/{job_id}/result`

返回：

- `job_id`
- `status`
- `provider_selected`
- `degraded` (仅 `local_fallback` 为 `true`)
- `output_path`
- `download_url` (`/v1/jobs/{job_id}/artifact`)
- `needs_review`
- `report`

`GET /v1/jobs/{job_id}/artifact` 用同一把 API Key 下载产物。

如果 job 成功但产物文件已被删除 → 返回 `410 ARTIFACT_MISSING`。

## 4.1 Download Artifact

`GET /v1/jobs/{job_id}/artifact`

用 `X-API-Key` 下载成功任务的输出文件。路径必须位于 `DWM_STORAGE_ROOT` 下。

- 任务尚未成功 → `409 JOB_NOT_READY`
- 成功但文件已删或不在存储根下 → `410 ARTIFACT_MISSING`

## 5. Cancel Job

`POST /v1/jobs/{job_id}/cancel`

当前行为：

- `queued` 可以取消并返回 `canceled`
- `running` 返回 `409 JOB_NOT_CANCELABLE`
- 其他不可取消状态返回对应错误

## 6. Health

`GET /healthz`

返回：

- API 是否存活
- DB 是否可用

## 7. Providers Health

`GET /v1/providers`

返回每个 provider 的探测结果：

- `installed`
- `runnable`
- `message`
- `details`

这个接口很关键，因为本地去水印系统最怕"看起来接上了，实际上 provider 不能跑"。

对于 `comfy_diffueraser`，`details` 包含：

- `api_url`
- `auto_start_comfyui`
- `comfyui_dir`
- `venv_python`
- `custom_nodes_dir`
- `models_dir`
- `workflow_path`
- `workflow_name`
- `quality_profile` (当前使用的 quality profile)
- `runtime_root`
- `missing_installation_bits`
- `missing_models`
- `workflow_ready`
- `workflow_issue`
- `api_issue`
- `automatic_ai_pipeline`
- `detector` (`static_overlay` / `sam3_text` / `auto`)
- `sam3_available` / `sam3_issues` (ComfyUI 原生 SAM 3.1 节点 + `models/checkpoints/sam3.1_multiplex_fp16.safetensors`)
- `roi_enabled` / `roi_workflow_name` / `roi_workflow_ready` / `roi_workflow_issue` (MiniMax ROI 裁剪修复链)

探测结果会被缓存 (默认 10s)，缓存 key 包含 runtime_root、comfyui_api_url、local_fallback_mode、quality_mode 等。

## 8. Callback Payload

回调 body 固定为：

```json
{
  "job_id": "job_123",
  "status": "succeeded",
  "provider": "minimax_remover",
  "degraded": false,
  "output_path": "/abs/path/to/output.mp4",
  "error_code": null,
  "error_message": null,
  "needs_review": false,
  "report": { "...": "同 2.1" }
}
```

回调投递失败按 `DWM_CALLBACK_RETRY_DELAY_SECONDS × 2^n` 指数退避（上限 1 小时），共 `DWM_CALLBACK_RETRY_COUNT` 次。

如果设置了 `callback_secret`，回调头里带：

- `X-Signature` (HMAC-SHA256 签名)
- `X-Timestamp`

签名算法: `HMAC-SHA256("{timestamp}.{body}", secret)`

## 9. 错误响应

错误响应结构：

```json
{
  "error_code": "VALIDATION_ERROR",
  "error_message": "media_type is required"
}
```

错误码列表：

| 错误码 | HTTP 状态 | 说明 |
|--------|-----------|------|
| `VALIDATION_ERROR` | 400 | 请求参数校验失败 |
| `MEDIA_TYPE_NOT_SUPPORTED` | 400 | 仅支持 video，且扩展名为 mp4/mov/mkv |
| `FILE_TOO_LARGE` | 400 | 上传超过 `DWM_MAX_UPLOAD_BYTES` |
| `VIDEO_TOO_LONG` | 400 | 视频时长超过 `DWM_MAX_VIDEO_SECONDS` |
| `FILE_NOT_FOUND` | 400/404 | `input_path` 越出存储根（400）或不存在（404） |
| `AUTH_ERROR` | 401 | API Key 缺失或无效 |
| `JOB_NOT_FOUND` | 404 | Job 不存在或无权访问 |
| `JOB_NOT_CANCELABLE` | 409 | Job 状态不允许取消 |
| `JOB_NOT_READY` | 409 | 任务尚未成功，不能下载产物 |
| `IDEMPOTENCY_CONFLICT` | 409 | 幂等键已使用但参数不同 |
| `NO_WATERMARK_DETECTED` | 422 | 所有检测器都确认没有水印（`DWM_NO_WATERMARK_POLICY=fail` 时） |
| `DETECTOR_LOW_CONFIDENCE` | 422 | 检测不确定（区域过大/过碎、SAM 3.1 不可用等），不瞎修 |
| `INPAINT_NO_EFFECT` | 500 | 所有修复模型输出在水印区域与输入无差别（常见于显存不足被节点吞掉） |
| `QA_RESIDUAL_WATERMARK` | 500 | 修复后仍检测到水印且 `DWM_QA_RESIDUAL_POLICY=fail` |
| `MEDIA_PROCESSING_FAILED` | 500 | ffprobe/ffmpeg 规范化或拼接失败 |
| `RATE_LIMITED` | 429 | 提交频率超限 |
| `ARTIFACT_MISSING` | 410 | Job 成功但产物文件已删除或不在存储根下 |
| `PROVIDER_RUN_FAILED` | 500 | Provider 执行失败 |
| `JOB_RETRY_EXHAUSTED` | 500 | 任务重试次数用尽 |
| `INTERNAL_ERROR` | 500 | 内部服务器错误 |
| `PROVIDER_NOT_AVAILABLE` | 503 | Provider 不可用 |

任务级错误（`NO_WATERMARK_DETECTED` 及以下）不是 HTTP 响应，而是任务 `status=failed` 时的 `error_code`；表中 HTTP 状态仅表示语义类别。

## 10. 接入建议

- 用 `Idempotency-Key`（例如上游的素材 ID）提交，重试不会产生重复任务。
- 以回调为主、轮询为辅；回调验签后按 `job_id` 去重。
- 处理结果三分：`succeeded && !needs_review` 直接用；`needs_review` 转人工或降级使用；`failed` 看 `error_code`：`NO_WATERMARK_DETECTED` 可直接用原片，`DETECTOR_LOW_CONFIDENCE` / `QA_RESIDUAL_WATERMARK` 转人工，`PROVIDER_*` / `INPAINT_NO_EFFECT` 可稍后重投。

## 11. Quality Profiles (高级)

`POST /v1/jobs` 可选 form 字段 `quality_mode=fast|balanced|quality|corner_hq`。缺省使用 `DWM_QUALITY_MODE`（默认 `balanced`）。该值写入 job，MiniMax / DiffuEraser 按任务读取。

| 值 | 用途 |
|----|------|
| `fast` | 最快处理, 质量较低 |
| `balanced` | 默认, 平衡速度和质量 |
| `quality` | 高质量, 较慢 |
| `corner_hq` | 角落水印专用高质量模式 |

每个 profile 注入不同的 steps / subvideo_length / neighbor_length / mask_dilation_iter / MiniMax steps / ROI 贴回的 expand 与 feather。DiffuEraser 只有一个 workflow（外部 mask），profile 只改参数。
