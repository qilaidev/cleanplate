# 运维手册

面向单机（可多 GPU）自托管部署：上线门槛、配置、GPU 首跑、扩容、健康检查、备份、回滚。

## 1. 上线前门槛

上线前必须满足：

- `uv run pytest` 全量通过。
- `uv run cleanplate-worker --doctor` 可运行，并保存输出作为上线前快照。
- `GET /healthz` 返回 API 和 DB 正常。
- `GET /v1/providers` 中至少一个目标 provider 是 `runnable=true`。
- `DWM_ENV=production` 时必须设置非默认 `DWM_DEFAULT_API_KEY`。
- `DWM_STORAGE_ROOT`、`DWM_DB_PATH`、`.runtime` 路径位于可持久化磁盘。
- 已配置数据库和文件目录备份。
- 已准备回滚版本、旧进程停止方式和旧数据恢复方式。
- `--doctor` 的 `licenses.commercial_ready` 为 `false` 时，确认业务是否属于商用；属于的话先解决 MiniMax-Remover / ProPainter 的授权或换模型（见 model-stack.md）。
- 用评测集在目标 GPU 上跑一遍 `eval_runner --split all --report`，保存报告作为基线。

## 2. 推荐生产配置

最小生产环境变量：

```sh
export DWM_ENV=production
export DWM_DEFAULT_TENANT_ID="tenant-main"
export DWM_DEFAULT_API_KEY="replace-with-a-long-random-secret"
export DWM_STORAGE_ROOT="/srv/cleanplate/storage"
export DWM_DB_PATH="/srv/cleanplate/storage/app.db"
export DWM_RUNTIME_ROOT="/srv/cleanplate/runtime"
export DWM_COMFYUI_API_URL="http://127.0.0.1:8188"
export DWM_ALLOW_PRIVATE_CALLBACK_URLS=false
export DWM_SUBMIT_RATE_LIMIT_COUNT=60
export DWM_SUBMIT_RATE_LIMIT_WINDOW_SECONDS=60
export DWM_FILE_RETENTION_DAYS=7
export DWM_DETECTOR_MODE=auto
export DWM_CHUNK_FRAMES=81
export DWM_NO_WATERMARK_POLICY=fail      # 或 passthrough：确认无水印时原样返回
export DWM_QA_RESIDUAL_POLICY=flag       # 或 fail：残留直接判失败
```

完整示例见仓库根目录 `.env.example`。

生产模式保护：

- 当 `DWM_ENV=production` 且 `DWM_DEFAULT_API_KEY=dev-secret-key` 时，API/worker 启动会失败。
- 生产默认拒绝 localhost、私网、链路本地、保留地址和组播地址作为 callback 目标。
- 需要本地调试私网 callback 时，只在开发环境设置 `DWM_ALLOW_PRIVATE_CALLBACK_URLS=true`。

## 3. 启动与进程模型

API 和 worker 必须分进程运行：

```sh
uv run cleanplate-api --host 127.0.0.1 --port 8000
uv run cleanplate-worker --log-level INFO
```

推荐用 systemd、supervisord、launchd 或容器编排托管进程。托管配置至少要包含：

- 自动重启。
- stdout/stderr 日志采集。
- 明确的工作目录和环境变量文件。
- 停止 API 和 worker 的独立命令。
- 部署前后执行 `--doctor` 和 `/healthz` 检查。

## 3.1 云 GPU 首跑

机型建议：24GB+ VRAM（A10 / A5000 / 4090 级即可跑 MiniMax；VOID 以后再说）。  
系统：Ubuntu 22.04，驱动 + CUDA 与 PyTorch 匹配。

步骤：

```sh
uv sync
uv run cleanplate-worker --install-runtime --models
uv run cleanplate-worker --start-comfyui
uv run cleanplate-worker --doctor
uv run cleanplate-api --host 0.0.0.0 --port 8000
# 另开进程
uv run cleanplate-worker
# GET /v1/providers 确认 minimax_remover.runnable
```

`--install-runtime` 默认只装仓库和 Python 依赖，不会拉权重。加上 `--models` 才会按 `.runtime/models/manifest.yaml` 下载。只要权重时可用 `--models-only`。

环境变量最少集：

```sh
export DWM_ENV=production
export DWM_DEFAULT_API_KEY=<强密钥>
export DWM_STORAGE_ROOT=/data/cleanplate/storage
export DWM_RUNTIME_ROOT=/data/cleanplate/runtime
export DWM_COMFYUI_API_URL=http://127.0.0.1:8188
export DWM_AUTO_START_COMFYUI=true
export DWM_QUALITY_MODE=balanced
export DWM_AUTO_ALLOW_COPY_FALLBACK=false
```

不要在本机文档演练里启动生产服务或下载真实权重；以上命令只在已租好的云 GPU 上执行。

ComfyUI 没有官方 Docker 镜像，也没有内置鉴权：`cleanplate-worker --start-comfyui` 默认固定传入 `--listen 127.0.0.1 --output-directory <comfy_spool> --disable-auto-launch --disable-api-nodes`（隔离临时推理产物并禁用外部云端 API 节点）；需要跨机访问时前面加带鉴权的反向代理。

## 3.2 吞吐与扩容

- 一个 worker 同时处理一个任务；ComfyUI 同时执行一个 prompt。
- 一台 GPU：1 个 ComfyUI + 2 个 worker。第二个 worker 在 CPU 上做规范化、检测、拼接、质检，和另一个任务的 GPU 推理重叠。
- 多台 GPU（同一台机器）：每张卡起一个 ComfyUI（`CUDA_VISIBLE_DEVICES=N`，不同端口），每个 ComfyUI 配 1–2 个 worker，用各自的 `DWM_COMFYUI_API_URL`。SQLite 队列在单机多进程下没有问题。
- 分段越大越快（每段有模型加载与编解码开销）、显存越高。24 GB 卡上 ROI 模式可以试 `DWM_CHUNK_FRAMES=161`，整帧 1080p 保持 81。
- `DWM_COMFYUI_PROMPT_TIMEOUT_SECONDS` 是单段的等待上限，包含在 ComfyUI 队列里排队的时间；多个 worker 共用一个 ComfyUI 时要留足。
- 中间文件在 `storage/work/<job_id>/`，任务结束即删除；worker 崩溃遗留的目录由 `--cleanup` 回收。磁盘预留：同时在跑的任务数 × 输入大小 × 3。

## 3.3 非 CUDA 机器（Apple Silicon）

M 系列没有 CUDA，ComfyUI 只能走 MPS。能装、能起，但不要按 3.1 的机型口径预期：

- 主链 `minimax_remover`（transformer 2.25 GB + VAE 0.5 GB）在 16 GB 统一内存上只适合短片段；`DWM_CHUNK_FRAMES=81` 在 MPS 上明显慢于同价位 CUDA 卡，需要调小。
- 次级链 `comfy_diffueraser` 要常驻 SD1.5 的 `unet_main` 5.25 GB + `brushnet` 3.54 GB，16 GB 统一内存基本顶到上限，不建议在这类机器上启用。
- 两条链目前只在 mock ComfyUI + 真实 ffmpeg 下验证过，真 GPU 评测仍是下一步，效果数字一律不承诺。

## 4. 健康检查

API 存活：

```sh
curl -fsS http://127.0.0.1:8000/healthz
```

Provider 和生产安全检查：

```sh
uv run cleanplate-worker --doctor
curl -fsS -H "X-API-Key: $DWM_DEFAULT_API_KEY" http://127.0.0.1:8000/v1/providers
```

`--doctor` 输出包含：

- `system_dependencies`: SQLite、git、ffmpeg。
- `providers`: `minimax_remover`、`comfy_diffueraser` 和 `local_fallback` 的安装、运行和缺失项。探测应区分节点缺失、权重缺失、ComfyUI API 不通。
- `model_status`: 清单里每项权重是否已落到 `expected_path`。
- `production_safety`: 生产配置风险，如默认开发 key、生产放开私网 callback、关闭限流、保留时间过短、inbox/outbox 所在磁盘空间不足（低于 5GB 或 10%）。

### 4.1 环境变量代理与 `502 Bad Gateway` 排查

自 v0.4.0 起，Cleanplate 访问本机回环地址（`127.0.0.1` / `localhost` / `[::1]`）的 ComfyUI 时已自动在客户端设置 `trust_env=False` 绕过系统代理。但若你的 ComfyUI 部署在非回环内网地址，或你在终端直接用 `curl` 排查且环境里设了 `HTTP_PROXY` / `HTTPS_PROXY`，请求仍可能被送去代理并收到误导性的 `502 Bad Gateway`（响应体 `upstream connect failed: Connection refused`）。先用带 `--noproxy` 的 curl 直接验一次：

```sh
curl --noproxy '*' -i http://127.0.0.1:8188/system_stats
```

排查顺序：

1. 确认 ComfyUI 进程起来了（见 `--comfyui-plan` 输出的命令）。
2. 探测前绕过代理：`export NO_PROXY=127.0.0.1,localhost`，并把 `no_proxy` 写进托管服务的环境变量文件。
3. 再看 `--doctor` 的 `api_issue` 是否消失。

## 5. 数据一致性

当前数据层是 SQLite WAL：

- job claim 使用事务写入 `running + claimed_at + lock_owner`。
- worker 处理期间心跳刷新 `claimed_at`。
- 文件锁防止同一 job 被多进程重复执行。
- stale claim 会在 worker 轮询时回收。
- callback 使用 outbox 表，回调失败不会回滚 job 最终状态。
- `Idempotency-Key` 写入在 SQLite `BEGIN IMMEDIATE` 临界区内按租户串行化，避免并发请求用同一幂等键创建多条不同任务。

当前边界：

- 单机 SQLite 适合本地或单节点私有化部署。
- 多 API/worker 节点共享同一个 SQLite 文件不是推荐生产形态。
- 需要多机或更高并发时，先迁移到 PostgreSQL/Redis 队列，再放大部署。

## 6. 备份与恢复

必须备份：

- `DWM_DB_PATH`
- `DWM_STORAGE_ROOT/inbox`
- `DWM_STORAGE_ROOT/outbox`
- `.runtime/lock.yaml`
- `.runtime/models/manifest.yaml`
- 生产环境变量文件或密钥管理记录

SQLite 在线备份建议使用 `.backup`，避免直接复制正在写入的数据库：

```sh
sqlite3 "$DWM_DB_PATH" ".backup '/backup/cleanplate/app-$(date +%Y%m%d%H%M%S).db'"
rsync -a "$DWM_STORAGE_ROOT/inbox/" /backup/cleanplate/inbox/
rsync -a "$DWM_STORAGE_ROOT/outbox/" /backup/cleanplate/outbox/
```

恢复顺序：

1. 停止 API 和 worker。
2. 恢复 SQLite 备份到 `DWM_DB_PATH`。
3. 恢复 `inbox/` 和 `outbox/`。
4. 启动 API。
5. 运行 `/healthz`。
6. 启动 worker。
7. 运行 `--doctor` 和一条小样本任务。

## 7. 日志与可观测性

当前已具备：

- API 未处理异常会写 `logger.exception`，对外返回统一 `INTERNAL_ERROR`。
- worker 记录启动、任务成功/失败、stale claim 回收和心跳丢失。
- callback worker 记录重试、最终成功/失败事件。
- `jobs` 表保存状态、provider、耗时、错误码和错误消息。
- `callback_events` 表保存回调尝试历史。
- `run_metadata` 表保存 ComfyUI workflow、quality profile、设备和关键参数。
- `--doctor` 提供运行时和 provider 快照。

上线后至少关注：

- queued job 数量和最长排队时间。
- running job 数量和是否超过 `DWM_JOB_CLAIM_TIMEOUT_SECONDS`。
- succeeded/failed 比例。
- `PROVIDER_NOT_AVAILABLE` 和 `PROVIDER_RUN_FAILED` 数量。
- 质量类信号：`INPAINT_NO_EFFECT`（通常是显存不足被节点吞掉，调小 `DWM_CHUNK_FRAMES`）、`needs_review` 比例、`NO_WATERMARK_DETECTED` / `DETECTOR_LOW_CONFIDENCE` 比例（检测器覆盖不到的水印类型，拿样本进评测集）。按 `report_json` 统计即可：

```sql
SELECT json_extract(report_json, '$.detection.scene_type') AS scene,
       status, error_code, json_extract(report_json, '$.needs_review') AS review, count(*)
FROM jobs WHERE created_at >= date('now', '-1 day')
GROUP BY 1, 2, 3, 4 ORDER BY 5 DESC;
```
- callback outbox 中 `failed` 和长期 `pending` 数量。
- storage 剩余空间。
- ComfyUI API 是否可达。

当前没有内置 Prometheus `/metrics`。需要正式指标系统时，先基于 SQLite 查询和日志采集落地外部监控，再评估是否新增原生 metrics 接口。

## 8. 清理与容量

容量分两块算，清理命令只管前面那块：

| 位置 | 内容 | 量级 |
| --- | --- | --- |
| `DWM_STORAGE_ROOT` | inbox / outbox / work，`--cleanup` 按保留期回收 | 随任务量增长 |
| `DWM_RUNTIME_ROOT`（`.runtime`） | ComfyUI 仓库、`.runtime/.venv`、约 14 GB 权重 | 稳定后约 18–20 GB，`--cleanup` 不回收 |

`storage_space` 只检查 inbox 所在磁盘，`.runtime` 所在磁盘要自己盯。

默认文件保留 `DWM_FILE_RETENTION_DAYS=7` 天。先 dry-run：

```sh
uv run cleanplate-worker --cleanup
```

确认候选文件后再执行：

```sh
uv run cleanplate-worker --cleanup --execute-cleanup
```

清理只处理非 `queued/running` 任务的过期输入/输出文件，并跳过仍受保护的路径。

## 9. 回滚

代码回滚：

1. 停止 worker，避免新任务继续写状态。
2. 停止 API。
3. 切回上一版本代码。
4. 恢复上一版本依赖环境。
5. 如果本次变更改过数据结构，先恢复上线前数据库备份。
6. 启动 API，检查 `/healthz`。
7. 启动 worker，检查 `--doctor`。

当前 schema 变更只有加列（0.3.0 加 `jobs.report_json`），旧版本代码忽略新列，可以直接回滚代码。未来如果引入数据迁移，必须提供：

- migration 前备份命令。
- migration 后验证 SQL。
- rollback SQL 或恢复备份步骤。
- 新旧 worker/API 是否兼容的说明。

## 10. 仍需确认的生产边界

当前实现适合单机私有化上线验证，但还不应承诺：

- 多机高并发调度。
- SaaS 级租户隔离和配额。
- Web 管理后台。
- 一键 Docker 生产部署。
- 商业 SLA 监控。
- 所有视频场景的 AI 去水印效果（质量以评测集和 `report.qa` 统计为准）。
