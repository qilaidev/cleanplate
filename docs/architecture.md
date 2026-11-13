# 架构

## 进程

```text
上游系统 ──HTTP──> cleanplate-api ──SQLite(WAL)──> cleanplate-worker ×N ──HTTP──> ComfyUI ×M (GPU)
                        │                              │
                        └──── storage/{inbox,outbox,work} ────┘
                                                       └─ callback 线程 ──HTTP──> 上游回调地址
```

- **API**（FastAPI）：鉴权、限流、幂等、落盘、建任务、查询、下载。不做推理。
- **worker**：轮询抢任务（事务 + 文件锁 + 心跳），执行处理管线，写结果，入队回调。每个 worker 同时只处理一个任务。
- **callback 线程**：在 worker 进程内独立轮询 `callback_outbox`，HMAC 签名投递，失败按指数退避重试。
- **ComfyUI**：独立进程，承载所有 GPU 推理（SAM 3.1、MiniMax-Remover、DiffuEraser）。通过 `/prompt`、`/history`、`/view` 调用。

扩展方式：一台 GPU 一个 ComfyUI，每个 ComfyUI 配 1–2 个 worker（第二个 worker 能让 CPU 上的规范化/检测/拼接和 GPU 推理重叠），worker 用 `DWM_COMFYUI_API_URL` 指向各自的 ComfyUI。SQLite 适合单机；多机共享队列时先换 PostgreSQL/Redis。

## 处理管线（`provider_runtime.ProviderRuntime.run_with_fallback`）

| 步骤 | 代码 | 做什么 | 失败时 |
| --- | --- | --- | --- |
| 0. 探测 | `probe()` | 过滤链上不可运行的 provider | 全不可用 → `PROVIDER_NOT_AVAILABLE` |
| 1. 规范化 | `media.normalize_input` | ffprobe；VFR/旋转/奇数尺寸/非 yuv420p 时转成 CFR 中间文件（x264 CRF 12，音轨原样） | `MEDIA_PROCESSING_FAILED` |
| 2. 检测 | `watermark_detect.detect_watermark` | 只跑一次。`auto`：静态检测有把握就用；否则 SAM 3.1。输出 FFV1 无损 mask、外接框、每个区域的框、并集 mask | 两个检测器都跑且都为空 → `NO_WATERMARK_DETECTED`（或 passthrough）；其他 → `DETECTOR_LOW_CONFIDENCE` |
| 3. 规划 | `media.plan_chunks`、`plan_minimax_rois` | 4k+1 帧分段（尾段回滑重叠）；每个水印一个 16 px 对齐的 ROI，相交则合并，总面积过大则整帧 | — |
| 4. 修复 | provider `run()` → `run_comfy_chunks` | 每段一个 ComfyUI prompt（`comfy_graph.set_frame_window` 改两个加载节点的帧窗口；多 ROI 用 `expand_roi_chain` 串联） | 记录本次尝试，换下一个 provider |
| 5. 拼接 | `media.assemble_output` | 裁掉重叠帧、拼接、回填原音轨，x264 CRF 16 | 同上 |
| 6. 质检 | `quality_gate.check_output` | mask 内与输入的平均差 < 1.5 → `unchanged`；静态水印在输出上仍被检测到（与原 mask 重叠 ≥ 35%）→ `residual_suspected` | `unchanged` 换下一个；全部 unchanged → `INPAINT_NO_EFFECT`。只剩 residual → 按 `DWM_QA_RESIDUAL_POLICY` 标 `needs_review` 或 `QA_RESIDUAL_WATERMARK` |
| 7. 收尾 | `_finish` | 结果移到 `outbox/`，写 `report`，删 `work/<job_id>/` | — |

`provider=auto` 的链是 `minimax_remover → comfy_diffueraser`，只有 `DWM_AUTO_ALLOW_COPY_FALLBACK=true` 才追加 `local_fallback`。指定单个 provider 时链只有它自己。

### 为什么这些步骤是必须的

- **规范化**：OpenCV 检测器、VideoHelperSuite 和修复模型都按帧序号取帧。可变帧率或带旋转元数据的手机视频，在不同解码器里帧数、朝向不一致，mask 会错位。
- **分段**：MiniMax-Remover 节点把整段视频一次送进扩散模型，1080p 几十秒就会爆显存；Wan 系 VAE 时间维压缩 4 倍，非 4k+1 帧会被补冻结帧。
- **质检**：MiniMax 节点捕获所有异常（包括显存不足）后原样返回输入帧，ComfyUI 显示成功。不检查就会把带水印的视频当成功结果交出去。
- **多 ROI**：常见的"左上台标 + 右下用户名"如果用一个外接框，窗口几乎覆盖整帧，失去 ROI 的意义。

### 已知限制

- 分段边界是硬切：相邻两段在修复区域内可能有轻微跳变。需要时加大 `DWM_CHUNK_FRAMES`（显存允许）或后续做重叠羽化。
- 残留检测只对静态水印有效；移动水印只做 `unchanged` 检查。
- 静态检测器只看四周边带；画面中央的水印依赖 SAM 3.1。
- 水印只在视频一部分时间出现时，mask 按"全程存在"处理（静态检测）或逐帧跟踪（SAM 3.1）。

## 模块

| 模块 | 职责 |
| --- | --- |
| `api_app.py`、`api_validation.py`、`rate_limit.py` | HTTP 接口、参数校验、限流 |
| `repository.py`、`db.py`、`models.py` | SQLite 表、状态流转、Pydantic 模型 |
| `worker_service.py`、`job_locks.py` | 抢任务、心跳、文件锁、回调投递 |
| `provider_runtime.py` | provider 注册、探测、编排管线、ROI 规划 |
| `media.py` | ffprobe/ffmpeg：探测、规范化、分段规划、拼接 |
| `watermark_detect.py` | 静态/半透明检测、SAM 3.1 检测、mask 输出 |
| `quality_gate.py` | 出片质检 |
| `comfy_jobs.py`、`comfy_graph.py`、`comfy_runtime.py` | ComfyUI 调用、prompt 变换、启动与探活 |
| `runtime_contract.py`、`runtime_installer.py`、`doctor.py` | runtime 锁与模型清单、安装、自检（含许可证） |
| `eval_*.py` | 评测夹具、指标、执行、人工判定 |
| `model_scout.py` | 模型追踪扫描（`cleanplate-scout`） |

## 数据

| 表 | 内容 |
| --- | --- |
| `jobs` | 状态、provider、输入/输出路径、错误码、质量档、`report_json` |
| `run_metadata` | 每次推理的 workflow、参数、设备、种子、检测类型/置信度/mask 占比 |
| `callback_outbox` / `callback_events` | 回调队列和投递历史 |
| `api_keys` | 租户与 key 哈希 |

schema 变更只做加列（`db._ensure_job_columns`），旧库启动时自动补齐。

## 工程约束

1. API 进程不做推理；worker 崩溃不影响 API。
2. 用户不提供水印位置；`manual bbox/mask` 不进正式接口。
3. 检测不确定就失败，不编造 mask；修复没效果就失败，不把原片当结果。
4. provider 失败返回统一错误码，并在 `report.attempts` 里留下每一步。
5. 回调默认拒绝 localhost/私网；`DWM_ENV=production` 禁止默认 key。
6. 每个模型权重在清单里声明许可；`--doctor` 报告不可商用项。
