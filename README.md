**简体中文 (README.md)** · [English (README.en.md)](README.en.md) · [llms.txt](llms.txt) · [codemeta.json](codemeta.json) · [API 合同](docs/api.md) · [架构说明](docs/architecture.md) · [模型栈](docs/model-stack.md) · [贡献指南](CONTRIBUTING.md) · [安全策略](SECURITY.md) · [MIT 协议](LICENSE)

# Cleanplate — 自托管异步视频水印检测、ROI 修复与自动质检管线

**`Cleanplate`（v0.3.0）是一款基于 FastAPI、ComfyUI 与 FFmpeg 构建的开源（MIT）自托管异步视频去水印处理节点，面向自动化媒体工作流：无需人工框选水印位置，服务自动完成静态/动态水印检测、多 ROI 裁剪规划、81 帧分段扩散修复回退链、音画无损拼接与出片质量门禁，并通过轮询或 HMAC-SHA256 回调交付成片与结构化 `report`。**

[![CI](https://github.com/qilaidev/cleanplate/actions/workflows/ci.yml/badge.svg)](https://github.com/qilaidev/cleanplate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20HTTP-009688.svg)](docs/api.md)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.38.1-FF6F00.svg)](docs/model-stack.md)
[![Architecture](https://img.shields.io/badge/Worker-Async%20Queue-6E44FF.svg)](docs/architecture.md)

名字取自影视后期的 *clean plate*（去掉台标、字幕后的干净底片）。单机单 GPU 起步，支持多 Worker 横向扩展，服务端代码采用 MIT 协议。

> **合规与许可提示**：只处理你拥有权利或已获明确授权的内容。本仓库服务端代码为 **MIT** 许可，但默认主修复模型 **MiniMax-Remover** 权重为 **CC-BY-NC-4.0（禁止商用）**，**ProPainter** 先验权重为 **S-Lab License 1.0（非商用）**。商用部署前请务必阅读 [模型栈 · 许可](docs/model-stack.md#许可证商用前必看)。

---

## 核心事实卡（Entity Snapshot）

| 属性 | 规格与事实 |
| --- | --- |
| **项目实体** | `Cleanplate`（`qilaidev/cleanplate` · v0.3.0） |
| **软件许可** | 服务端代码 **MIT License**；默认模型权重含非商用条款（`CC-BY-NC-4.0` / `S-Lab 1.0`），可按指南替换 |
| **核心技术栈** | Python `>=3.11,<3.13` · FastAPI · Uvicorn · SQLite (WAL) · OpenCV · FFmpeg/ffprobe · ComfyUI `v0.38.1` · `uv` |
| **CLI 入口** | `cleanplate-api`（HTTP API）· `cleanplate-worker`（异步 Worker / 安装 / 医生自检 / 清理）· `cleanplate-scout`（新模型追踪） |
| **六阶段管线** | 1. 摄入规范化 → 2. 单次检测（OpenCV + SAM 3.1）→ 3. 多 ROI 与 81 帧分段 → 4. 修复回退链 → 5. 音画拼接 → 6. 自动质检与 HMAC 回调 |
| **质检与兜底策略** | 自动核验 `mask_mean_abs_diff` 与 `residual_overlap`；拒绝对低置信度画面盲修（`DETECTOR_LOW_CONFIDENCE`），无效修复自动切换下一模型（`INPAINT_NO_EFFECT`） |
| **硬件与验证边界** | 单测（`pytest` 139 条）与 `local_fallback` 冒烟全程可在无 GPU 环境运行；真实 AI 去水印需 CUDA GPU（推荐 16–24 GB+ VRAM）与约 18–20 GB `.runtime/` 空间 |
| **所属工程矩阵** | [qilai · qilaidav.com](https://qilaidav.com) · [05 · 企业需求管理与 AI 底座（Workspace）](https://qilaidav.com/workspace/) |

---

## 核心定位：它是什么 / 不是什么

| 维度 | 它是什么（What It Is） | 它不是什么（What It Is Not） |
| --- | --- | --- |
| **服务形态** | 工作流中的一个异步处理节点（HTTP 提交 → 轮询 / HMAC 回调取结果与报告） | **不是**面向终端用户的在线去水印网站或桌面 GUI 编辑器 |
| **检测机制** | 全自动 CPU 静态角标/持久边缘检测 + ComfyUI 原生 SAM 3.1 兜底，每任务只检测一次 | **不是**需要人工在界面框选矩形或手绘遮罩的工具（正式接口不接受手动 bbox） |
| **修复与质检** | 多 ROI 独立裁剪 + 81 帧（`4k+1`）分段修复回退链 + 出片质量门禁与 `report` | **不是**绑定单一模型或出错后默默返回带水印原片的 demo 脚本 |
| **模型生态** | 可替换模型栈，内置 `cleanplate-scout`、`--doctor` 许可门禁与 clean-plate 评测集 | **不是**默认直接用于无授权商业 SaaS 运营的黑盒权重包 |
| **扩展架构** | 单机 / 单 GPU 起步，通过多 `cleanplate-worker` 对接各自 ComfyUI 横向扩展 | **不是**多机集群调度器、计费系统或多租户 SaaS 平台 |

---

## 适用场景与同类方案对比

### 典型适用场景

- **自动化媒体加工流水线**：作为上游素材入库或分发流水线中的一个异步 Worker 节点，通过 `POST /v1/jobs`（支持 `Idempotency-Key`）提交视频并接收结构化 `report` 与 HMAC 签名回调。
- **批量角标/台标/字幕擦除研发管线**：针对混杂固定台标与移动字幕的素材，先用低成本 CPU OpenCV 检测静态水印，仅在不确定时唤醒 GPU `SAM 3.1`，配合多 ROI 串联避免整帧重绘失真。
- **模型评测与可替换底座**：利用仓库内置的 `eval/` clean-plate 真值夹具与 `cleanplate-scout` 定期跟踪 Hugging Face / arXiv / GitHub 最新修复与字幕擦除模型，在替换商用友好权重前完成指标与许可门禁校验。

### 架构对比（基于代码与架构事实，无虚构对比数据）

| 对比维度 | `Cleanplate` | 在线去水印 SaaS 网站 | 手动遮罩剪辑软件（AE / PR） | 裸 ComfyUI GUI 工作流 | FFmpeg `delogo` 滤镜 |
|---|---|---|---|---|---|
| **自动化程度** | HTTP 异步队列，全自动检测 + 修复 + 质检 | 需网页手动上传并逐个框选 | 需人工逐帧描遮罩与手动导出 | 需手工连线、人工看显存与手动切片 | 需硬编码固定矩形坐标 |
| **移动/半透明水印** | CPU 持久边缘投票 + `SAM 3.1` 文本提示跟踪 | 视网站黑盒能力而定 | 人工关键帧跟踪 | 需手工拼装并调参 | 仅支持固定矩形坐标模糊 |
| **多水印分离修复** | 自动多 ROI 独立裁剪串联（`expand_roi_chain`） | 通常整块或单框处理 | 手动建多个遮罩层 | 单外接框易退化为整帧重绘 | 需叠加多个硬编码滤镜 |
| **防静默失败质检** | 内置 `quality_gate` 检测 `INPAINT_NO_EFFECT` 并回退 | 无结构化机器可读质检报告 | 人工肉眼复检 | 节点异常可能原样吐出输入帧 | 无质量门禁 |
| **音频与封装一致性** | 摄入 VFR/旋转规范化 + 无损回填原音轨 | 可能重压缩或丢失音轨特性 | 人工导出设置 | 易出现音画不同步或帧数补齐问题 | 纯滤镜处理 |

---

## 六阶段处理管线

```text
POST /v1/jobs (multipart 上传或 DWM_STORAGE_ROOT 下的路径)
  └─ worker 取任务
      1. 摄入规范化   VFR / 旋转 / 奇数尺寸 / 非 yuv420p → 恒定帧率中间文件
      2. 检测（一次）  OpenCV 静态角标 + 半透明持久边缘检测；不确定时回退 ComfyUI 原生 SAM 3.1
                     └ 没有水印 → NO_WATERMARK_DETECTED 或原样返回（可配）
                     └ 看不准   → DETECTOR_LOW_CONFIDENCE，拒绝盲修
      3. 分段规划     每个水印独立 ROI 裁剪窗口；按 81 帧（4k+1）分段
      4. 修复回退链   MiniMax-Remover（主）→ DiffuEraser（次），ComfyUI 执行
      5. 音画拼接     裁掉分段重叠帧，无损回填原音轨，统一编码输出
      6. 自动质检     计算 mask_mean_abs_diff 与 residual_overlap；无效修复判 INPAINT_NO_EFFECT
                     触发下一模型，疑似残留标 needs_review；写库并发送 HMAC 签名回调
```

每个任务的 `report` 字段记录输入信息、检测结果（类型、置信度、区域）、分段、每个模型的尝试和质检结论，下游据此决定直接用、转人工还是丢弃。详见 [`docs/api.md`](docs/api.md)。

---

## 快速开始

### 1. 无 GPU Mock 快速验证（单测与本地链路冒烟）

前置：Python 3.11 / 3.12（`>=3.11,<3.13`）、[uv](https://docs.astral.sh/uv/)、`PATH` 里有 `ffmpeg` / `ffprobe`。

```sh
uv sync --extra dev --locked
uv run ruff check src tests apps                # 代码风格与静态规则检查
uv run pytest -q                                # 139 个测试全部在 mock ComfyUI 下跑，无需 GPU

uv run cleanplate-api --host 127.0.0.1 --port 8000
uv run cleanplate-worker                        # 另一个终端常驻消费队列（单轮调试可加 --once）
```

不装模型也能验证完整 HTTP → 队列 → Worker → FFmpeg → 报告链路：显式使用 `provider=local_fallback`（仅规范化并复制视频，不执行 AI 去水印）。

```sh
ffmpeg -v quiet -f lavfi -i testsrc=duration=2:size=320x240:rate=10 -pix_fmt yuv420p /tmp/smoke.mp4 -y
curl -s -X POST http://127.0.0.1:8000/v1/jobs -H "X-API-Key: dev-secret-key" \
  -F media_type=video -F provider=local_fallback -F file=@/tmp/smoke.mp4
curl -s http://127.0.0.1:8000/v1/jobs/<job_id> -H "X-API-Key: dev-secret-key"
```

### 2. GPU 生产环境部署（ComfyUI + 模型权重）

```sh
uv run cleanplate-worker --runtime-plan               # 预览本地 Runtime 与模型安装计划
uv run cleanplate-worker --install-runtime --models   # 按 .runtime/lock.yaml 和 manifest 安装（仅下权重用 --models-only）
uv run cleanplate-worker --start-comfyui              # 启动 ComfyUI 并等待就绪（单独探活可用 --comfyui-health）
uv run cleanplate-worker --doctor                     # 检查 providers.runnable 与各权重商业许可状态
```

> **本地代理提示**：若终端设置了 `HTTP_PROXY` / `HTTPS_PROXY`，请务必设置 `NO_PROXY=127.0.0.1,localhost`，否则 `--doctor` 或 `--comfyui-health` 访问 `127.0.0.1:8188` 时可能出现 `502 Bad Gateway`（已在 [`.env.example`](.env.example) 中注明）。

`GET /v1/providers` 里 `minimax_remover.runnable=true` 后，用默认的 `provider=auto` 提交即可。定期清理过期任务文件可执行 `uv run cleanplate-worker --cleanup`（预览）与 `--cleanup --execute-cleanup`（实际删除）。部署、备份、扩容见 [`docs/operations.md`](docs/operations.md)。

---

## 配置

全部通过环境变量配置，示例见 [`.env.example`](.env.example)。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DWM_ENV` | `development` | `production` 时拒绝默认 API key |
| `DWM_DEFAULT_API_KEY` | `dev-secret-key` | 请求头 `X-API-Key` |
| `DWM_API_HOST` / `DWM_API_PORT` | `127.0.0.1` / `8000` | `cleanplate-api` 默认监听地址与端口 |
| `DWM_STORAGE_ROOT` | `storage` | inbox / outbox / work / app.db |
| `DWM_COMFYUI_API_URL` | `http://127.0.0.1:8188` | 每个 worker 可以指向不同 GPU 上的 ComfyUI |
| `DWM_DETECTOR_MODE` | `auto` | `static_overlay` / `sam3_text` / `auto`（静态不确定时再跑 SAM 3.1） |
| `DWM_WATERMARK_DETECT_PROMPT` | `watermark, logo, on-screen text overlay, corner badge, platform stamp` | SAM 3.1 文本提示，支持 `watermark:3` 限实例数 |
| `DWM_NO_WATERMARK_POLICY` | `fail` | `passthrough`：检测器都确认没水印时原样返回 |
| `DWM_CHUNK_FRAMES` | `81` | 每段帧数（自动取 4k+1）；`0` = 整段一次（只适合短片） |
| `DWM_QA_ENABLED` | `true` | 出片质检 |
| `DWM_QA_RESIDUAL_POLICY` | `flag` | 水印残留时：`flag` 返回并标 `needs_review`；`fail` 判失败 |
| `DWM_QUALITY_MODE` | `balanced` | `fast` / `balanced` / `quality` / `corner_hq`，提交时可按任务覆盖 |
| `DWM_MINIMAX_ROI_ENABLED` | `true` | 只修水印周围的裁剪窗口 |
| `DWM_MAX_VIDEO_SECONDS` | `180` | 超过直接拒绝，`0` 关闭 |
| `DWM_COMFYUI_PROMPT_TIMEOUT_SECONDS` | `1800` | 单段等待上限（含 ComfyUI 排队） |
| `DWM_AUTO_ALLOW_COPY_FALLBACK` | `false` | 为 `true` 时 `auto` 可能以"复制原片"成功，生产别开 |
| `DWM_FILE_RETENTION_DAYS` | `7` | `--cleanup --execute-cleanup` 的保留期 |
| `DWM_KEEP_WORK_FILES` | `false` | 保留中间文件（mask、分段）排查问题 |

其余（限流、回调、ROI 尺寸、delogo 坐标等）见 [`src/wm_platform/config.py`](src/wm_platform/config.py)。

---

## 模型与许可

| 角色 | 当前 | 许可 |
| --- | --- | --- |
| 检测 | 静态/半透明角标检测（OpenCV，本仓库）+ SAM 3.1（ComfyUI 原生节点） | MIT / SAM License |
| 主修复 | MiniMax-Remover 1.3B | **CC-BY-NC-4.0** |
| 次修复 | DiffuEraser + ProPainter | Apache-2.0 / **S-Lab（非商用）** |

`--install-runtime --models` 按清单下载约 14 GB（`MiniMax-Remover` 2.8 GB、`DiffuEraser` 8.8 GB、`sam3.1` 1.7 GB、`sd-vae-ft-mse` 0.3 GB、`clip_l` 0.25 GB）；加上仓库和 `.runtime/.venv`，`.runtime` 稳定后约 18–20 GB，`--cleanup` 不回收它。另外 `propainter` 在清单里标着 `required: true`，但它的 source 是 GitHub release 页，安装器只认 Hugging Face，需要手动把 5 个权重文件（共约 0.4 GB）放到 `.runtime/ComfyUI/models/DiffuEraser/propainter/`，否则 `comfy_diffueraser` 一直 `installed=false`。

选型依据、最新调研与商用替代候选见 [`docs/model-stack.md`](docs/model-stack.md)；怎么持续追踪新模型见 [`docs/model-scouting.md`](docs/model-scouting.md)（`uv run cleanplate-scout --days 30`）。

---

## 常见问题（FAQ）

### 1. 运行 Cleanplate 需要什么样的 GPU 和显存？
真实 AI 去水印需要配备 NVIDIA CUDA GPU 的机器运行 ComfyUI（按 81 帧扩散分段运行 `MiniMax-Remover` 1.3B 或 `DiffuEraser` 建议 16–24 GB+ 显存）。显存紧张时，保持默认 `DWM_MINIMAX_ROI_ENABLED=true` 仅对水印 ROI 裁剪窗口修复，或适当调小 `DWM_CHUNK_FRAMES`（自动对齐 `4k+1`）；需注意 Apple Silicon 仅支持 MPS 推理，而非本架构面向的 CUDA 生产基线。

### 2. 没有 GPU 或 ComfyUI 可以测试与开发吗？
可以。`uv run pytest -q` 下的全部 139 个测试均基于 Mock ComfyUI 与真实 FFmpeg 运行，完全不需要 GPU。本地启动 `cleanplate-api` 与 `cleanplate-worker` 时，也可在提交任务时显式指定 `provider=local_fallback` 来端到端验证 HTTP 鉴权、FFmpeg 规范化、SQLite 状态流转与回调签名。

### 3. 服务端代码是 MIT 协议，为什么不能直接用默认模型做商业运营？
Cleanplate 的**服务端代码**采用 MIT 许可，可自由集成与自托管；但默认的**模型权重**（`MiniMax-Remover` 为 CC-BY-NC-4.0，`ProPainter` 先验为 S-Lab License 1.0）均限制非商业用途。商用场景请先取得权重商业授权，或按 [`docs/model-stack.md`](docs/model-stack.md) 替换为 Apache-2.0 等商用友好模型（如 `Wan2.1 VACE`、`CLEAR`）。

### 4. 静态角标水印和移动/动态水印分别怎么处理？
在默认 `DWM_DETECTOR_MODE=auto` 下，管线先用纯 CPU 的 OpenCV 检测器（`static_overlay`）扫描画面四周 22% 边带的静态或半透明持久水印；若置信度不足或未发现静态水印，则触发一次 ComfyUI 原生 `SAM 3.1`（`sam3_text`）文本提示检测并生成逐帧 mask，覆盖画面中央与移动水印。

### 5. 为什么有时任务会直接报 `DETECTOR_LOW_CONFIDENCE` 或 `INPAINT_NO_EFFECT`？
这是 Cleanplate 防止“盲修”和“静默返回带水印原片”的两道保护机制。当检测出的 mask 面积过大、过于零碎（`complex`）或静态不确定且 `SAM 3.1` 不可用时，Worker 返回 `DETECTOR_LOW_CONFIDENCE` 拒绝盲修；若修复节点异常返回了未修改的原帧（`mask_mean_abs_diff < 1.5`），`quality_gate` 会判为 `INPAINT_NO_EFFECT` 并自动切换回退模型。

### 6. 如何在回调与上传接口中保证生产安全？
设置 `DWM_ENV=production` 后服务会拒绝使用默认开发密钥 `dev-secret-key`，并将所有 `X-API-Key` 以 SHA-256 哈希保存并按租户执行速率限制（默认 60 次/分钟）。同时 `POST /v1/jobs` 的 `input_path` 严格限制在 `DWM_STORAGE_ROOT` 内，`callback_url` 默认拒绝 `localhost` 与私网 IP，回调请求附带 HMAC-SHA256 签名。

### 7. 如何多机/多卡横向扩展 Worker，以及定期清理磁盘空间？
保持一个共享存储与 SQLite 数据库（或升级共享队列）的 `cleanplate-api` 实例，启动多个 `cleanplate-worker` 进程并为每个 Worker 配置独立的 `DWM_COMFYUI_API_URL`（如 `http://127.0.0.1:8188`、`http://127.0.0.1:8189`）指向不同 GPU 的 ComfyUI。过期任务文件可通过 `uv run cleanplate-worker --cleanup` 预览并加 `--execute-cleanup` 清理（注意 `.runtime/` 本身约 18–20 GB，不会被 `--cleanup` 回收）。

---

## 仓库结构

```text
cleanplate/
├── src/wm_platform/   # 核心服务代码（API、worker、检测、分段编排、质检、ComfyUI 客户端、评测、模型追踪）
├── apps/              # api / worker 进程启动入口
├── workflows/         # ComfyUI API 格式 workflow 模板
├── .runtime/          # Runtime 锁定配置（lock.yaml）与模型权重清单（models/manifest.yaml）
├── eval/              # 带 clean plate 真值的评测夹具与模型追踪扫描快照
├── tests/             # pytest 测试套件（139 个测试，mock ComfyUI，无需真 GPU）
├── docs/              # 架构、API、运维、模型栈、模型追踪、路线图
├── llms.txt           # 面向 LLM/AI 检索的规范摘要（llmstxt.org）
├── codemeta.json      # CodeMeta 2.0 结构化软件元数据
├── SECURITY.md        # 安全漏洞披露与防护说明
├── CONTRIBUTING.md    # 开发规范与提交流程
├── .env.example       # 环境变量配置模板
└── pyproject.toml     # 包依赖与 CLI 脚本入口（cleanplate-api / cleanplate-worker / cleanplate-scout）
```

---

<details>
<summary><strong>检索意图与高频词索引（Search Intent & Query Mapping）</strong></summary>

| 用户检索意图 | 典型检索词（中文 / English） | 仓库对应事实与入口 |
| --- | --- | --- |
| 自托管开源 AI 视频去水印服务 | 开源视频去水印 API · 自托管去水印管线 · self-hosted video watermark removal API | [`src/wm_platform/api_app.py`](src/wm_platform/api_app.py) · [`docs/api.md`](docs/api.md) |
| 基于 ComfyUI + FastAPI 的异步批量去水印节点 | ComfyUI 去水印流水线 · MiniMax-Remover DiffuEraser API · ComfyUI video inpainting queue | [`src/wm_platform/provider_runtime.py`](src/wm_platform/provider_runtime.py) · [`workflows/`](workflows/) |
| 自动检测静态/移动水印并防盲修与质检兜底 | 自动检测水印位置 · ROI 视频修复质检 · automatic watermark detection ROI inpainting QA | [`src/wm_platform/watermark_detect.py`](src/wm_platform/watermark_detect.py) · [`src/wm_platform/quality_gate.py`](src/wm_platform/quality_gate.py) |
| 核查视频去水印模型商用许可与替换评估 | MiniMax-Remover 商用许可 · 视频去水印开源评测 · video inpainting license gate scout | [`docs/model-stack.md`](docs/model-stack.md) · [`src/wm_platform/model_scout.py`](src/wm_platform/model_scout.py) |

</details>

<sub>**关键词**：开源视频去水印 · 自托管去水印 API · FastAPI 异步队列 · ComfyUI 视频修复 · MiniMax-Remover · DiffuEraser · SAM 3.1 水印检测 · ROI 分段修复 · self-hosted video watermark removal · async video inpainting pipeline · automated watermark QA · FFmpeg audio remux</sub>

---

## 文档

- [`docs/architecture.md`](docs/architecture.md)：处理管线、模块、数据表、扩展方式
- [`docs/api.md`](docs/api.md)：HTTP 接口、`report` 结构、错误码、回调
- [`docs/operations.md`](docs/operations.md)：部署、GPU 首跑、健康检查、扩容、备份、回滚
- [`docs/model-stack.md`](docs/model-stack.md)：当前模型、许可、候选和取舍
- [`docs/model-scouting.md`](docs/model-scouting.md)：找最新模型的固定流程和评测门槛
- [`docs/roadmap.md`](docs/roadmap.md)：接下来做什么
- [`CHANGELOG.md`](CHANGELOG.md)：版本变更记录

---

## 工程架构与生态 · qilai (`qilaidav.com`)

本项目由 **[qilai · qilaidav](https://qilaidav.com)** 独立工程工作室设计与开源维护，聚焦 **确定性系统工程与高信噪比技术诊断（Deterministic Systems Engineering & High-Signal Technical Diagnostics）**。

- **所属工程支柱**：[05 · 企业需求管理与知识/技能/能力底座（Workspace）](https://qilaidav.com/workspace/) · [English Pillar](https://qilaidav.com/en/workspace/)
- **工作室主站**：[https://qilaidav.com](https://qilaidav.com) · [English](https://qilaidav.com/en/)
- **工程技术笔记（Field Reports）**：[https://qilaidav.com/notes/](https://qilaidav.com/notes/)
- **开源项目矩阵**：[https://github.com/qilaidev](https://github.com/qilaidev)
- **工程联络**：[Telegram @BD8888](https://t.me/BD8888) · [公开频道 @AIKB8](https://t.me/AIKB8) · `hello@qilaidav.com`
