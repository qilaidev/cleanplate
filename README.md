# Cleanplate — 自托管异步视频水印检测、ROI 修复与自动质检管线

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20HTTP-009688.svg)](docs/api.md)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.38.1-FF6F00.svg)](docs/model-stack.md)
[![Architecture](https://img.shields.io/badge/Worker-Async%20Queue-6E44FF.svg)](docs/architecture.md)

**中文 (README.md)** · [English (README.en.md)](README.en.md) · [llms.txt](llms.txt) · [API 合同](docs/api.md) · [贡献指南](CONTRIBUTING.md) · [安全策略](SECURITY.md) · [开源协议](LICENSE)

**Cleanplate** 是基于 **FastAPI**、**ComfyUI** 与 **FFmpeg** 构建的自托管异步视频去水印处理节点，面向自动化媒体工作流：无需人工框选水印位置，服务自动检测静态或移动水印、规划 ROI 裁剪与 81 帧分段、执行多模型修复回退链、无痕回填音轨并完成出片质检，最终通过轮询或 HMAC 签名回调交付成片与结构化报告。

名字取自影视后期的 *clean plate*（去掉台标、字幕后的干净底片）。服务端代码采用 MIT 许可，单机单 GPU 起步，支持多 Worker 横向扩展。

> **合规与许可提示**：只处理你拥有权利或已获明确授权的内容。本仓库服务端代码为 **MIT** 许可，但默认主修复模型 **MiniMax-Remover** 权重为 **CC-BY-NC-4.0（禁止商用）**，**ProPainter** 先验权重为 **S-Lab License 1.0（非商用）**。商用部署前请务必阅读 [模型栈 · 许可](docs/model-stack.md#许可证商用前必看)。

---

## 定位

| 是 | 不是 |
| --- | --- |
| 工作流中的一个异步处理节点（HTTP 提交 → 回调 / 轮询取结果） | 面向终端用户的在线去水印网站 |
| 自动检测 + ROI 修复 + 自动质检，每个任务带结构化 JSON `report` | 需要人工标注水印位置的工具 |
| 可替换的模型栈，具备标准化的「找新模型 → 许可门禁 → 评测 → 替换」流程 | 绑死某个模型的 demo |
| 单机 / 单 GPU 起步，按 Worker 横向扩展 GPU | 多机调度、计费、SaaS 多租户平台 |

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

每个任务的 `report` 字段记录输入信息、检测结果（类型、置信度、区域）、分段、每个模型的尝试和质检结论，下游据此决定直接用、转人工还是丢弃。详见 [docs/api.md](docs/api.md)。

---

## 快速开始

### 1. 无 GPU Mock 快速验证（单测与本地链路冒烟）

前置：Python 3.11 / 3.12（`>=3.11,<3.13`）、[uv](https://docs.astral.sh/uv/)、`PATH` 里有 `ffmpeg` / `ffprobe`。

```sh
uv sync --extra dev
uv run pytest                                   # 139 个测试全部在 mock ComfyUI 下跑，无需 GPU

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

`GET /v1/providers` 里 `minimax_remover.runnable=true` 后，用默认的 `provider=auto` 提交即可。定期清理过期任务文件可执行 `uv run cleanplate-worker --cleanup`（预览）与 `--cleanup --execute-cleanup`（实际删除）。部署、备份、扩容见 [docs/operations.md](docs/operations.md)。

---

## 配置

全部通过环境变量配置，示例见 [.env.example](.env.example)。

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

其余（限流、回调、ROI 尺寸、delogo 坐标等）见 `src/wm_platform/config.py`。

---

## 模型与许可

| 角色 | 当前 | 许可 |
| --- | --- | --- |
| 检测 | 静态/半透明角标检测（OpenCV，本仓库）+ SAM 3.1（ComfyUI 原生节点） | MIT / SAM License |
| 主修复 | MiniMax-Remover 1.3B | **CC-BY-NC-4.0** |
| 次修复 | DiffuEraser + ProPainter | Apache-2.0 / **S-Lab（非商用）** |

`--install-runtime --models` 按清单下载约 14 GB（`MiniMax-Remover` 2.8 GB、`DiffuEraser` 8.8 GB、`sam3.1` 1.7 GB、`sd-vae-ft-mse` 0.3 GB、`clip_l` 0.25 GB）；加上仓库和 `.runtime/.venv`，`.runtime` 稳定后约 18–20 GB，`--cleanup` 不回收它。另外 `propainter` 在清单里标着 `required: true`，但它的 source 是 GitHub release 页，安装器只认 Hugging Face，需要手动把 5 个权重文件（共约 0.4 GB）放到 `.runtime/ComfyUI/models/DiffuEraser/propainter/`，否则 `comfy_diffueraser` 一直 `installed=false`。

选型依据、最新调研与商用替代候选见 [docs/model-stack.md](docs/model-stack.md)；怎么持续追踪新模型见 [docs/model-scouting.md](docs/model-scouting.md)（`uv run cleanplate-scout --days 30`）。

---

## 常见问题 (FAQ)

### 1. 运行 Cleanplate 需要什么样的 GPU 和显存？
真实 AI 去水印需要配备 NVIDIA GPU 的机器运行 ComfyUI（按 81 帧扩散分段运行 `MiniMax-Remover` 1.3B 或 `DiffuEraser` 建议 16 GB+ 显存）。显存紧张时，保持默认 `DWM_MINIMAX_ROI_ENABLED=true` 仅对水印 ROI 裁剪窗口修复，或调小 `DWM_CHUNK_FRAMES`。

### 2. 没有 GPU 或 ComfyUI 可以测试与开发吗？
可以。`uv run pytest` 下的全部 139 个测试均基于 Mock ComfyUI 运行，完全不需要 GPU。本地启动 `cleanplate-api` 与 `cleanplate-worker` 时，也可以在提交任务时指定 `provider=local_fallback` 来端到端验证 HTTP 鉴权、FFmpeg 规范化、SQLite 状态流转与回调签名。

### 3. 代码是 MIT 协议，为什么不能直接用默认模型做商业运营？
Cleanplate **服务端代码**采用 MIT 许可，可自由集成与自托管；但默认的**模型权重**（`MiniMax-Remover` 为 CC-BY-NC-4.0，`ProPainter` 先验为 S-Lab License 1.0）均限制非商业用途。商用场景请先取得权重商业授权，或按 [docs/model-stack.md](docs/model-stack.md) 替换为 Apache-2.0 等商用友好模型（如 `Wan2.1 VACE`、`CLEAR`）。

### 4. 静态角标水印和移动/动态水印分别怎么处理？
在默认 `DWM_DETECTOR_MODE=auto` 下，管线先用纯 CPU 的 OpenCV 检测器（`static_overlay`）扫描画面四周边带的静态/半透明角标；若置信度不足或未发现静态水印，则触发一次 ComfyUI 原生 `SAM 3.1`（`sam3_text`）文本提示检测并生成逐帧 mask，覆盖画面中央与移动水印。

### 5. 为什么有时任务会直接报 `DETECTOR_LOW_CONFIDENCE` 拒绝修复？
为了避免"盲修"破坏正常画面。当检测出的 mask 面积过大或过于零碎（`complex`），或者静态检测不确定且 `SAM 3.1` 不可用时，Worker 会返回 `DETECTOR_LOW_CONFIDENCE` 终止修复，由上游系统决定转人工复核或保留原片。

### 6. 如何多机/多卡横向扩展 Worker？
保持一个 `cleanplate-api` 实例（共享存储与数据库），在同一台或多台机器上启动多个 `cleanplate-worker` 进程，为每个 Worker 分配不同的 `DWM_COMFYUI_API_URL`（如 `http://127.0.0.1:8188`、`http://127.0.0.1:8189`）指向各自绑定的 GPU ComfyUI 实例即可。Worker 之间通过文件锁与数据库事务安全抢占任务。

---

## 仓库结构

```text
cleanplate/
├── src/wm_platform/   # 核心服务代码（API、worker、检测、分段编排、质检、ComfyUI 客户端、评测、模型追踪）
├── apps/              # api / worker 进程启动入口
├── workflows/         # ComfyUI API 格式 workflow 模板
├── .runtime/          # Runtime 锁定配置（lock.yaml）与模型权重清单（models/manifest.yaml）
├── eval/              # 带 clean plate 真值的评测夹具与模型追踪扫描快照
├── tests/             # pytest 测试套件（mock ComfyUI，无需真 GPU）
├── docs/              # 架构、API、运维、模型栈、模型追踪、路线图
├── llms.txt           # 面向 LLM/AI 检索的规范摘要（llmstxt.org）
├── SECURITY.md        # 安全漏洞披露与防护说明
├── CONTRIBUTING.md    # 开发规范与提交流程
├── .env.example       # 环境变量配置模板
└── pyproject.toml     # 包依赖与 CLI 脚本入口（cleanplate-api / cleanplate-worker / cleanplate-scout）
```

---

## 文档

- [docs/architecture.md](docs/architecture.md)：处理管线、模块、数据表、扩展方式
- [docs/api.md](docs/api.md)：HTTP 接口、report 结构、错误码、回调
- [docs/operations.md](docs/operations.md)：部署、GPU 首跑、健康检查、扩容、备份、回滚
- [docs/model-stack.md](docs/model-stack.md)：当前模型、许可、候选和取舍
- [docs/model-scouting.md](docs/model-scouting.md)：找最新模型的固定流程和评测门槛
- [docs/roadmap.md](docs/roadmap.md)：接下来做什么
- [CHANGELOG.md](CHANGELOG.md)：版本变更记录
