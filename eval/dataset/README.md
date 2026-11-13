# 评测集说明

## 目录结构

```
dataset/
├── simple/     # 固定四角、面积 < 5%、位置稳定
├── medium/     # 半透明 / 轻微移动 / 面积 5–15%
├── hard/       # 大面积、多水印、与主体重叠
├── clean/      # 每个夹具的无水印 clean plate（无损 FFV1 .mkv）和真值区域 .region.png
├── results/    # runner 输出的 JSON
└── manifest.csv
```

仓库不收录商业平台未授权成片。默认夹具是 160×120、8 帧的自制 OpenCV 视频：动态纹理背景（滚动渐变 + 条纹 + 移动圆盘）上叠加水印，有水印版本按 mp4v 有损编码（贴近真实输入），clean plate 无损保存。只用来验收管道，不代表真实去水印效果。

## 生成夹具

```bash
uv run python -m wm_platform.eval_fixtures
```

会写入 `simple/` / `medium/` / `hard/` / `clean/` 并更新 `manifest.csv`（含 `clean_filename`、`region_filename` 列）。可用 `--dataset-root` 指到别的目录。

| split | 夹具 | 意图 |
| --- | --- | --- |
| simple | 固定左上/右下不透明色块 | 静态检测的时序方差路径应能抓住 |
| medium | 半透明色块（α=0.45） | 静态检测的持久边缘路径应能抓住 |
| medium | 缓慢位移色块 | 静态检测必须 low；SAM 3.1 跟踪应给出逐帧 mask（`scene_type=moving_overlay`） |
| hard | 大面积纹理带 / 多块 / 中心+边缘重叠 | 门禁应拒绝或判 complex |

## 场景分类

### simple（简单场景）

- 固定角标水印（位于画面四角）
- 小面积水印（< 5% 画面面积）
- 位置稳定的水印（多帧间位置变化小）

### medium（中等场景）

- 移动水印（位置有变化但不剧烈）
- 中等面积水印（5%-15% 画面面积）
- 半透明水印

### hard（复杂场景）

- 大面积水印（> 15% 画面面积）
- 动态水印（位置变化剧烈）
- 多个水印同时存在
- 水印与主体内容重叠

## 使用方法

```bash
uv run python -m wm_platform.eval_runner --split simple --provider auto
uv run python -m wm_platform.eval_runner --split all --provider local_fallback
uv run python -m wm_platform.eval_runner --split simple --providers minimax_remover,comfy_diffueraser --report
```

可选参数：

- `--dataset-root` 默认 `eval/dataset`
- `--split`：`simple` | `medium` | `hard` | `all`
- `--provider` 或 `--providers`：逗号分隔，默认 `auto,minimax_remover,comfy_diffueraser`
- `--report`：写出按 provider 汇总的 JSON/Markdown（succeeded/failed/error_code、mean duration/mask_ratio、judgment）
- `--from-results <json>`：只对已有结果出报告，不重跑
- `--judgments <csv>`：把 `id,judgment[,provider]` 合并进 pending 行

`local_fallback` 只用来验证 runner。报告里它单独记为 `degraded` / `local_fallback`，**不算** AI 去水印成功。空结果或 `PROVIDER_NOT_AVAILABLE` 也不算成功。

## 自动指标（有 clean plate 的 case）

`manifest.csv` 里带 `clean_filename` 的 case，任务成功后 runner 会用 `wm_platform.eval_metrics` 对比输出和 clean plate，只在水印区域（`region_filename` 的真值区域；没有时用输入与 clean 的差异推出来）算：

| 字段 | 含义 |
| --- | --- |
| `psnr_region` / `ssim_region` | 修复区域相对 clean plate 的保真度 |
| `psnr_outside` | 水印区域之外必须原样保留（ROI 贴回的保护指标）；`psnr_outside_input` 是输入自身的基线 |
| `flicker_region` | 修复区域内帧间变化与 clean plate 的差，衡量修复器"发明"的闪烁 |
| `residual_detected` | 在输出上重跑静态检测，仍能检出水印 = 没去掉 |

AI provider 的行会据此自动写 `judgment`（`judgment_source=auto`）：区域 PSNR ≥ 28 且区域外不劣于基线 3 dB、SSIM ≥ 0.80、闪烁 ≤ 6 为 `pass`；区域 PSNR ≥ 22 为 `degraded`；残留检出或更低为 `fail`。`local_fallback` 的行只记指标，`judgment` 保持 `pending`。人工 `--judgments` CSV 会覆盖自动判定。阈值是在没有真模型的机器上定的起点，实机评测后再调。

某个 split 没有视频时，runner 以退出码 0 写出空 `results` 列表，不会失败。

## 判定

```bash
uv run python -m wm_platform.eval_judge --result eval/dataset/results/<date>-<provider>.json --id simple-tl --judgment pass
```

`judgment` 只能是 `pass` | `fail` | `degraded`。也可传 `--csv judgments.csv`。改完后再 `--from-results ... --report`，报告里的 judgment 计数会变。

## 记录结果

每次运行写入：

`eval/dataset/results/<date>-<provider>.json`

`--report` 另写：

`eval/dataset/results/<date>-report.json`  
`eval/dataset/results/<date>-report.md`

每条记录包含 `job_id`、`provider_requested`、`provider_selected`、`status`、`error_code`、`duration_ms`、`scene_type`、`confidence_level`、`mask_ratio`、`workflow_name`、`output_path`、`notes`、`degraded`、`ai_success`、`metrics`、`judgment_source`。`judgment` 默认为 `pending`，有 clean plate 且 AI 成功时自动填写。
