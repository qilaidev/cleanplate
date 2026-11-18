# 模型追踪：怎么持续找到最新、最好用的去水印模型

目标：任何时候都能在一两个小时内回答"现在最好用、能用的去水印相关开源模型是哪个，比我们的好多少"。做法是固定的四步：**扫描 → 筛选 → 评测 → 决策**，每月一次（15 天窗口快扫，30 天窗口全扫），结论写回 [model-stack.md](model-stack.md)。

## 1. 扫描（自动）

```sh
uv run cleanplate-scout --days 30 --output eval/scouting/$(date +%F)-auto.md
uv run cleanplate-scout --days 15            # 快扫，直接打到终端
GITHUB_TOKEN=... uv run cleanplate-scout     # 带 token 避免 GitHub 搜索限流
```

它查三个源，只保留窗口内更新/发布的条目，按相关度和热度排序，并给出许可分类：

| 源 | 查询 | 排序 |
| --- | --- | --- |
| Hugging Face 模型 | `watermark`、`remover`、`object removal`、`video inpainting`、`video eraser`、`subtitle removal`、`text removal`、`logo detection` | `lastModified` 降序 |
| arXiv | `abs:"video object removal"`、`abs:"video inpainting"`、`abs:"visible watermark"`、`subtitle removal`、`text removal + video`、`ti:eraser` | `submittedDate` 降序 |
| GitHub | `video inpainting`、`video object removal`、`watermark removal`、`comfyui remover`、`subtitle removal video` + `pushed:>=窗口起点` | stars 降序 |

"隐形水印 / 指纹 / 溯源 / 音频"类条目名词相同但问题不同，自动降权。

脚本覆盖不到、需要人工补看的地方（每次都看）：

- Hugging Face 首页 Trending 与 `https://huggingface.co/models?search=remover&sort=trending`（新模型常先上热榜再出论文）。
- `https://huggingface.co/spaces?search=watermark&sort=trending`（有 Space 的模型通常能直接试）。
- ComfyUI 官方博客 / `docs.comfy.org` 的新模板（原生支持意味着接入成本最低），以及 `kijai/ComfyUI-WanVideoWrapper` 的提交记录。
- 观察名单（model-stack.md）里每一项的仓库：权重发布了没有、许可变了没有。

## 2. 筛选（人工，15 分钟）

按顺序过这几道门，任一不过就停：

1. **任务对口**：视频（不是单图）、可见叠加物（台标/角标/字幕/平台水印），不是隐形水印攻防。单图模型只在"逐帧 + 时序后处理"有明确方案时才考虑。
2. **权重可下载**：只有论文、只有"coming soon"的进观察名单。
3. **许可**：MIT / Apache-2.0 / BSD / OpenRAIL 可直接考虑；`*-nc-*`、"research only" 只能内部评估；AGPL/GPL 模型代码不进本仓库（可作为独立进程）；许可不明 = 不可用。底座模型的许可也要看（例如 CogVideoX、Wan 各自的条款）。
4. **能接进来**：优先 ComfyUI 原生节点 > 活跃维护的自定义节点 > 需要独立推理进程。独立进程可以接，但要写一个新 provider。
5. **成本**：单卡 24 GB 内能跑；给出或能估出每秒视频的处理时长。

## 3. 评测（量化，必须）

候选只在我们自己的评测集上和现役模型对比，论文数字只作参考（论文基准多是物体删除，不是水印）。

```sh
uv run python -m wm_platform.eval_fixtures              # 生成/刷新带 clean plate 真值的合成夹具
uv run python -m wm_platform.eval_runner --split all --providers auto,minimax_remover,<candidate> --report
```

每个用例输出（只在水印区域内计算，避免全局指标稀释局部错误）：

| 指标 | 含义 | 当前门槛 |
| --- | --- | --- |
| `psnr_region` | 修复区域与 clean plate 的保真度 | pass ≥ 28 dB，degraded ≥ 22 dB |
| `ssim_region` | 修复区域结构相似度 | ≥ 0.80 |
| `flicker_region` | 修复区域时序闪烁（相对 clean plate） | ≤ 6.0 |
| `psnr_outside` | 水印外区域不能被改坏 | ≥ 35 dB 或 输入自身 −3 dB |
| `residual_detected` | 输出上还能检测到水印 | 必须为 false |
| 处理时长 / 显存峰值 | 手工记录 | 不超过现役 1.5 倍，除非质量显著提升 |

合成夹具之外，维护一个**真实授权样本集**（各平台水印、字幕、移动水印、半透明水印、长视频各若干），没有 clean plate 的用例用 `eval_judge` 人工打 pass/degraded/fail，和合成集一起看。

替换门槛：候选在 `pass` 率上不低于现役，且在至少一类场景（例如字幕、移动水印、半透明）上明显更好（区域 PSNR +1 dB 以上或 pass 率 +10 个百分点），许可满足上线环境要求。

## 4. 决策与落地

- 结论写进 [model-stack.md](model-stack.md)：候选表加一行（日期、许可、权重、决定和一句理由）。
- **试用**：新增 provider 或 workflow，默认不进 `auto` 链，用 `provider=<name>` 显式调用，跑一两周真实流量的 `report.qa` 统计。
- **采用**：调整 `pipeline.auto_inpaint_chain`、更新 `.runtime/lock.yaml` 和 `models/manifest.yaml`（带 `license` / `commercial_use`），CHANGELOG 记录。
- **淘汰**：从链和清单移除；git 历史就是归档，不在仓库里留旧文件。

## 本轮记录

- 2026-11-18：自动扫描 `eval/scouting/2026-11-18-auto.md`（覆盖近 55 天 Hugging Face / arXiv / GitHub）；升级锁定 ComfyUI `v0.45.0` 与原生核 `ComfyUI_DiffuEraser` `9ef5c12f`，新增 `Hint, Not Hole`、`StreamErase`、`PROVE`、`MFVSR` 等候选研判，详见 [model-stack.md](model-stack.md)。
- 2026-09-30：历史扫描归档于 git 提交记录；结论见 CHANGELOG `0.3.0`。
