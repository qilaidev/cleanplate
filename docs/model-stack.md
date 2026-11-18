# 模型栈

最近一次复核：**2026-11-18**（上一轮 2026-09-30）。方法见 [model-scouting.md](model-scouting.md)，本轮自动扫描快照在 `eval/scouting/2026-11-18-auto.md`。

## 当前栈

| 环节 | 组件 | 版本 / 权重 | 许可 | 状态 |
| --- | --- | --- | --- | --- |
| 检测（默认） | 静态角标 + 半透明持久边缘检测 | 本仓库 `watermark_detect.py` | MIT | 只看画面四周 22% 的边带，纯 CPU 采样 |
| 检测（兜底） | SAM 3.1 文本提示检测 + 视频跟踪 | ComfyUI 原生节点，`Comfy-Org/sam3.1` `sam3.1_multiplex_fp16.safetensors` | SAM License | 覆盖画面中央与移动水印；上限 `<=180` 抽样帧 / `<=960p` 防爆显存，3 帧流式内存生成时序 mask |
| 主修复 | MiniMax-Remover 1.3B（Wan2.1 底座） | `zibojia/minimax-remover`，节点 `1038lab/ComfyUI-MiniMax-Remover` | **CC-BY-NC-4.0** | 节点 2025-06 后无更新；本仓库通过 81 帧分段、多 ROI 裁剪与无水印分段直通（`bypassed_chunks`）加固 |
| 次修复 | DiffuEraser + ProPainter 先验 | `lixiaowen/diffuEraser`，ProPainter v0.1.0（GitHub Release 3 权重自动下载 + SHA-256 校验） | Apache-2.0 / **S-Lab 1.0（非商用）** | 节点 `9ef5c12f`（2026-10-23）已迁移至 ComfyUI 原生 `ModelPatcher`，移除旧 `diffusers` 依赖冲突并在采样后主动释放显存 |
| 运行时 | ComfyUI | v0.45.0（`Comfy-Org/ComfyUI` `989d93d0`） | GPL-3.0（独立进程，不随本仓库分发） | 原生支持 V3 `io.ComfyNode`、`SAM3_VideoTrack` 与 `--disable-api-nodes` |

## 许可证（商用前必看）

`cleanplate-worker --doctor` 的 `licenses` 段会列出清单里 `commercial_use: false` 的权重。当前：

- **MiniMax-Remover 权重：CC-BY-NC-4.0**。Kijai 转换版（`Kijai/WanVideo_comfy/Wan2_1-MiniMaxRemover_1_3B_fp16.safetensors`）是同一权重，许可不变。
- **ProPainter：S-Lab License 1.0，非商用**。DiffuEraser 本体是 Apache-2.0，但它的工作流依赖 ProPainter 先验。
- PCM LoRA 模型卡未写许可，按"不明"处理。
- ComfyUI 与 `1038lab` 节点是 GPL-3.0，只作为独立进程调用、不打包进本仓库，不影响本仓库 MIT。

结论：**当前默认栈只适合研究/内部评估**。商业工作流要么取得授权，要么换到许可宽松的修复模型（见下方"候选"里的 Wan2.1 VACE 系、D2DF、CLEAR），替换要走 [model-scouting.md](model-scouting.md) 的评测门槛。

## 本轮（2026-11-18）结论

1. **主链不换，运行时升级至 ComfyUI v0.45.0 + 原生核 DiffuEraser**。近 55 天未出现同时满足"效果显著优于现役 + 权重公开可下 + 许可完全可商用 + 可稳定接入单卡 24 GB"的视频去水印/擦除新模型；但上游 `smthemex/ComfyUI_DiffuEraser`（`9ef5c12f`，2026-10-23）已完成 ComfyUI V3 原生 `ModelPatcher` 重构、彻底去掉 `diffusers` 版本锁冲突并在推理后自动 `unload()`，本轮随 ComfyUI `v0.45.0`（2026-11-10）一并升级锁定。
2. **管线工程加固闭环（v0.4.0）**：
   - `SAM 3.1` 长视频全帧跟踪显存与内存封顶（`<=180` 抽帧、`<=960p`、3 帧流式时序膨胀）。
   - 动态/间歇水印的**干净分段直通（Clean-Chunk GPU Bypass）**：无水印窗口直接走 FFmpeg 切片跳过扩散推理，兼顾零像素损耗与吞吐翻倍。
   - 音画与色彩无损闭环：容器感知无损音轨流拷贝（`-c:a copy`）、输入端 `-t` 截断防尾帧丢失、SDR `bt709` VUI/`setparams` 色彩元数据透传防偏色。
   - `ProPainter` 3 个 GitHub Release 先验权重纳入 `--install-runtime --models` 自动流式下载与 SHA-256 强校验。
3. **基准参考**：BeyondMasks（ECCV'26，180 对真实视频）上 DiffuEraser 25.19 dB / MiniMax 23.20 dB（PSNR），MiniMax 的 VLM 去除分略高（3.63 vs 3.49）；叠加 2026-10/11 新发布的 `PROVE`（Apache-2.0）与 `MFVSR`（硬字幕基准），保持"MiniMax 主 + DiffuEraser 次 + 质检兜底"双链仍是当前最稳健组合，最终替换仍以本仓 clean-plate 评测集为准。

## 候选与观察名单

| 模型 / 基准 | 日期 | 许可 / 权重 | 与我们的关系 | 决定 |
| --- | --- | --- | --- | --- |
| **CLEAR**（ICML'26 Oral） | 权重 2026-03-24 | Apache-2.0，`charlesw09/CLEAR-mask-free-video-subtitle-removal`（Wan2.1-Fun-1.3B-Control LoRA） | **无需 mask 的字幕擦除** | **试用**：作为字幕类任务的专用链；约 4.9 s/帧，需独立推理进程 |
| **D2DF**（ECCV'26） | 2026-07-16 | 代码 Apache-2.0，`BigD233333/D2DF`；底座 CogVideoX-5B-I2V（商用条款待核） | 一步去除，RORD 上 PSNR 32.20 vs MiniMax 30.60，约 1 s/片段 | **试用**：候选主修复替代，先核底座许可 |
| **Wan2.1/2.2 VACE、Fun-Inpaint** | ComfyUI 原生 | Apache-2.0 | 许可最干净的通用视频 inpaint；BeyondMasks 上分数偏低（VACE 19.99 dB） | 备选：商用场景的合规修复链 |
| **Hint, Not Hole**（`arXiv:2611.13711`） | 2026-11-10 | 论文已发，权重待开源 | **专为硬字幕/卡拉OK歌词/台标/水印设计的定位鲁棒视频叠加物擦除**（容忍粗略提示框，避免洞内幻觉） | **重点观察**：方向与 Cleanplate 最吻合，权重与许可明确后第一时间实测 |
| **StreamErase**（`arXiv:2610.20137`） | 2026-10-14 | 论文已发，权重待开源 | 因果流式超低延迟视频擦除，解决分钟级扩散延迟 | 观察：发布权重后评估长视频吞吐收益 |
| **LensErase**（`arXiv:2610.18072`） | 2026-10-13 | 论文已发，权重待开源 | 通用 mask 驱动 Erase Anything 基础模型 | 观察 |
| **YOSE**（CVPR'26，`Wucy0519/YOSE-CVPR26`） | 2026-11-04 | `NOASSERTION`（许可未声明） | Token 选择加速 DiT 视频物体擦除 | 观察：待补齐开源许可证与预训练权重 |
| **PROVE**（`xiaomi-research/prove`） | 2026-11-16 | Apache-2.0 | 图像/视频擦除开源评测集与基准代码 | **参考评测**：吸收评测协议补充实机基线 |
| **MFVSR**（`hxyiiii/MFVSR`，`arXiv:2610.18461`） | 2026-10-16 | 许可未声明 | 多样式高分辨率视频硬字幕擦除基准 | 观察：核实数据集许可后再决定是否抽样对照 |
| EffectErase（CVPR'26） | 2026-03 | CC-BY-NC-4.0 | 面向物体副作用（阴影/反射） | 不接：许可 + 目标不符 |
| EffectLearner / SEDiT / BridgeRemoval | 2026-01–08 | 未发布权重 | 物体擦除 / 一步无 mask 字幕擦除 | 观察 |
| Netflix VOID | 2026-04 | Apache-2.0；ComfyUI 原生 | 40 GB+，面向物理交互重场景 | 不接：批量去水印场景单卡成本过高 |
| VOR-Bench（BMVC'26） | 2026-09-15 | 待发布 | VLM 打分，与人评 ρ>0.9 | 发布后接入评测 |

## 检测器

仍然**没有许可干净、开箱可商用的专用公开水印检测框模型**：

- `corzent/yolov12x-dino3-watermark-detection`、`ayan4m1/Watermark-Detection-YOLO26-ONNX`、`qfisch/yolov8n-watermark-detection`：AGPL，不引入。
- `Gebsfrom404/yolo-watermark-detection-collection` 与 2026-10 新出的 `Richard-ZZZZZ/video-watermark-removal`（卡片标 MIT，实则基于 Ultralytics YOLOv8 训练与推理）：存在 **AGPL 传染风险**，按许可门禁拒绝引入。
- `prithivMLmods/Watermark-Detection-SigLIP2`（Apache-2.0，94% 准确率）只做**有/无**二分类，不出位置。可作为便宜的前置判断，减少对"干净视频"跑 SAM 3.1。**试用**。
- 长期路线：用 Apache-2.0 的 RF-DETR 在自造合成水印数据上微调一个定位器（`bastienp/visible-watermark-pita` 数据集未标许可，不直接用）。

## 运行时依赖

| 组件 | 锁定 | 说明 |
| --- | --- | --- |
| ComfyUI | v0.45.0 `989d93d0`（2026-11-10） | 原生支持 V3 节点 Schema、`SAM3_VideoTrack` / `SAM3_TrackToMask`、`--output-directory` 隔离暂存与 `--disable-api-nodes` 禁用外部付费 API 节点 |
| VideoHelperSuite | `4d907bee`（2026-09-02） | 截至 2026-11-18 仍为上游最新主干提交 |
| ComfyUI_DiffuEraser | `9ef5c12f`（2026-10-23） | 已重写为 ComfyUI 原生 `ModelPatcher` 推理核，移除旧版 `diffusers` 强锁冲突，采样结束自动卸载显存 |
| ComfyUI-MiniMax-Remover | `5957d627`（2025-06-27） | 上游停止更新；替代方向为 `kijai/ComfyUI-WanVideoWrapper` 的 `WanVideoMiniMaxRemoverEmbeds`，迁移需实机对齐效果后切换 |
| huggingface_hub | `>=0.24.0,<1` | 保持 `<1` 稳定下载契约；ProPainter 3 项 GitHub Release 权重由 `runtime_installer.py` 直接 HTTP 流式下载并做 SHA-256 校验 |

## 明确不做

- 不为了"新"换主链；换模型只看评测集数字、许可和成本。
- 不引入 AGPL 检测模型。
- 不接只能在商业云上跑的私有节点。
- 不在没有真权重、真 GPU 的机器上承诺效果数字。`eval_metrics` 的阈值（区域 PSNR ≥ 28 pass、≥ 22 degraded）是起点，实机评测后再调。
