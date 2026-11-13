# 模型栈

最近一次复核：**2026-09-30**（上一轮 2026-09-11）。方法见 [model-scouting.md](model-scouting.md)，本轮自动扫描快照在 `eval/scouting/2026-09-30-auto.md`。

## 当前栈

| 环节 | 组件 | 版本 / 权重 | 许可 | 状态 |
| --- | --- | --- | --- | --- |
| 检测（默认） | 静态角标 + 半透明持久边缘检测 | 本仓库 `watermark_detect.py` | MIT | 只看画面四周 22% 的边带 |
| 检测（兜底） | SAM 3.1 文本提示检测 + 视频跟踪 | ComfyUI 原生节点，`Comfy-Org/sam3.1` `sam3.1_multiplex_fp16.safetensors` | SAM License | 覆盖画面中央、移动水印；需要 GPU |
| 主修复 | MiniMax-Remover 1.3B（Wan2.1 底座） | `zibojia/minimax-remover`，节点 `1038lab/ComfyUI-MiniMax-Remover` | **CC-BY-NC-4.0** | 节点 2025-06 后无更新 |
| 次修复 | DiffuEraser + ProPainter 先验 | `lixiaowen/diffuEraser`，ProPainter v0.1.0 | Apache-2.0 / **S-Lab 1.0（非商用）** | 节点 2026-03 后无更新 |
| 运行时 | ComfyUI | v0.38.1（`Comfy-Org/ComfyUI`） | GPL-3.0（独立进程，不随本仓库分发） | 原生 SAM3 节点自 v0.35 无改动 |

## 许可证（商用前必看）

`cleanplate-worker --doctor` 的 `licenses` 段会列出清单里 `commercial_use: false` 的权重。当前：

- **MiniMax-Remover 权重：CC-BY-NC-4.0**。Kijai 转换版（`Kijai/WanVideo_comfy/Wan2_1-MiniMaxRemover_1_3B_fp16.safetensors`）是同一权重，许可不变。
- **ProPainter：S-Lab License 1.0，非商用**。DiffuEraser 本体是 Apache-2.0，但它的工作流依赖 ProPainter 先验。
- PCM LoRA 模型卡未写许可，按"不明"处理。
- ComfyUI 与 `1038lab` 节点是 GPL-3.0，只作为独立进程调用、不打包进本仓库，不影响本仓库 MIT。

结论：**当前默认栈只适合研究/内部评估**。商业工作流要么取得授权，要么换到许可宽松的修复模型（见下方"候选"里的 Wan2.1 VACE 系、D2DF、CLEAR），替换要走 [model-scouting.md](model-scouting.md) 的评测门槛。

## 本轮（2026-09-30）结论

1. **主链不换**。没有同时满足"效果明显更好 + 权重可用 + 许可可商用 + 能进 ComfyUI"的新模型。
2. **工程侧的缺陷比模型本身更要紧**，本轮已修（见 CHANGELOG 0.3.0）：
   - MiniMax 节点把整段视频一次送进扩散模型，长视频必 OOM；且不是 4k+1 帧时会用冻结帧补齐。→ 按 81 帧分段。
   - 同一节点**捕获所有异常后原样返回输入帧**，任务"成功"但水印还在。→ 出片质检（`INPAINT_NO_EFFECT`）。
   - 分离的多个水印共用一个外接框，常退化为整帧修复。→ 多 ROI 串联。
3. **新基准数据**（BeyondMasks，ECCV'26，180 对真实视频）：DiffuEraser 25.19 dB / MiniMax 23.20 dB（PSNR），MiniMax 的 VLM 去除分略高（3.63 vs 3.49）。两者互有胜负，保持"主 + 次"链并用质检兜底是合理的。但这些基准都是**物体删除**，不是水印/字幕；我们自己的 clean-plate 评测集仍是最终依据。

## 候选与观察名单

| 模型 | 日期 | 许可 / 权重 | 与我们的关系 | 决定 |
| --- | --- | --- | --- | --- |
| **CLEAR**（ICML'26 Oral） | 权重 2026-03-24 | Apache-2.0，`charlesw09/CLEAR-mask-free-video-subtitle-removal`（Wan2.1-Fun-1.3B-Control LoRA） | **无需 mask 的字幕擦除**；上一轮误记为"未开源" | **试用**：作为字幕类任务的专用链；约 4.9 s/帧，只给字幕场景用；无 ComfyUI 节点，需要独立推理进程 |
| **D2DF**（ECCV'26） | 2026-07-16 | 代码 Apache-2.0，`BigD233333/D2DF`；底座 CogVideoX-5B-I2V（CogVideoX License，商用条款待核） | 一步去除，RORD 上 PSNR 32.20 vs MiniMax 30.60，VFID 反而更差；约 1 s/片段 | **试用**：候选主修复替代，先核底座许可 |
| Wan2.1/2.2 VACE、Fun-Inpaint | ComfyUI 原生 | Apache-2.0 | 许可最干净的通用视频 inpaint；BeyondMasks 上分数偏低（VACE 19.99 dB） | 备选：商用场景的兜底修复链 |
| EffectErase（CVPR'26） | 2026-03 | CC-BY-NC-4.0 | 面向物体副作用（阴影/反射） | 不接：许可 + 目标不符 |
| EffectLearner | 2026-08-06 | 未发布权重 | ROSE-Bench 领先 | 观察 |
| SEDiT（一步无 mask 字幕擦除） | 2026-05-14 | 未发布 | 1080p 65 帧 4 s | 观察：发布后与 CLEAR 对比 |
| BridgeRemoval（ICML'26） | 2026-01 | 仍无权重 | — | 观察 |
| GenEraser / DualEraser | 2026-05 | 仓库 Apache-2.0，无权重 | — | 观察 |
| Object-WIPER | 2026-01 | 官方仅指标；有非官方 Apache-2.0 移植 | HunyuanVideo 底座，重 | 不接 |
| Netflix VOID | 2026-04 | Apache-2.0；ComfyUI 已原生（2026-05） | 40 GB+，面向物理交互 | 不接：批量场景成本过高 |
| VOR-Bench（BMVC'26） | 2026-09-15 | 待发布 | VLM 打分，与人评 ρ>0.9 | 发布后接入评测 |

## 检测器

仍然**没有许可宽松、能定位（出框）的公开水印检测模型**：

- `corzent/yolov12x-dino3-watermark-detection`、`ayan4m1/Watermark-Detection-YOLO26-ONNX`、`qfisch/yolov8n-watermark-detection`：AGPL，不引入。
- `Gebsfrom404/yolo-watermark-detection-collection` 标 MIT，但基于 Ultralytics 且部分转载，许可可疑。
- `prithivMLmods/Watermark-Detection-SigLIP2`（Apache-2.0，94% 准确率）只做**有/无**二分类，不出位置。可作为便宜的前置判断，减少对"干净视频"跑 SAM 3.1。**试用**。
- 长期路线：用 Apache-2.0 的 RF-DETR 在自造合成水印数据上微调一个定位器（`bastienp/visible-watermark-pita` 数据集未标许可，不直接用）。

## 运行时依赖

| 组件 | 锁定 | 说明 |
| --- | --- | --- |
| ComfyUI | v0.38.1 `20ca544e` | 仓库迁到 Comfy-Org；v0.35→v0.38 去掉了 torchaudio 依赖、改了 assets 系统；SAM3 相关 open issue #16675（选 CLIP 设备时报错）、#15811（`:N` 后缀混进 prompt） |
| VideoHelperSuite | `4d907bee`（2026-09-01） | 已是最新 |
| ComfyUI_DiffuEraser | `c8461876`（2026-03-27） | 已是最新，低活跃 |
| ComfyUI-MiniMax-Remover | `5957d627`（2025-06-27） | 无人维护；替代是 `kijai/ComfyUI-WanVideoWrapper` 的 `WanVideoMiniMaxRemoverEmbeds`（支持 context window），迁移需要重写 workflow 并实机验证 |
| huggingface_hub | `<1` | 已出 2.0（2026-09-24，httpx2、默认系统证书库）；升级前需去掉 `resume_download` 等旧参数 |

## 明确不做

- 不为了"新"换主链；换模型只看评测集数字、许可和成本。
- 不引入 AGPL 检测模型。
- 不接只能在商业云上跑的私有节点。
- 不在没有真权重、真 GPU 的机器上承诺效果数字。`eval_metrics` 的阈值（区域 PSNR ≥ 28 pass、≥ 22 degraded）是起点，实机评测后再调。
