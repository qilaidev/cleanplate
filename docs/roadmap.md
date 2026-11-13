# 路线图

已完成的阶段（API/worker/队列/回调、ComfyUI runtime、检测 + 修复主链、评测夹具、批量场景加固）见 [CHANGELOG](../CHANGELOG.md)。这里只列接下来的事，按优先级排序。

## 1. 实机基线（下一步，阻塞后续所有效果决策）

- 在 24 GB+ GPU 环境中执行 `--install-runtime --models`，跑 `eval_runner --split all --report`，保存基线报告。
- 用真实授权样本（各平台角标、字幕、移动水印、半透明水印、长视频）补评测集，逐类统计 pass 率。
- 实测并标定：`DWM_CHUNK_FRAMES`（显存/速度）、质检阈值（`UNCHANGED_MEAN_ABS_DIFF`、`RESIDUAL_OVERLAP`）、`eval_metrics` 的 pass/degraded 阈值、各 quality profile 参数。

## 2. 许可可商用的修复链（如果服务用于商业）

- 以 Wan2.1/2.2 VACE（Apache-2.0，ComfyUI 原生）做一条 provider，和 MiniMax/DiffuEraser 在评测集上对比。
- 试 D2DF（先核 CogVideoX 底座许可）。
- 达到替换门槛后调整 `auto` 链，清单里去掉非商用权重。

## 3. 按水印类型路由

- 字幕类：接 CLEAR（Apache-2.0，无 mask）作为独立 provider，检测到底部大面积文字时路由过去。
- 前置"有无水印"分类器（`prithivMLmods/Watermark-Detection-SigLIP2`，Apache-2.0），干净视频不跑 SAM 3.1。
- 长期：用 RF-DETR（Apache-2.0）在合成数据上训一个水印定位器，替代阈值型静态检测。

## 4. 运行时

- MiniMax 节点无人维护：评估迁移到 `kijai/ComfyUI-WanVideoWrapper` 的 MiniMax 支持（带 context window，可能替代我们自己的分段）。
- 分段边界重叠羽化（当前是硬切）。
- `huggingface_hub` 升到 1.x/2.x（去掉已移除的参数）。

## 5. 规模化（有需求时）

- Prometheus `/metrics`（队列长度、各错误码、`needs_review` 比例、每秒视频处理时长）。
- 多机：队列迁 PostgreSQL / Redis，存储迁对象存储。
