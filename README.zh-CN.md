# Hydro Yearbook OCR VLM

[English](README.md)

面向水文年鉴表格图像的 VLM 微调与评测项目。

## 项目概览

本项目聚焦水文年鉴流量表 OCR。当前 v1 的训练与评测范围刻意收窄，只做一个固定流量表版式、一个模型方向、一个输出契约。

当前仓库同时维护两层任务表示：
- 标签权威层：人工校准 CSV 仍然是 source-of-truth
- 当前 GOT 基线目标：单表图像 -> 官方 GOT 格式化表格文本

当前基线方向为：
- 模型：`GOT-OCR2.0`
- 训练：以高保真合成数据为主
- 最终测试：只在真实已校准流量表上做
- 训练目标：官方 `OCR with format:` 输出
- 评测：用共享表格解析器评测 raw/pretty GOT 表格输出，并以 CSV source-of-truth 为目标

## 当前范围

当前 v1 已固定的方案：
- 只做 `流量`
- `水位` 不进入当前训练和最终评测范围
- 只做一种固定流量表版式
- 合成表格程序绘制，不复用真实截图
- 保留自然日历空位和少量缺测值
- manifest 目标会删除空白伪影行和末尾空白伪影列，但不改动原始 CSV 文件
- 扰动以几何和光照为主
- 训练优先采用 `ms-swift + LoRA`

## 当前状态

截至 2026-04-24，仓库当前状态如下：
- 原始 PDF 和人工校准 CSV 位于 `datasets/`
- 已将校准 CSV 作为 source-of-truth 标签
- `docs/` 下的主文档已补齐
- v1 合成数据生成已实现，采用接近真实裁切图的横版比例、斜线日/月表头和严格 shuffle split
- 真实流量测试 manifest 构建已实现并通过 smoke test
- 真实 `2006 流量` 表的 OCR 日值裁切已实现
- 已实现从真实日值裁切图到校准 CSV 的打分对齐
- 真实 `2014 水位` 表的 OCR 日值裁切也已实现
- 真实数据裁切主路径当前采用“左下统计区 ROI + `平均` 优先、`平/均` 受限弱锚点”的规则
- 面向 `GOT-OCR2.0` 的 `ms-swift` manifest 转换已实现并通过 smoke test
- 已建立共享核心包 `src/yearbook_ocr/`，承载稳定的数据、OCR、推理与评测逻辑
- GOT 推理入口已统一到 `scripts/models/got_ocr2/run_inference.py`
- 面向 raw GOT/LaTeX 输出的表格评测脚本已实现，并有测试覆盖
- Linux 环境安装脚本已提供
- `swift sft` 训练包装脚本已提供，并已完成 smoke run 验证
- `data/` 下已有小规模样本与 manifest 预览产物
- `got` 环境中的数据、manifest、推理和评测相关 focused suites 当前通过
- 一次基于 `sdpa` 的 GOT-OCR2.0 LoRA smoke training 已经跑通
- 当前 v1 Swift manifest 规模为 `8000` 条训练记录和 `2000` 条验证记录
- 已对齐的真实流量测试 manifest 含 `35` 条记录

当前仍未完成：
- 基于 v1 synthetic manifest 重新训练 `GOT-OCR2.0` LoRA
- 在真实流量测试集上重新跑 base 和微调模型推理
- 产出一份正式的真实流量测试对比报告，比较 base GOT 与 v1 微调 GOT

## 当前稳定路径

当前推荐的稳定路径是：
- conda 环境：`got`
- 训练方法：`ms-swift + LoRA`
- attention backend：`sdpa`
- 缓存目录：`outputs/cache/`
- 训练入口：`scripts/models/got_ocr2/run_swift_sft.sh`

已验证出的环境结论：
- 当前机器不应依赖 `flash-attn`
- 这一条训练路径不应使用 `bitsandbytes==0.41.0`
- 较旧的 `peft==0.4.0` 与当前 `ms-swift` 组合不兼容

## 数据快照

数据说明：
- `datasets/流量/` 是当前主目标数据
- `datasets/水位/` 已存在，并已完成真实表格抽取与日值裁切，但暂不进入当前模型基准
- `datasets/derived/` 下的 `station_tables_daily/` 是当前 `2006 流量` 正式真实测试图像集
- 同样的 `station_tables_daily/` 产物也已生成到 `2014 水位`
- 部分 CSV 使用本地中文编码而非 UTF-8
- 文件名中编码了站点、年份和河流信息，应保持不变

## 仓库结构

```text
.
├── configs/                  # 实验配置
├── data/                     # 已生成的样本与 manifest
├── datasets/                 # 原始 PDF、已校准 CSV 标签，以及真实数据薄入口
├── docs/                     # 项目说明与细化规范
├── references/               # 外部参考代码和论文
├── src/yearbook_ocr/         # 共享核心包：数据、OCR、GOT 推理、评测
├── scripts/data/             # 合成数据入口
├── scripts/eval/             # 评测薄包装
├── scripts/models/           # 模型相关薄包装
├── tests/                    # 自动化测试
├── AGENTS.md                 # 极简协作入口
├── README.md                 # 英文首页
└── README.zh-CN.md           # 中文首页
```

## 快速开始

本项目当前使用 Linux 下的 `got` conda 环境。

1. 阅读 [docs/project-overview.md](docs/project-overview.md)
2. 阅读 [docs/data-spec.md](docs/data-spec.md) 了解当前数据契约
3. 阅读 [docs/environment-setup.md](docs/environment-setup.md) 了解环境要求
4. 查看 `datasets/流量/2006/` 下的校准 CSV

代表性命令：

```bash
conda run -n rapid python ./datasets/crop_flow_table_daily_region.py
conda run -n rapid python ./datasets/run_ocr_on_image.py 'datasets/derived/debug_rois/2014_水位/page_0004_table1_汉江汉川站_2014.jpg' --ocr-cuda off
python3 ./datasets/build_real_flow_alignment.py
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000 --val-ratio 0.2 --seed 20260408 --num-workers 16
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
python3 ./scripts/models/got_ocr2/run_inference.py --manifest data/manifests/flow_real_test_aligned.jsonl --limit 5 --backend official_chat --query-mode official_format
python3 ./scripts/eval/evaluate_predictions.py --predictions outputs/got_ocr2_base/eval/checkpoint-0/flow_real_first5_official_chat.jsonl --output outputs/reports/got_ocr2_base_first5_eval.json
```

这些脚本已经在仓库中实现，并且本机已在 `got` 环境中完成数据准备、测试和 1-step GOT-OCR2.0 LoRA smoke run 验证。

## 路线图

近期执行顺序：
- 基于 v1 synthetic Swift manifest 训练 `GOT-OCR2.0` LoRA
- 在已对齐真实流量测试集上重新跑 base 和微调模型推理
- 对 raw/pretty GOT 输出运行表格评测和模型对比
- 基于代表性失败案例决定下一轮迭代

## 文档

- [项目概览](docs/project-overview.md)
- [数据规范](docs/data-spec.md)
- [合成数据方案](docs/synthetic-data.md)
- [实验计划](docs/experiment-plan.md)
- [环境配置](docs/environment-setup.md)
- [真实测试抽取](docs/real-test-extraction.md)
- [GOT-OCR2.0 微调](docs/got-ocr2-finetune.md)
- [协作指南](AGENTS.md)

## 贡献

详细说明优先放在 `docs/`，并保持 `AGENTS.md` 极简。

## 许可证

许可证信息尚未添加。
