# Hydro Yearbook OCR VLM

[English](README.md)

面向水文年鉴表格图像的 VLM 微调与评测项目。

## 项目概览

本项目聚焦水文年鉴流量表 OCR。当前 v0 范围刻意收窄，只做一个固定版式、一个模型方向、一个输出契约。

v0 任务固定为：
- 单表图像 -> 原样 CSV 文本

当前基线方向为：
- 模型：`GOT-OCR2.0`
- 训练：以高保真合成数据为主
- 最终测试：只在真实已校准流量表上做
- 评测：直接基于模型原始输出，不做后处理

## 当前范围

当前 v0 已固定的方案：
- 只做 `流量`
- 暂不做 `水位`
- 只做一种固定流量表版式
- 合成表格程序绘制，不复用真实截图
- 保留自然日历空位和少量缺测值
- 扰动以几何和光照为主
- 训练优先采用 `ms-swift + LoRA`

## 当前状态

切换到 Linux 后，仓库当前状态如下：
- 原始 PDF 和人工校准 CSV 位于 `datasets/`
- 已将校准 CSV 作为 source-of-truth 标签
- `docs/` 下的主文档已补齐
- v0 合成数据生成脚手架已完成
- 真实流量测试 manifest 构建脚手架已完成
- 面向 `GOT-OCR2.0` 的 `ms-swift` manifest 转换已完成
- 严格 CSV 评测脚本已实现
- Linux 环境安装脚本已提供
- `swift sft` 训练包装脚本已提供
- `data/` 下已有小规模样本与 manifest 预览产物

当前仍未完成：
- 在 `rapid` conda 环境中真正装通并验证 Linux 训练环境
- 将真实 PDF 裁成站点级单表图像
- 启动第一轮 `GOT-OCR2.0` LoRA 微调

## 数据快照

数据说明：
- `datasets/流量/` 是当前 v0 主目标数据
- `datasets/水位/` 已存在，但不在 v0 范围内
- 部分 CSV 使用本地中文编码而非 UTF-8
- 文件名中编码了站点、年份和河流信息，应保持不变

## 仓库结构

```text
.
├── configs/                  # 实验配置
├── data/                     # 已生成的样本与 manifest
├── datasets/                 # 原始 PDF 与已校准 CSV 标签
├── docs/                     # 项目说明与细化规范
├── references/               # 外部参考代码和论文
├── scripts/common/           # 流量表公共工具
├── scripts/data/             # 数据处理与合成数据生成
├── scripts/eval/             # 评测
├── scripts/models/           # 模型相关训练适配
├── tests/                    # 自动化测试
├── AGENTS.md                 # 极简协作入口
├── README.md                 # 英文首页
└── README.zh-CN.md           # 中文首页
```

## 快速开始

本项目当前使用 Linux 下的 `rapid` conda 环境。

1. 阅读 [docs/project-overview.md](docs/project-overview.md)
2. 阅读 [docs/data-spec.md](docs/data-spec.md) 了解 v0 数据契约
3. 阅读 [docs/environment-setup.md](docs/environment-setup.md) 了解环境要求
4. 查看 `datasets/流量/2006/` 下的校准 CSV

代表性命令：

```bash
python3 ./scripts/data/build_real_test_manifest.py
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
python3 ./scripts/eval/evaluate_strict_csv.py --predictions outputs/predictions/got_ocr2_v0.jsonl --output outputs/reports/got_ocr2_v0_eval.json
```

这些脚本已经在仓库中实现，但需要 Python 3.10+ 环境；本机上的训练栈还没有完成端到端验证。

## 路线图

近期执行顺序：
- 在 Linux `rapid` 环境中完成 smoke test
- 完成真实 PDF 到站点级单表图像的提取
- 启动第一轮 `GOT-OCR2.0` LoRA 微调
- 在真实流量测试集上完成严格评测
- 基于代表性失败案例决定下一轮迭代

## 文档

- [项目概览](docs/project-overview.md)
- [数据规范](docs/data-spec.md)
- [合成数据方案](docs/synthetic-data.md)
- [实验计划](docs/experiment-plan.md)
- [环境配置](docs/environment-setup.md)
- [GOT-OCR2.0 微调](docs/got-ocr2-finetune.md)
- [协作指南](AGENTS.md)

## 贡献

详细说明优先放在 `docs/`，并保持 `AGENTS.md` 极简。

## 许可证

许可证信息尚未添加。
