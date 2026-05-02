# 常用运行命令

本文记录项目中需要手动运行的常用命令。默认在仓库根目录执行：

```bash
cd /home/yubin/py_prj/yearbook_VLM
```

约定：
- 模型训练、合成数据、推理、评测使用 `got` 环境。
- 真实 PDF 抽取、OCR 裁切使用 `rapid` 环境。
- 原始校准 CSV 不直接修改；manifest target 会去掉空白伪影行和末尾空白伪影列。
- 本仓库不要求 agent 执行 `git push`。需要推送时手动运行 `git push origin <branch>`。

## 快速检查

```bash
git status --short
```

```bash
conda run --no-capture-output -n got python -V
conda run --no-capture-output -n got python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run --no-capture-output -n got python -c "import swift; print('swift ok')"
```

```bash
conda run --no-capture-output -n rapid python -c "import fitz; print('fitz ok')"
conda run --no-capture-output -n rapid python -c "from rapidocr import RapidOCR; print('rapidocr ok')"
```

## 测试

当前 focused suite：

```bash
conda run --no-capture-output -n got pytest -q \
  tests/test_flow_common.py \
  tests/test_backfill_got_format_targets.py \
  tests/test_real_flow_test_prep.py \
  tests/test_evaluate_predictions.py \
  tests/test_output_classification.py \
  tests/test_generate_synthetic_flow_v0.py \
  tests/test_got_inference.py \
  tests/test_model_comparison.py
```

如需跑全部测试：

```bash
conda run --no-capture-output -n got pytest -q
```

## 真实测试集准备

抽取 PDF 中的站点表格：

```bash
conda run --no-capture-output -n rapid python datasets/extract_station_tables.py
```

只 smoke test 每个 PDF 的第一页：

```bash
conda run --no-capture-output -n rapid python datasets/extract_station_tables.py --limit-pages 1
```

基于已抽取表格生成 daily-only crop：

```bash
conda run --no-capture-output -n rapid python datasets/crop_flow_table_daily_region.py
```

调试单张 OCR ROI：

```bash
conda run --no-capture-output -n rapid python datasets/run_ocr_on_image.py \
  'datasets/derived/debug_rois/2014_水位/page_0004_table1_汉江汉川站_2014.jpg' \
  --ocr-cuda off
```

构建真实流量测试 manifest：

```bash
conda run --no-capture-output -n got python datasets/build_real_flow_alignment.py
```

检查真实 manifest 的 target shape：

```bash
conda run --no-capture-output -n got python - <<'PY'
import csv, json
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path("data/manifests/flow_real_test_aligned.jsonl").read_text(encoding="utf-8").splitlines()]
shapes = Counter()
for row in rows:
    target = list(csv.reader(row["target_csv"].splitlines()))
    shapes[(len(target), max(len(r) for r in target))] += 1
print(dict(shapes))
PY
```

## 合成数据与 Swift Manifest

生成当前 v2 合成数据，并构建 `train.jsonl` / `val.jsonl`。v2 重点增强 blank-vs-zero、zero-heavy 和 29/30/31 行：

```bash
conda run --no-capture-output -n got python scripts/data/generate_synthetic_flow_v0.py \
  --dataset-dir datasets/流量/2006 \
  --output-dir data/flow_v2 \
  --manifest-dir data/manifests/flow_v2 \
  --dataset-version flow_v2 \
  --num-samples 8000 \
  --val-ratio 0.2 \
  --seed 20260428 \
  --num-workers 16
```

只重建 v2 manifest，不重新生成图片和 layout：

```bash
conda run --no-capture-output -n got python scripts/data/generate_synthetic_flow_v0.py \
  --dataset-dir datasets/流量/2006 \
  --output-dir data/flow_v2 \
  --manifest-dir data/manifests/flow_v2 \
  --dataset-version flow_v2 \
  --num-samples 8000 \
  --val-ratio 0.2 \
  --seed 20260428 \
  --manifest-only
```

保留 v1/v0 路径的旧合成数据命令：

```bash
conda run --no-capture-output -n got python scripts/data/generate_synthetic_flow_v0.py \
  --dataset-dir datasets/流量/2006 \
  --output-dir data/flow_v0 \
  --num-samples 10000 \
  --val-ratio 0.2 \
  --seed 20260408 \
  --num-workers 16
```

只重建旧 manifest，不重新生成图片和 layout：

```bash
conda run --no-capture-output -n got python scripts/data/generate_synthetic_flow_v0.py \
  --dataset-dir datasets/流量/2006 \
  --output-dir data/flow_v0 \
  --num-samples 10000 \
  --val-ratio 0.2 \
  --seed 20260408 \
  --manifest-only
```

构建 `ms-swift` 微调样本：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/build_swift_manifest.py \
  --input data/manifests/flow_v2/train.jsonl \
  --output data/manifests/flow_v2/train_swift.jsonl
```

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/build_swift_manifest.py \
  --input data/manifests/flow_v2/val.jsonl \
  --output data/manifests/flow_v2/val_swift.jsonl
```

检查 manifest 行数：

```bash
wc -l data/manifests/flow_v2/train.jsonl \
  data/manifests/flow_v2/val.jsonl \
  data/manifests/flow_v2/train_swift.jsonl \
  data/manifests/flow_v2/val_swift.jsonl \
  data/manifests/flow_real_test_aligned.jsonl
```

检查 v2 regime 分布和 zero-heavy 覆盖：

```bash
conda run --no-capture-output -n got python - <<'PY'
import csv, json
from collections import Counter
from pathlib import Path

rows = []
for path in [Path("data/manifests/flow_v2/train.jsonl"), Path("data/manifests/flow_v2/val.jsonl")]:
    rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
regimes = Counter(row["data_regime"] for row in rows)
zero_counts = []
for row in rows:
    target = list(csv.reader(row["target_csv"].splitlines()))
    zero_counts.append(sum(cell == "0" for r in target for cell in r))
print("regimes:", dict(regimes))
print("zero_count min/median/p90/max:", min(zero_counts), sorted(zero_counts)[len(zero_counts)//2], sorted(zero_counts)[int(len(zero_counts) * 0.9)], max(zero_counts))
print("zero_count >=250:", sum(count >= 250 for count in zero_counts))
PY
```

## SFT 微调

单卡稳定训练命令。当前 12GB GPU 建议 `BATCH_SIZE=1`；用 `GRAD_ACC_STEPS` 控制有效 batch size。

```bash
CUDA_VISIBLE_DEVICES=5 \
TRAIN_MANIFEST=data/manifests/flow_v2/train_swift.jsonl \
VAL_MANIFEST=data/manifests/flow_v2/val_swift.jsonl \
OUTPUT_DIR=outputs/got_ocr2_v2_swift \
NPROC_PER_NODE=1 \
EPOCHS=2 \
BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACC_STEPS=50 \
ATTN_IMPL=sdpa \
MAX_LENGTH=3072 \
SAVE_STEPS=32 \
EVAL_STEPS=32 \
LOGGING_STEPS=5 \
GRADIENT_CHECKPOINTING=true \
GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}' \
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

查看训练学习率和 loss 日志：

```bash
sed -n '1,20p' outputs/got_ocr2_v2_swift/<run-id>/logging.jsonl
```

```bash
conda run --no-capture-output -n got python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/got_ocr2_v2_swift/<run-id>/logging.jsonl")
for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    print(row.get("global_step/max_steps"), row.get("learning_rate"), row.get("loss"), row.get("token_acc"))
PY
```

## 推理

Base 模型全量真实测试推理：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/run_inference.py \
  --manifest data/manifests/flow_real_test_aligned.jsonl \
  --gpu-ids 1,2,3,4,5,6,7 \
  --backend official_chat \
  --query-mode official_format \
  --dtype float16 \
  --per-image-format raw
```

微调模型全量真实测试推理：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/run_inference.py \
  --manifest data/manifests/flow_real_test_aligned.jsonl \
  --adapter-dir outputs/got_ocr2_v2_swift/v0-20260428-154351/checkpoint-256 \
  --gpu-ids 1,2,3,4,5,6,7 \
  --backend official_chat \
  --query-mode official_format \
  --dtype float16 \
  --per-image-format raw

# 在合成的验证集上推理
conda run --no-capture-output -n got python scripts/models/got_ocr2/run_inference.py \
  --manifest data/manifests/flow_v2/val.jsonl \
  --adapter-dir outputs/got_ocr2_v2_swift/v0-20260428-154351/checkpoint-256 \
  --gpu-ids 1,2,3,4,5,6,7 \
  --limit 100 \
  --backend official_chat \
  --query-mode official_format \
  --dtype float16 \
  --output outputs/got_ocr2_v2_swift/v0-20260428-154351/eval/checkpoint-256/flow_v2_val_100_official_chat.jsonl \
  --per-image-format raw


```

历史 v1 checkpoint 示例：

```bash
outputs/got_ocr2_v1_swift/v0-20260425-032043/checkpoint-40
outputs/got_ocr2_v1_swift/v0-20260425-032043/checkpoint-240
outputs/got_ocr2_v1_swift/v0-20260425-032043/checkpoint-320
```

默认输出位置：
- base: `outputs/got_ocr2_base/eval/checkpoint-0/flow_real_all_official_chat.jsonl`
- adapter: `outputs/got_ocr2_v2_swift/<run-id>/eval/<checkpoint>/flow_real_all_official_chat.jsonl`
- per-image raw: 对应 JSONL 目录下的 `per_image_raw/`

如果需要强制重跑已经存在的样本，加 `--overwrite`：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/run_inference.py \
  --manifest data/manifests/flow_real_test_aligned.jsonl \
  --adapter-dir outputs/got_ocr2_v2_swift/<run-id>/<checkpoint> \
  --gpu-ids 1,2,3,4,6,7 \
  --backend official_chat \
  --query-mode official_format \
  --dtype float16 \
  --per-image-format raw \
  --overwrite
```

只跑前 5 张 smoke test：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/run_inference.py \
  --manifest data/manifests/flow_real_test_aligned.jsonl \
  --limit 5 \
  --backend official_chat \
  --query-mode official_format \
  --dtype float16 \
  --per-image-format raw
```

单张图片推理：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/run_inference.py \
  --image 'datasets/derived/final_test_layout/2006_流量_6-16_汉江区/station_tables_daily/page_0005_table0_沉湖月万福闸(泵站)站_2006.jpg' \
  --backend official_chat \
  --query-mode official_format \
  --dtype float16
```

手动合并 shard 文件：

```bash
conda run --no-capture-output -n got python scripts/models/got_ocr2/merge_inference_shards.py \
  outputs/got_ocr2_v2_swift/<run-id>/eval/<checkpoint>/flow_real_all_official_chat.shard*of*.jsonl \
  --output outputs/got_ocr2_v2_swift/<run-id>/eval/<checkpoint>/flow_real_all_official_chat.jsonl
```

## 评测与模型对比

表格格式评测：

当前推理输出使用 raw GOT 或 pretty LaTeX 表格文本；默认评测会解析这些表格行，再与
`target_csv` 的 source-of-truth 单元格比较。当预测可以解析成表格时，character accuracy
也基于解析后的单元格文本，而不是原始 LaTeX markup。

```bash
conda run --no-capture-output -n got python scripts/eval/evaluate_predictions.py \
  --predictions outputs/got_ocr2_base/eval/checkpoint-0/flow_real_all_official_chat.jsonl \
  --output outputs/reports/got_ocr2_base_eval.json
```

```bash
conda run --no-capture-output -n got python scripts/eval/evaluate_predictions.py \
  --predictions outputs/got_ocr2_v2_swift/v0-20260428-154351/eval/checkpoint-256/flow_real_all_official_chat.jsonl \
  --output outputs/reports/got_ocr2_v2_256_eval.json
```

按结构/单元格正确性分类 per-image raw 输出，便于人工检查：

```bash
conda run --no-capture-output -n got python scripts/eval/classify_prediction_outputs.py \
  --predictions outputs/got_ocr2_v2_swift/v0-20260428-154351/eval/checkpoint-256/flow_real_all_official_chat.jsonl \
  --raw-dir outputs/got_ocr2_v2_swift/v0-20260428-154351/eval/checkpoint-256/per_image_raw \
  --output-dir outputs/got_ocr2_v2_swift/v0-20260428-154351/eval/checkpoint-256/per_image_raw_by_eval
```

Base vs fine-tuned 结构和单元格准确率对比：

```bash
conda run --no-capture-output -n got python scripts/eval/compare_structure_cell_accuracy.py \
  --before-predictions outputs/got_ocr2_base/eval/checkpoint-0/flow_real_all_official_chat.jsonl \
  --after-predictions outputs/got_ocr2_v2_swift/v0-20260428-154351/eval/checkpoint-256/flow_real_all_official_chat.jsonl \
  --before-label base \
  --after-label checkpoint-256 \
  --output-dir outputs/model_comparisons/base_vs_v2_checkpoint-256 \
  --no-plots
```

已跑过的对比目录示例：

```bash
outputs/model_comparisons/base_vs_v1_ckpt40
outputs/model_comparisons/base_vs_v1_ckpt240
outputs/model_comparisons/base_vs_v1_ckpt320
```

查看对比摘要：

```bash
sed -n '1,20p' outputs/model_comparisons/base_vs_v1_<checkpoint>/summary.csv
sed -n '1,40p' outputs/model_comparisons/base_vs_v1_<checkpoint>/structure_transitions.csv
```

## 重新生成 target 后刷新旧 prediction JSONL

如果只改了 manifest target 清洗规则，已有 prediction JSONL 中嵌入的 `target_csv` / `target_got_format` 可能过期。可先生成临时刷新版再评测：

```bash
conda run --no-capture-output -n got python - <<'PY'
import json
from pathlib import Path

manifest_path = Path("data/manifests/flow_real_test_aligned.jsonl")
prediction_path = Path("outputs/got_ocr2_v2_swift/<run-id>/eval/<checkpoint>/flow_real_all_official_chat.jsonl")
output_path = Path("/tmp/flow_real_all_official_chat_refreshed_targets.jsonl")

manifest = {row["sample_id"]: row for row in (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines())}
rows = []
for line in prediction_path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    target = manifest[row["sample_id"]]
    row["target_csv"] = target["target_csv"]
    row["target_got_format"] = target["target_got_format"]
    rows.append(row)
output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
print(output_path)
PY
```

## 常用 Git 命令

提交前查看变更：

```bash
git status --short
git diff --stat
git diff --check
```

提交：

```bash
git add .
git commit -m "Your summary line" -m "- First change summary
- Second change summary
- Third change summary"
```

推送需要手动执行：

```bash
git push origin <branch>
```
