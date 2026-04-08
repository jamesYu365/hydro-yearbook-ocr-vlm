#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL_ID="${MODEL_ID:-stepfun-ai/GOT-OCR2_0}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/manifests/flow_v0/train_swift.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-data/manifests/flow_v0/val_swift.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/got_ocr2_v0_swift}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
SAVE_STEPS="${SAVE_STEPS:-200}"

cd "$PROJECT_ROOT"

swift sft \
  --model_type got-ocr2 \
  --model_id_or_path "$MODEL_ID" \
  --sft_type lora \
  --dataset "$TRAIN_MANIFEST" \
  --val_dataset "$VAL_MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --max_length "$MAX_LENGTH" \
  --save_steps "$SAVE_STEPS"

