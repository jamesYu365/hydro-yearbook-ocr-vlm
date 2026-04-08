#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL_ID="${MODEL_ID:-stepfun-ai/GOT-OCR2_0}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/manifests/flow_v0/train_swift.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-data/manifests/flow_v0/val_swift.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/got_ocr2_v0_swift}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/outputs/cache}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_ROOT/modelscope" "$CACHE_ROOT/hf_home" "$CACHE_ROOT/matplotlib" "$CACHE_ROOT/home"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export HOME="${HOME:-$CACHE_ROOT/home}"

swift sft \
  --model_type got-ocr2 \
  --model_id_or_path "$MODEL_ID" \
  --attn_impl "$ATTN_IMPL" \
  --sft_type lora \
  --dataset "$TRAIN_MANIFEST" \
  --val_dataset "$VAL_MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --max_length "$MAX_LENGTH" \
  --save_steps "$SAVE_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --evaluation_strategy steps \
  --logging_steps "$LOGGING_STEPS"
