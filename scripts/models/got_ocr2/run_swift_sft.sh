#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL="${MODEL:-${MODEL_ID:-stepfun-ai/GOT-OCR2_0}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/manifests/flow_v0/train_swift.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-data/manifests/flow_v0/val_swift.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/got_ocr2_v0_swift}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/outputs/cache}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/configs/deepspeed_zero2.json}"

if [ -z "${DDP_FIND_UNUSED_PARAMETERS+x}" ]; then
  DDP_FIND_UNUSED_PARAMETERS=false
fi

if [ -z "${GRADIENT_CHECKPOINTING+x}" ]; then
  GRADIENT_CHECKPOINTING=true
fi

if [ -z "${GRADIENT_CHECKPOINTING_KWARGS+x}" ]; then
  GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}'
fi

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_ROOT/modelscope" "$CACHE_ROOT/hf_home" "$CACHE_ROOT/matplotlib" "$CACHE_ROOT/home"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export HOME="${HOME:-$CACHE_ROOT/home}"

cmd=(
  swift sft
  --model "$MODEL" \
  --attn_impl "$ATTN_IMPL" \
  --train_type lora \
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
  --logging_steps "$LOGGING_STEPS" \
  --deepspeed "$DEEPSPEED_CONFIG"
)

if [ -n "$MAX_STEPS" ]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi

if [ -n "${DDP_FIND_UNUSED_PARAMETERS:-}" ]; then
  cmd+=(--ddp_find_unused_parameters "$DDP_FIND_UNUSED_PARAMETERS")
fi

if [ -n "${GRADIENT_CHECKPOINTING:-}" ]; then
  cmd+=(--gradient_checkpointing "$GRADIENT_CHECKPOINTING")
fi

if [ -n "${GRADIENT_CHECKPOINTING_KWARGS:-}" ]; then
  cmd+=(--gradient_checkpointing_kwargs "$GRADIENT_CHECKPOINTING_KWARGS")
fi

"${cmd[@]}"
