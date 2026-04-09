# GOT-OCR2.0 Fine-tuning

## Recommended Path

For this repository, use `ms-swift + LoRA` for v0 instead of modifying the native GOT training code first.

Reason:
- the official repository already documents `ms-swift` fine-tuning for custom data
- the native GOT training path requires editing internal dataset registration in `constants.py` and `conversation_dataset_qwen.py`
- v0 only needs a stable single-model baseline on fixed-layout flow tables

Reference code:
- `references/GOT-OCR2.0-main/GOT-OCR-2.0-master/GOT/train/train_GOT.py`
- `references/GOT-OCR2.0-main/GOT-OCR-2.0-master/GOT/train/train_lora.py`
- `references/GOT-OCR2.0-main/README.md`

## Environment

- use the Linux `got` conda environment
- keep GOT reference code under `references/`
- keep this repository's adapters under `scripts/models/got_ocr2/`
- use `sdpa` as the stable attention backend on this machine

If you need a reproducible Linux setup, start with:

```bash
bash ./scripts/setup_ocr_env.sh
```

Then verify the result against the current stable path documented in [environment-setup.md](/home/yubin/py_prj/yearbook_VLM/docs/environment-setup.md). Do not reintroduce `flash-attn`, `bitsandbytes==0.41.0`, or `peft==0.4.0` into the environment.

## Current Verified Status

The current verified training state as of 2026-04-08 is:
- tests pass in `got` with `6 passed`
- synthetic data smoke generation has passed
- real test manifest generation has passed with 35 records
- `train_swift.jsonl` and `val_swift.jsonl` have been built successfully
- a GOT-OCR2.0 LoRA smoke training run has succeeded with `sdpa`
- single-GPU training has been validated as the current stable route on this machine

Recorded smoke run:
- output directory: `outputs/got_ocr2_v0_smoke_sdpa/v0-20260408-133207`
- checkpoint: `outputs/got_ocr2_v0_smoke_sdpa/v0-20260408-133207/checkpoint-1`
- train loss around `0.2235`
- eval loss around `0.2534`

## Data Preparation

1. Generate synthetic tables:

```bash
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000
```

2. Build `ms-swift` manifests:

```bash
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
```

The manifest format is:

```json
{"query":"<image>Read the table and output only CSV text....","response":"...", "images":["...png"]}
```

## Training Procedure

1. Run a smoke test on a small subset first.
2. Run LoRA fine-tuning with `swift sft`.
3. Save checkpoints every fixed number of steps.
4. Select the best checkpoint by external evaluation on synthetic validation data.
5. Run final inference on the real extracted flow table images.

Suggested v0 defaults:
- model: `stepfun-ai/GOT-OCR2_0`
- method: LoRA
- attention backend: `sdpa`
- task: single table image to raw CSV
- train source: synthetic only
- val source: synthetic holdout
- test source: real flow tables only
- stable baseline route on this machine: single GPU

For the current official baseline on this machine, the recommended starting point is:
- `CUDA_VISIBLE_DEVICES=0`
- `NPROC_PER_NODE=1`
- per-device train batch size: `1`
- per-device eval batch size: `1`
- gradient accumulation steps: `2`
- max length: `4096`
- `ATTN_IMPL=sdpa`
- gradient checkpointing with `use_reentrant=false`

Reason:
- the current flow-table manifests are well below `4096` total sequence length in practice
- `4096` ran successfully on a single GPU
- single-GPU training avoids the current multi-GPU parallelism instability on this machine

For the current `8 x 12GB` GPU setup, the recommended starting point is:
- DDP: `NPROC_PER_NODE=8`
- DeepSpeed: `configs/deepspeed_zero2.json`
- per-device train batch size: `1`
- per-device eval batch size: `1`
- gradient accumulation steps: `2`
- max length: `3072`
- `ddp_find_unused_parameters=false`
- gradient checkpointing with `use_reentrant=false`
- effective global train batch size: `16`

Reason:
- a 1-step smoke test already fit on one GPU with roughly 8.8 GiB used
- `sdpa` is currently more reliable than `flash-attn` on this machine
- `batch_size=1` leaves headroom for longer sequences and evaluation
- `grad_acc=2` gives a useful global batch without pushing 12GB cards too hard
- `3072` is a safer default than `4096` on `12GB` TITAN V cards
- non-reentrant gradient checkpointing is safer with DDP than the default reentrant mode for this model

Known environment findings:
- `bitsandbytes==0.41.0` blocked `swift` startup through `peft`
- older `peft==0.4.0` was not compatible with the current `ms-swift` stack
- `flash-attn==2.8.3` failed to load because it required `GLIBC_2.32` while the system provides `2.31`
- `device_map=auto` should not be used as the multi-GPU training strategy here
- `device_map=auto` resolved to `model.hf_device_map: {'': 7}` in this environment, so the model was not actually sharded across all GPUs
- `LoRA + gradient checkpointing + DDP(device_map)` triggered `Expected to mark a variable ready only once`
- explicit ZeRO2 config parsing worked, but the 8-GPU smoke attempt still hung before producing `logging.jsonl`

## Training Command

Use the wrapper:

```bash
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

Or run `swift sft` directly if needed.

The wrapper now follows the same CLI style as the verified successful smoke command:
- it passes `--model`
- it passes `--train_type lora`

For compatibility with older local shell habits:
- `MODEL` is the primary environment variable
- `MODEL_ID` is still accepted as a fallback alias by the wrapper

The wrapper is already set up for multi-GPU DDP through `ms-swift`.
`swift` will invoke distributed launch automatically when `NPROC_PER_NODE` is set.

Recommended multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DEEPSPEED_CONFIG=configs/deepspeed_zero2.json \
BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACC_STEPS=2 \
ATTN_IMPL=sdpa \
MAX_LENGTH=3072 \
DDP_FIND_UNUSED_PARAMETERS=false \
GRADIENT_CHECKPOINTING=true \
GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}' \
SAVE_STEPS=200 \
EVAL_STEPS=200 \
LOGGING_STEPS=20 \
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

Recommended single-GPU baseline example:

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACC_STEPS=2 \
ATTN_IMPL=sdpa \
MAX_LENGTH=4096 \
GRADIENT_CHECKPOINTING=true \
GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}' \
SAVE_STEPS=20 \
EVAL_STEPS=20 \
LOGGING_STEPS=20 \
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

If memory is still tight, reduce pressure in this order:
- keep `BATCH_SIZE=1`
- increase `GRAD_ACC_STEPS`
- lower `MAX_LENGTH`

If the run is stable and GPU memory headroom remains, the next thing to try is:
- keep `BATCH_SIZE=1`
- raise `GRAD_ACC_STEPS` from `2` to `4`
- then consider raising `MAX_LENGTH` from `3072` to `4096`

If you hit a DDP error like `Expected to mark a variable ready only once`, first try:
- keep `DDP_FIND_UNUSED_PARAMETERS=false`
- keep `GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}'`

If it still happens, the next fallback is:
- `GRADIENT_CHECKPOINTING=false`

Avoid this for training on the current machine:
- single-process pseudo-multi-GPU with `device_map=auto`

Preferred direction for the next multi-GPU attempt:
- standard multi-process launch with `NPROC_PER_NODE=8`
- `--deepspeed configs/deepspeed_zero2.json`
- `ATTN_IMPL=sdpa`
- `GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}'`
- `MAX_LENGTH=3072`

Current recommendation:
- proceed with the single-GPU baseline first
- treat multi-GPU tuning as a separate follow-up task

## Model Weights And Cache

There are two different kinds of artifacts during training:

- base model weights downloaded by `swift` and `modelscope`
- fine-tuning outputs produced by this repository

The fine-tuning outputs should stay inside this project, for example:

```bash
outputs/got_ocr2_v0_swift/
```

The base model weights do not have to live inside the project root, but using a project-local cache is often easier to inspect and reproduce.

The wrapper now defaults to a project-local cache root:

```bash
outputs/cache/
```

This project should keep using `outputs/cache/` as the unified cache location for training runs.

Inside that cache root, the important locations are:

```bash
outputs/cache/modelscope/   # downloaded GOT-OCR2.0 weights from ModelScope
outputs/cache/hf_home/      # Hugging Face dynamic modules and related cache
outputs/cache/matplotlib/   # matplotlib runtime cache
outputs/cache/home/         # temporary HOME for tools that insist on writing under ~
```

You can override the cache root if needed:

```bash
CACHE_ROOT=/path/to/cache bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

You can also point `swift` to an already-downloaded local model directory instead of the remote model id:

```bash
MODEL_ID=/path/to/GOT-OCR2_0 bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

This is useful when:
- the machine has limited network access
- multiple projects share the same base model
- you want exact control over which model snapshot is used

## How The Download Happens

When `MODEL_ID` is left as `stepfun-ai/GOT-OCR2_0`, `swift sft` resolves the model through `ms-swift`, which then calls ModelScope download utilities.

In practice the chain is:

- `swift sft`
- `ms-swift` model registry for `got_ocr2`
- ModelScope snapshot download
- cached model directory reused on later runs

So the download is not implemented in this repository's Python code directly; it is delegated to the installed `ms-swift` and `modelscope` packages.

This repository's wrapper around that process is:
- [run_swift_sft.sh](/home/yubin/py_prj/yearbook_VLM/scripts/models/got_ocr2/run_swift_sft.sh)

## Evaluation

This repository uses strict raw-output evaluation:
- no post-processing
- no normalization
- no structure repair

Run evaluation after inference with:

```bash
python3 ./scripts/eval/evaluate_strict_csv.py --predictions outputs/predictions/got_ocr2_v0.jsonl --output outputs/reports/got_ocr2_v0_eval.json
```

## Notes

- The current repository already includes the data-prep and training adapter scripts.
- Real test image extraction from the source PDF is still a separate task.
- If `ms-swift` is not available in `got`, install it there rather than in another environment.
- The next meaningful training milestone is the first full multi-GPU run, not another fresh environment bootstrap.
