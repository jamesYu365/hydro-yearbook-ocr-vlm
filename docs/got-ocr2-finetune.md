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

If you need a reproducible Linux setup, start with:

```bash
bash ./scripts/setup_ocr_env.sh
```

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
- task: single table image to raw CSV
- train source: synthetic only
- val source: synthetic holdout
- test source: real flow tables only

## Training Command

Use the wrapper:

```bash
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

Or run `swift sft` directly if needed.

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
