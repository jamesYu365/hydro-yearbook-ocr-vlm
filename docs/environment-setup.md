# Environment Setup

## Goal

Prepare a Linux `got` conda environment that can run:
- synthetic data generation
- GOT-OCR2.0 data conversion
- `ms-swift` LoRA fine-tuning
- GOT inference and report generation
- strict CSV evaluation when needed

Prepare a separate Linux `rapid` conda environment that can run:
- real PDF page rendering
- page-level layout detection
- station-level table cropping

## Recommended Baseline

Use a clean Linux environment instead of reusing a mismatched local setup.

Recommended baseline:
- Python 3.10
- CUDA-compatible PyTorch
- `transformers==4.37.2`
- `deepspeed==0.12.3`
- `accelerate==0.28.0`
- `ms-swift[llm]`

Verified stable path on this machine:
- environment: `got`
- attention backend: `sdpa`
- fine-tuning method: LoRA
- test command: `conda run -n got pytest -q`

Avoid on the current machine:
- `flash-attn`
- `bitsandbytes==0.41.0`
- `peft==0.4.0`

## System Checks

Before installation, confirm:

```bash
nvidia-smi
nvcc --version
gcc --version
git --version
conda --version
```

## Conda Environment

If the environment already exists, reuse it. Otherwise:

```bash
conda create -n got python=3.10 -y
conda activate got
```

For real-test extraction, use a separate environment:

```bash
conda create -n rapid python=3.10 -y
conda activate rapid
```

## Python Dependencies

```bash
pip install torch torchvision
pip install transformers==4.37.2 deepspeed==0.12.3 accelerate==0.28.0
pip install sentencepiece tokenizers==0.15.2 timm==0.6.13 einops==0.6.1 einops-exts==0.0.4
pip install markdown2[all] numpy requests wandb shortuuid httpx==0.24.0 albumentations opencv-python scikit-learn==1.2.2 tiktoken==0.6.0
pip install Pillow PyYAML pytest
pip install -U ms-swift[llm]
```

Install the local GOT reference after the Python stack is in place:

```bash
cd /path/to/yearbook_VLM/references/GOT-OCR2.0-main/GOT-OCR-2.0-master
pip install -e .
```

If `ms-swift` installs a newer compatible `peft`, keep that version. Do not pin `peft==0.4.0` back into the environment.

## Rapid Extraction Dependencies

Install the extraction stack in `rapid`:

```bash
conda activate rapid
pip install pymupdf opencv-python
pip install rapid_table_det
pip install rapidocr==3.8.1
```

Verify the extraction environment:

```bash
conda run -n rapid python -c "import fitz; print('fitz ok')"
conda run -n rapid python -c "import cv2; print('cv2 ok')"
conda run -n rapid python -c "from rapid_table_det.inference import TableDetector; print('rapid_table_det ok')"
conda run -n rapid python -c "from rapidocr import RapidOCR; print('rapidocr ok')"
```

## Attention Backend

Use `sdpa` as the current default backend.

Do not treat `flash-attn` as part of the stable setup on this machine:
- `flash-attn==2.8.3` failed to load because it required `GLIBC_2.32`
- the current system `glibc` is `2.31`
- the smoke training run succeeded without `flash-attn`

## Verification

Return to the project root and verify the environment:

```bash
cd /path/to/yearbook_VLM
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import swift; print('swift ok')"
python -c "import deepspeed; print('deepspeed ok')"
python -c "import GOT; print('got ok')"
pytest -q
```

Current verified result on this project:
- focused data, manifest, inference, and evaluation suites pass in the `got` environment
- current key command: `conda run -n got pytest -q tests/test_flow_common.py tests/test_backfill_got_format_targets.py tests/test_real_flow_test_prep.py tests/test_evaluate_strict_csv.py tests/test_generate_synthetic_flow_v0.py tests/test_got_inference.py tests/test_model_comparison.py`

## Chinese Fonts

Synthetic rendering requires a font that can display Chinese headers such as `流量表` and `日期`.

Recommended Linux font paths on this machine class include:

```bash
/usr/share/fonts/MyFonts/simsun.ttc
/usr/share/fonts/MyFonts/MSYH.TTC
/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf
```

You can inspect available Chinese fonts with:

```bash
fc-list :lang=zh family file
```

## Project Smoke Test

```bash
python ./scripts/data/generate_synthetic_flow_v0.py --num-samples 100 --val-ratio 0.2 --seed 20260408
python ./datasets/build_real_flow_alignment.py
python ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
python ./scripts/models/got_ocr2/run_inference.py --manifest data/manifests/flow_real_test_aligned.jsonl --limit 1 --backend official_chat --query-mode official_format
```

Then run:

```bash
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

The current wrapper defaults already match the stable path:
- `ATTN_IMPL=sdpa`
- `NPROC_PER_NODE=8`
- `DEEPSPEED_CONFIG=configs/deepspeed_zero2.json`
- `BATCH_SIZE=1`
- `EVAL_BATCH_SIZE=1`
- `GRAD_ACC_STEPS=2`
- `MAX_LENGTH=3072`
- `DDP_FIND_UNUSED_PARAMETERS=false`
- `GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}'`

If you want the downloaded base model and runtime caches to stay inside the repository instead of under your home directory, keep using the wrapper above.
It writes caches under:

```bash
outputs/cache/
```

## Notes

- Install everything inside the `got` environment.
- Use the `rapid` environment for real-test PDF rendering and layout extraction.
- Keep reference code under `references/`.
- Keep shared project logic under `src/yearbook_ocr/` and use `datasets/` / `scripts/` as thin entrypoints.
- Ensure the synthetic renderer is using a Chinese-capable font before generating the final dataset.
- Real test extraction code lives under `datasets/`, and its outputs should stay under `datasets/derived/`.
- Title OCR for extracted tables runs only on a dedicated title ROI built from the added top buffer strip plus a small downward compensation below the original table top.
- A Linux kernel warning below `5.5.0` was observed during training logs. It did not block the smoke run, but longer runs should be monitored.
- Avoid using `device_map=auto` as the training parallelism strategy for multi-GPU fine-tuning.
