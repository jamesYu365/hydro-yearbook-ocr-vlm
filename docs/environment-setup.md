# Environment Setup

## Goal

Prepare a Linux `rapid` conda environment that can run:
- synthetic data generation
- GOT-OCR2.0 data conversion
- `ms-swift` LoRA fine-tuning
- strict CSV evaluation

## Recommended Baseline

Use a clean Linux environment instead of reusing a mismatched local setup.

Recommended target versions:
- Python 3.10
- CUDA-compatible PyTorch
- `transformers==4.37.2`
- `deepspeed==0.12.3`
- `peft==0.4.0`
- `accelerate==0.28.0`

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
conda create -n rapid python=3.10 -y
conda activate rapid
```

## Python Dependencies

```bash
pip install torch torchvision
pip install transformers==4.37.2 deepspeed==0.12.3 peft==0.4.0 accelerate==0.28.0 bitsandbytes==0.41.0
pip install sentencepiece tokenizers==0.15.2 timm==0.6.13 einops==0.6.1 einops-exts==0.0.4
pip install markdown2[all] numpy requests wandb shortuuid httpx==0.24.0 albumentations opencv-python scikit-learn==1.2.2 tiktoken==0.6.0
pip install Pillow PyYAML pytest
```

## Flash Attention

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

If `flash-attn` fails, keep moving with a smoke test first, then come back to optimize.

## ms-swift

```bash
pip install -U ms-swift[llm]
```

## Install Local GOT Reference

```bash
cd /path/to/yearbook_VLM/references/GOT-OCR2.0-main/GOT-OCR-2.0-master
pip install -e .
```

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

## Project Smoke Test

```bash
python ./scripts/data/generate_synthetic_flow_v0.py --num-samples 100
python ./scripts/data/build_real_test_manifest.py
python ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
```

Then run:

```bash
bash ./scripts/models/got_ocr2/run_swift_sft.sh
```

## Notes

- Install everything inside the `rapid` environment.
- Keep reference code under `references/`.
- Keep project adapters and automation in this repository.
- Real test image extraction from the source PDF is still a separate step.
