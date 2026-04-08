#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-got}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GOT_REF_DIR="${GOT_REF_DIR:-$PROJECT_ROOT/references/GOT-OCR2.0-main/GOT-OCR-2.0-master}"

echo "[1/8] Checking basic system tools"
command -v git >/dev/null
command -v gcc >/dev/null
command -v conda >/dev/null

echo "[2/8] Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y

echo "[3/8] Installing Python packages"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install torch torchvision
conda run -n "$ENV_NAME" python -m pip install \
  transformers==4.37.2 \
  deepspeed==0.12.3 \
  peft==0.4.0 \
  accelerate==0.28.0 \
  bitsandbytes==0.41.0 \
  sentencepiece \
  tokenizers==0.15.2 \
  timm==0.6.13 \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  markdown2[all] \
  numpy \
  requests \
  wandb \
  shortuuid \
  httpx==0.24.0 \
  albumentations \
  opencv-python \
  scikit-learn==1.2.2 \
  tiktoken==0.6.0 \
  Pillow \
  PyYAML

echo "[4/8] Installing flash-attn prerequisites"
conda run -n "$ENV_NAME" python -m pip install ninja || true
conda run -n "$ENV_NAME" python -m pip install flash-attn --no-build-isolation || true

echo "[5/8] Installing ms-swift"
conda run -n "$ENV_NAME" python -m pip install -U ms-swift[llm]

echo "[6/8] Installing local GOT reference"
if [[ ! -d "$GOT_REF_DIR" ]]; then
  echo "GOT reference directory not found: $GOT_REF_DIR" >&2
  exit 1
fi
conda run -n "$ENV_NAME" python -m pip install -e "$GOT_REF_DIR"

echo "[7/8] Verifying key imports"
conda run -n "$ENV_NAME" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
conda run -n "$ENV_NAME" python -c "import transformers; print('transformers', transformers.__version__)"
conda run -n "$ENV_NAME" python -c "import swift; print('swift ok')"
conda run -n "$ENV_NAME" python -c "import deepspeed; print('deepspeed ok')"
conda run -n "$ENV_NAME" python -c "import GOT; print('got ok')"

echo "[8/8] Done"
echo "Activate with: conda activate $ENV_NAME"
