#!/usr/bin/env bash
set -euo pipefail

# Docker-free DiffusionGemma Transformers env.
# vLLM 0.23 currently pulls torch cu130 in this environment, which cannot
# initialize CUDA with the installed 12.9 driver. This env intentionally avoids
# vLLM and pins torch to a CUDA 12.8 wheel that works on the local H200 driver.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
ENV_DIR="${ENV_DIR:-$FABLE_DIR/.venvs/diffusiongemma-transformers-cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

cd "$ROOT_DIR"
mkdir -p "$FABLE_DIR/.venvs"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$ENV_DIR"
fi

uv pip install --python "$ENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
uv pip install --python "$ENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" "torchvision"
uv pip install --python "$ENV_DIR/bin/python" \
  "transformers==5.12.1" \
  "accelerate>=1.10.0" \
  "huggingface_hub>=0.35.0" \
  "datasets>=4.0.0" \
  "safetensors>=0.4.5" \
  "sentencepiece>=0.2.0" \
  "pillow>=10.0.0"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import accelerate

print("python cuda env ok")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
PY
