#!/usr/bin/env bash
set -euo pipefail

# Docker-free vLLM environment for GLM-5.2-FP8.
# Official vLLM recipe for GLM-5.2 targets vLLM 0.23.0 and
# transformers >= 5.9.0. Keep this env separate from the older LFM vLLM env.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
ENV_DIR="${ENV_DIR:-$FABLE_DIR/.venvs/glm52-vllm-cu129-release-driver570}"
UV_BIN="${UV_BIN:-/home/work/.local/bin/uv}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
VLLM_CUDA_VERSION="${VLLM_CUDA_VERSION:-129}"
TORCH_BACKEND="${TORCH_BACKEND:-cu${VLLM_CUDA_VERSION}}"
CPU_ARCH="${CPU_ARCH:-$(uname -m)}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcu${VLLM_CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl}"
BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION:-0.49.2}"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$HF_HUB_CACHE" "$FABLE_DIR/.venvs"

if [[ ! -x "$UV_BIN" ]]; then
  echo "missing uv: $UV_BIN" >&2
  exit 2
fi

cd "$FABLE_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv "$ENV_DIR" --python 3.12
fi

"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --upgrade pip wheel packaging setuptools
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" "$VLLM_WHEEL_URL" --torch-backend="$TORCH_BACKEND" --extra-index-url "https://download.pytorch.org/whl/cu${VLLM_CUDA_VERSION}" --index-strategy unsafe-best-match
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  "transformers>=5.9.0" \
  "openai>=1.0.0" \
  "peft>=0.19.0" \
  "accelerate>=1.14.0" \
  "datasets>=5.0.0" \
  "kernels>=0.12,<0.13"
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -m pip install \
  --no-deps \
  --force-reinstall \
  "bitsandbytes==$BITSANDBYTES_VERSION"

SITE_PACKAGES="$(env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
NVIDIA_LIB_PATHS="$(
  find "$SITE_PACKAGES/nvidia" -type d -name lib 2>/dev/null | paste -sd: -
)"
PRIMARY_CUDA_RUNTIME_LIB="$SITE_PACKAGES/nvidia/cuda_runtime/lib"
OTHER_CUDA_RUNTIME_LIB_PATHS="$(
  find "$SITE_PACKAGES/nvidia" -name 'libcudart.so*' -printf '%h\n' 2>/dev/null \
    | sort -u \
    | awk -v primary="$PRIMARY_CUDA_RUNTIME_LIB" '$0 != primary' \
    | paste -sd: -
)"
CUDA_RUNTIME_LIB_PATHS="$PRIMARY_CUDA_RUNTIME_LIB${OTHER_CUDA_RUNTIME_LIB_PATHS:+:$OTHER_CUDA_RUNTIME_LIB_PATHS}"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_PATHS}${NVIDIA_LIB_PATHS:+:$NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import vllm
import bitsandbytes as bnb

print("env ready")
print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("bitsandbytes", bnb.__version__)
PY

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/vllm" --version

echo "ENV_DIR=$ENV_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "VLLM_WHEEL_URL=$VLLM_WHEEL_URL"
echo "TORCH_BACKEND=$TORCH_BACKEND"
echo "BITSANDBYTES_VERSION=$BITSANDBYTES_VERSION"
echo "CUDA runtime lib paths=$CUDA_RUNTIME_LIB_PATHS"
