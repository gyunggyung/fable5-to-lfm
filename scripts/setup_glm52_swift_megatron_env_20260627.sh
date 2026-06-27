#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
ENV_DIR="${ENV_DIR:-/home/work/.cache/fable_distillation/venvs/glm52-swift-megatron}"
UV_BIN="${UV_BIN:-/home/work/.local/bin/uv}"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/work/.data/modelscope}"

mkdir -p "$(dirname "$ENV_DIR")" "$HF_HUB_CACHE" "$MODELSCOPE_CACHE"
cd "$FABLE_DIR"

if [[ ! -x "$UV_BIN" ]]; then
  echo "missing uv: $UV_BIN" >&2
  exit 2
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv "$ENV_DIR" --python 3.12
fi

"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --upgrade pip wheel packaging setuptools ninja pybind11
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  --torch-backend=cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  "torch>=2.11,<2.13"

env -u PIP_CONSTRAINT -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -m pip install \
  -U "ms-swift" "deepspeed>=0.18,<0.20" "mcore-bridge>=1.3" "transformer-engine[pytorch]" \
  "peft>=0.11,<0.20"

env -u PIP_CONSTRAINT -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
try:
    import swift
    print("swift", getattr(swift, "__version__", "unknown"))
except Exception as exc:
    print("swift import failed", repr(exc))
    raise
PY

echo "ENV_DIR=$ENV_DIR"
