#!/usr/bin/env bash
set -euo pipefail

# Docker-free vLLM environment for GLM-5.2-FP8.
# Official vLLM recipe for GLM-5.2 targets vLLM 0.23.0 and
# transformers >= 5.9.0. Keep this env separate from the older LFM vLLM env.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
ENV_DIR="${ENV_DIR:-$FABLE_DIR/.venvs/glm52-vllm-cu128}"
UV_BIN="${UV_BIN:-/home/work/.local/bin/uv}"

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
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" "vllm==0.23.0" --torch-backend=auto
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" "transformers>=5.9.0" "openai>=1.0.0"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import vllm

print("env ready")
print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY

echo "ENV_DIR=$ENV_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
