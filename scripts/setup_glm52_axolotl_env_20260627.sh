#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
ENV_DIR="${ENV_DIR:-/home/work/.cache/fable_distillation/venvs/glm52-axolotl-8bit-moe}"
UV_BIN="${UV_BIN:-/home/work/.local/bin/uv}"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$(dirname "$ENV_DIR")" "$HF_HUB_CACHE"
cd "$FABLE_DIR"

if [[ ! -x "$UV_BIN" ]]; then
  echo "missing uv: $UV_BIN" >&2
  exit 2
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv "$ENV_DIR" --python 3.12
fi

"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --upgrade pip wheel packaging setuptools ninja
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  --torch-backend=cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  "torch>=2.11,<2.13"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" -m pip install \
  --no-build-isolation \
  "axolotl[deepspeed] @ git+https://github.com/axolotl-ai-cloud/axolotl.git"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
import axolotl
print("axolotl", getattr(axolotl, "__version__", "unknown"))
PY

echo "ENV_DIR=$ENV_DIR"
