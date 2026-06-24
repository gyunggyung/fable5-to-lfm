#!/usr/bin/env bash
set -euo pipefail

# Optional isolated vLLM env for DiffusionGemma. The existing .vllm-lfm-cu12
# can run LFM evaluations, but DiffusionGemma needs current vLLM diffusion
# support and should not depend on LFM-specific environment quirks.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
ENV_DIR="${ENV_DIR:-$FABLE_DIR/.venvs/diffusiongemma-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# Official vLLM recipe says DiffusionGemma support is vLLM 0.24.0+ / gemma
# docker image. The current package index may lag behind that image; default to
# the newest visible wheel and let callers override VLLM_SPEC when 0.24+ lands.
VLLM_SPEC="${VLLM_SPEC:-vllm==0.23.0}"

cd "$ROOT_DIR"
mkdir -p "$FABLE_DIR/.venvs"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$ENV_DIR"
fi
uv pip install --python "$ENV_DIR/bin/python" --upgrade pip wheel setuptools
uv pip install --python "$ENV_DIR/bin/python" "$VLLM_SPEC" "transformers>=5.0.0" "huggingface_hub>=0.35.0" "datasets>=4.0.0"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ENV_DIR/bin/python" - <<'PY'
import sys
import vllm
import transformers
print("python", sys.version)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
PY
