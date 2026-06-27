#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
UV_BIN="${UV_BIN:-/home/work/.local/bin/uv}"

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  echo "missing python: $TRAIN_ENV/bin/python" >&2
  exit 2
fi

echo "python=$("$TRAIN_ENV/bin/python" -V 2>&1)"

if "$TRAIN_ENV/bin/python" - <<'PY'
import deepspeed
print("deepspeed", getattr(deepspeed, "__version__", "ok"))
PY
then
  echo "deepspeed already available"
  exit 0
fi

echo "installing deepspeed into $TRAIN_ENV"
DS_BUILD_OPS="${DS_BUILD_OPS:-0}" \
  "$UV_BIN" pip install --python "$TRAIN_ENV/bin/python" 'deepspeed>=0.17.0'

"$TRAIN_ENV/bin/python" - <<'PY'
import deepspeed
print("deepspeed", getattr(deepspeed, "__version__", "ok"))
PY
