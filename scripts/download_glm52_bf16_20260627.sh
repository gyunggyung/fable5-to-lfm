#!/usr/bin/env bash
set -euo pipefail

# Download the trainable BF16 GLM-5.2 checkpoint into /home/work/.data.
# The native FP8 checkpoint is already useful for vLLM serving/eval, but the
# BF16 repo is needed for 4-bit QLoRA because the FP8 matmul path is not
# trainable in current Transformers.
#
# Dry-run:
#   bash fable_distillation/scripts/download_glm52_bf16_20260627.sh
#
# Real resumable download:
#   RUN_NOW=1 bash fable_distillation/scripts/download_glm52_bf16_20260627.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

if [[ -f "$FABLE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FABLE_DIR/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HUB_CACHE"

RUN_NOW="${RUN_NOW:-0}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-5.2}"
DOWNLOAD_ENV="${DOWNLOAD_ENV:-$FABLE_DIR/.venvs/glm52-vllm-cu129-release-driver570}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/20260627_glm52_bf16_download}"
MAX_WORKERS="${MAX_WORKERS:-8}"
mkdir -p "$LOG_DIR"

hf_cmd=(
  "$DOWNLOAD_ENV/bin/hf" download "$MODEL_ID"
  --type model
  --cache-dir "$HF_HUB_CACHE"
  --max-workers "$MAX_WORKERS"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf_cmd+=(--token "$HF_TOKEN")
fi

printf 'model=%s\ncache=%s\nlog_dir=%s\nmax_workers=%s\n' \
  "$MODEL_ID" "$HF_HUB_CACHE" "$LOG_DIR" "$MAX_WORKERS"
df -h "$HF_HOME" | tee "$LOG_DIR/df_before.txt"

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to download."
  printf 'download command:\n  '
  printf '%q ' "${hf_cmd[@]/${HF_TOKEN:-__NO_TOKEN__}/__HF_TOKEN_REDACTED__}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$DOWNLOAD_ENV/bin/hf" ]]; then
  echo "missing hf CLI: $DOWNLOAD_ENV/bin/hf" >&2
  exit 2
fi

"${hf_cmd[@]}" 2>&1 | tee "$LOG_DIR/download.log"
df -h "$HF_HOME" | tee "$LOG_DIR/df_after.txt"
