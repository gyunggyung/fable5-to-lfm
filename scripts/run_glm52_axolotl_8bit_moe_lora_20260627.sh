#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

if [[ -f "$FABLE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FABLE_DIR/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ACCELERATE_USE_FSDP=1

RUN_ID="${RUN_ID:-20260627_glm52_axolotl_8bit_moe_lora}"
AXOLOTL_ENV="${AXOLOTL_ENV:-$FABLE_DIR/.venvs/glm52-axolotl-8bit-moe}"
CONFIG_PATH="${CONFIG_PATH:-$FABLE_DIR/configs/axolotl_glm52_8bit_moe_lora_20260627.yml}"
SOURCE_JSONL="${SOURCE_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.jsonl}"
AXOLOTL_JSONL="${AXOLOTL_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.axolotl_chatml.jsonl}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"

mkdir -p "$LOG_DIR"

if [[ ! -s "$AXOLOTL_JSONL" ]]; then
  "$AXOLOTL_ENV/bin/python" scripts/prepare_axolotl_glm52_chatml_20260627.py \
    --input "$SOURCE_JSONL" \
    --output "$AXOLOTL_JSONL"
fi

if [[ "${RUN_NOW:-0}" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  echo "RUN_ID=$RUN_ID"
  echo "AXOLOTL_ENV=$AXOLOTL_ENV"
  echo "CONFIG_PATH=$CONFIG_PATH"
  echo "AXOLOTL_JSONL=$AXOLOTL_JSONL"
  exit 0
fi

if [[ ! -x "$AXOLOTL_ENV/bin/axolotl" ]]; then
  echo "missing axolotl env: $AXOLOTL_ENV/bin/axolotl" >&2
  echo "run scripts/setup_glm52_axolotl_env_20260627.sh first" >&2
  exit 2
fi

env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$AXOLOTL_ENV/bin/accelerate" launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m axolotl.cli.train "$CONFIG_PATH" 2>&1 | tee "$LOG_DIR/train.log"
