#!/usr/bin/env bash
set -euo pipefail

# Trainable GLM-5.2 path: load the BF16 checkpoint as 4-bit BitsAndBytes
# QLoRA and train only LoRA adapters. The native FP8 checkpoint is kept for
# vLLM serving/eval because its fine-grained FP8 matmul currently has no
# backward formula in Transformers.
#
# Dry-run:
#   bash fable_distillation/scripts/run_glm52_bf16_qlora_device_map_20260627.sh
#
# 1-step smoke:
#   RUN_NOW=1 MAX_STEPS=1 MAX_TRAIN_ROWS=2 MAX_SEQ_LENGTH=512 SAVE_STEPS=1 \
#     bash fable_distillation/scripts/run_glm52_bf16_qlora_device_map_20260627.sh
#
# Long resumable run:
#   RUN_NOW=1 bash fable_distillation/scripts/run_glm52_bf16_qlora_device_map_20260627.sh

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
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME="${BITSANDBYTES_NOWELCOME:-1}"
mkdir -p "$HF_HUB_CACHE"

RUN_NOW="${RUN_NOW:-0}"
RUN_ID="${RUN_ID:-20260627_glm52_bf16_fable_official_agentic_qlora}"
TRAIN_ENV="${TRAIN_ENV:-$FABLE_DIR/.venvs/glm52-vllm-cu129-release-driver570}"
MODEL_PATH="${MODEL_PATH:-zai-org/GLM-5.2}"
TRAIN_JSONL="${TRAIN_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work/.data/harness1/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/GLM-5.2__Fable-OfficialAgentic-QLoRA-20260627}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-$FABLE_DIR/.cache/tokenized/$RUN_ID}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_STEPS="${MAX_STEPS:-1200}"
LEARNING_RATE="${LEARNING_RATE:-8e-6}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.02}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
# Keep the 4-bit base on the 8x H200s. If CPU memory is exposed to
# Transformers' auto device map, BitsAndBytes can dispatch some modules to CPU
# and abort before loading the model.
GPU_MAX_MEMORY_GIB="${GPU_MAX_MEMORY_GIB:-140}"
CPU_MAX_MEMORY_GIB="${CPU_MAX_MEMORY_GIB:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
BNB_4BIT_USE_DOUBLE_QUANT="${BNB_4BIT_USE_DOUBLE_QUANT:-true}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ -x "$TRAIN_ENV/bin/python" ]]; then
  SITE_PACKAGES="$(env -u PYTHONPATH PYTHONNOUSERSITE=1 "$TRAIN_ENV/bin/python" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
else
  SITE_PACKAGES="$TRAIN_ENV/lib/python3.12/site-packages"
fi
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
TRAIN_LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_PATHS}${NVIDIA_LIB_PATHS:+:$NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

train_cmd=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  LD_LIBRARY_PATH="$TRAIN_LD_LIBRARY_PATH"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  HF_HOME="$HF_HOME"
  HF_HUB_CACHE="$HF_HUB_CACHE"
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE"
  HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD:-1}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  BITSANDBYTES_NOWELCOME="$BITSANDBYTES_NOWELCOME"
  "$TRAIN_ENV/bin/python" "$FABLE_DIR/training/train_glm52_fp8_device_map_lora.py"
  --model-path "$MODEL_PATH"
  --train-jsonl "$TRAIN_JSONL"
  --output-dir "$OUTPUT_DIR"
  --tokenized-cache-dir "$TOKENIZED_CACHE_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --max-train-rows "$MAX_TRAIN_ROWS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --per-device-train-batch-size "$PER_DEVICE_BATCH"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --target-modules "$TARGET_MODULES"
  --torch-dtype bfloat16
  --gpu-max-memory-gib "$GPU_MAX_MEMORY_GIB"
  --cpu-max-memory-gib "$CPU_MAX_MEMORY_GIB"
  --load-in-4bit
  --bnb-4bit-quant-type "$BNB_4BIT_QUANT_TYPE"
  --bnb-4bit-compute-dtype "$BNB_4BIT_COMPUTE_DTYPE"
  --bnb-4bit-use-double-quant "$BNB_4BIT_USE_DOUBLE_QUANT"
  --resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"
)

printf 'run_id=%s\nmodel=%s\noutput=%s\ntrain_jsonl=%s\nlog_dir=%s\n' \
  "$RUN_ID" "$MODEL_PATH" "$OUTPUT_DIR" "$TRAIN_JSONL" "$LOG_DIR"
printf 'train_env=%s\nsite_packages=%s\ncuda_runtime_lib_paths=%s\n' \
  "$TRAIN_ENV" "$SITE_PACKAGES" "$CUDA_RUNTIME_LIB_PATHS"
printf 'gpus=%s max_seq=%s steps=%s lr=%s lora_r=%s gpu_max_memory_gib=%s qlora=%s/%s double_quant=%s\n' \
  "$TRAIN_GPUS" "$MAX_SEQ_LENGTH" "$MAX_STEPS" "$LEARNING_RATE" "$LORA_RANK" \
  "$GPU_MAX_MEMORY_GIB" "$BNB_4BIT_QUANT_TYPE" "$BNB_4BIT_COMPUTE_DTYPE" "$BNB_4BIT_USE_DOUBLE_QUANT"
printf 'train command:\n  '
printf '%q ' "${train_cmd[@]}"
printf '\n'

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  exit 0
fi

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  echo "missing training env: $TRAIN_ENV/bin/python" >&2
  exit 2
fi

"${train_cmd[@]}" 2>&1 | tee "$LOG_DIR/train.log"
