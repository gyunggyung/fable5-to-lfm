#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 FP8 LoRA SFT runner for official-agentic targets.
#
# Dry-run:
#   bash fable_distillation/scripts/run_glm52_official_agentic_lora_20260627.sh
#
# Build data only:
#   BUILD_ONLY=1 RUN_NOW=1 bash fable_distillation/scripts/run_glm52_official_agentic_lora_20260627.sh
#
# Actual training:
#   RUN_NOW=1 bash fable_distillation/scripts/run_glm52_official_agentic_lora_20260627.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HUB_CACHE"

RUN_NOW="${RUN_NOW:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
SKIP_BUILD_DATASET="${SKIP_BUILD_DATASET:-0}"

RUN_ID="${RUN_ID:-20260627_glm52_fp8_fable_official_agentic_lora}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-zai-org/GLM-5.2-FP8}"
TRAIN_JSONL="${TRAIN_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.jsonl}"
TRAIN_META="${TRAIN_META:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.meta.json}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work/.data/harness1/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/GLM-5.2-FP8__Fable-OfficialAgentic-LoRA-20260627}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-$FABLE_DIR/.cache/tokenized/$RUN_ID}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$FABLE_DIR/configs/deepspeed_zero3_glm52_lora_20260627.json}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_STEPS="${MAX_STEPS:-1200}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-8e-6}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.02}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-{}}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

build_dataset_cmd=(
  python "$FABLE_DIR/scripts/build_official_agentic_sft_mix_20260627.py"
  --output "$TRAIN_JSONL"
  --meta "$TRAIN_META"
)

train_cmd=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  TOKENIZERS_PARALLELISM=false
  "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE"
  "$FABLE_DIR/training/train_multifamily_chat_sft.py"
  --model-path "$MODEL_PATH"
  --model-class causal-lm
  --train-jsonl "$TRAIN_JSONL"
  --output-dir "$OUTPUT_DIR"
  --finetune-mode lora
  --torch-dtype auto
  --max-seq-length "$MAX_SEQ_LENGTH"
  --max-train-rows "$MAX_TRAIN_ROWS"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --per-device-train-batch-size "$PER_DEVICE_BATCH"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --logging-steps 1
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --target-modules "$TARGET_MODULES"
  --chat-serialization native
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON"
  --deepspeed-config "$DEEPSPEED_CONFIG"
)

if [[ -n "$TOKENIZED_CACHE_DIR" ]]; then
  train_cmd+=(--tokenized-cache-dir "$TOKENIZED_CACHE_DIR")
fi

print_cmd() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

printf 'run_id=%s\nmodel=%s\noutput=%s\ntrain_jsonl=%s\nmeta=%s\nlog_dir=%s\n' \
  "$RUN_ID" "$MODEL_PATH" "$OUTPUT_DIR" "$TRAIN_JSONL" "$TRAIN_META" "$LOG_DIR"
printf 'hf_home=%s\nhf_hub_cache=%s\n' "$HF_HOME" "$HF_HUB_CACHE"
printf 'gpus=%s nproc=%s max_seq=%s steps=%s lr=%s lora_r=%s targets=%s\n' \
  "$TRAIN_GPUS" "$NPROC_PER_NODE" "$MAX_SEQ_LENGTH" "$MAX_STEPS" "$LEARNING_RATE" "$LORA_RANK" "$TARGET_MODULES"
print_cmd "dataset command" "${build_dataset_cmd[@]}"
print_cmd "train command" "${train_cmd[@]}"

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  exit 0
fi

if [[ "$SKIP_BUILD_DATASET" == "1" ]]; then
  echo "skip dataset build: TRAIN_JSONL=$TRAIN_JSONL" | tee "$LOG_DIR/build_dataset.log"
else
  "${build_dataset_cmd[@]}" 2>&1 | tee "$LOG_DIR/build_dataset.log"
fi

if [[ "$BUILD_ONLY" == "1" ]]; then
  echo "BUILD_ONLY=1; not starting training."
  exit 0
fi

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  echo "missing training env: $TRAIN_ENV/bin/python" >&2
  echo "Create it first with scripts/setup_glm52_lora_env_20260627.sh or set TRAIN_ENV." >&2
  exit 2
fi

"${train_cmd[@]}" 2>&1 | tee "$LOG_DIR/train.log"
