#!/usr/bin/env bash
set -euo pipefail

# Smoke LoRA SFT runner for Gemma/Qwen-style models.
# Dry-run by default:
#   bash fable_distillation/scripts/run_multifamily_sft_smoke_20260624.sh
# Actual run:
#   RUN_NOW=1 MODEL_PRESET=gemma4_12b_it bash fable_distillation/scripts/run_multifamily_sft_smoke_20260624.sh
#   RUN_NOW=1 MODEL_PRESET=qwen35_9b bash fable_distillation/scripts/run_multifamily_sft_smoke_20260624.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

RUN_NOW="${RUN_NOW:-0}"
SKIP_BUILD_DATASET="${SKIP_BUILD_DATASET:-0}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PRESET="${MODEL_PRESET:-gemma4_12b_it}"
RUN_ID="${RUN_ID:-20260624_${MODEL_PRESET}_hermes_fable_smoke}"
TRAIN_JSONL="${TRAIN_JSONL:-$FABLE_DIR/datasets/hermes_agent_traces_chat_20260624.jsonl}"
TRAIN_META="${TRAIN_META:-$FABLE_DIR/datasets/hermes_agent_traces_chat_20260624.meta.json}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work/.data/harness1/models}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-$FABLE_DIR/.cache/tokenized/$RUN_ID}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-2000}"
MAX_STEPS="${MAX_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-}"

case "$MODEL_PRESET" in
  gemma4_12b_it)
    MODEL_PATH="${MODEL_PATH:-google/gemma-4-12B-it}"
    MODEL_CLASS="${MODEL_CLASS:-multimodal-lm}"
    CHAT_SERIALIZATION="${CHAT_SERIALIZATION:-native}"
    CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-{\"enable_thinking\": false}}"
    ;;
  qwen35_9b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
    MODEL_CLASS="${MODEL_CLASS:-image-text-to-text}"
    CHAT_SERIALIZATION="${CHAT_SERIALIZATION:-simple-chatml}"
    DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-true}"
    CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-{}}"
    ;;
  *)
    : "${MODEL_PATH:?MODEL_PATH is required for custom MODEL_PRESET}"
    MODEL_CLASS="${MODEL_CLASS:-multimodal-lm}"
    CHAT_SERIALIZATION="${CHAT_SERIALIZATION:-native}"
    CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-{}}"
    ;;
esac

DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

SAFE_MODEL_NAME="${MODEL_PATH//\//__}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/${SAFE_MODEL_NAME}__Fabliq-Hermes-Smoke-LoRA-20260624}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

build_dataset_cmd=(
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$TRAIN_ENV/bin/python"
  "$FABLE_DIR/scripts/build_hermes_agent_traces_mix_20260624.py"
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
  --model-class "$MODEL_CLASS"
  --train-jsonl "$TRAIN_JSONL"
  --output-dir "$OUTPUT_DIR"
  --finetune-mode lora
  --max-seq-length "$MAX_SEQ_LENGTH"
  --max-train-rows "$MAX_TRAIN_ROWS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --per-device-train-batch-size "$PER_DEVICE_BATCH"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --save-steps "$SAVE_STEPS"
  --logging-steps 1
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --chat-serialization "$CHAT_SERIALIZATION"
  --ddp-find-unused-parameters "$DDP_FIND_UNUSED_PARAMETERS"
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON"
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

printf 'run_id=%s\nmodel=%s\noutput=%s\nlog_dir=%s\ntokenized_cache=%s\n' "$RUN_ID" "$MODEL_PATH" "$OUTPUT_DIR" "$LOG_DIR" "$TOKENIZED_CACHE_DIR"
print_cmd "dataset command" "${build_dataset_cmd[@]}"
print_cmd "train command" "${train_cmd[@]}"

if [[ "$RUN_NOW" == "1" ]]; then
  if [[ "$SKIP_BUILD_DATASET" == "1" ]]; then
    echo "skip dataset build: TRAIN_JSONL=$TRAIN_JSONL" | tee "$LOG_DIR/build_dataset.log"
  else
    "${build_dataset_cmd[@]}" 2>&1 | tee "$LOG_DIR/build_dataset.log"
  fi
  "${train_cmd[@]}" 2>&1 | tee "$LOG_DIR/train.log"
else
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
fi
