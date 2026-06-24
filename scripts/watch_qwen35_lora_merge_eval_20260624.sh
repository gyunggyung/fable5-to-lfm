#!/usr/bin/env bash
set -euo pipefail

# Wait for the current Qwen3.5 LoRA SFT run, merge the final adapter, then
# launch the TB2-lite vLLM sharded evaluation. Run from fable_distillation.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

RUN_NOW="${RUN_NOW:-0}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
MODEL_CLASS="${MODEL_CLASS:-image-text-to-text}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-20260624_qwen35_9b_glm52_terminalmix_lora_sft300_chatml_ddptrue}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-20260624}"
ADAPTER_DIR="${ADAPTER_DIR:-$TRAIN_OUTPUT_DIR/final_lora}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-Merged-20260624}"
MERGE_GPU="${MERGE_GPU:-0}"
MERGE_DEVICE_MAP="${MERGE_DEVICE_MAP:-auto}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-72000}"
EVAL_RUN_ID="${EVAL_RUN_ID:-20260624_qwen35_9b_glm52_terminalmix_lora_sft300_chatml_ddptrue_vllm}"
MODEL_SHORT="${MODEL_SHORT:-qwen35-9b-glm52-terminalmix-lora-sft300-chatml-vllm}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/${TRAIN_RUN_ID}_post_eval}"

mkdir -p "$LOG_DIR"

print_cmd() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

merge_cmd=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  CUDA_VISIBLE_DEVICES="$MERGE_GPU"
  "$TRAIN_ENV/bin/python"
  "$FABLE_DIR/scripts/merge_multifamily_lora_for_vllm.py"
  --base-model "$BASE_MODEL"
  --adapter-dir "$ADAPTER_DIR"
  --output-dir "$MERGED_MODEL_DIR"
  --model-class "$MODEL_CLASS"
  --tokenizer-path "$ADAPTER_DIR"
  --dtype "$MERGE_DTYPE"
  --device-map "$MERGE_DEVICE_MAP"
  --max-shard-size "$MAX_SHARD_SIZE"
  --overwrite
)

eval_cmd=(
  env
  RUN_NOW=1
  RUN_ID="$EVAL_RUN_ID"
  MODEL_PATH="$MERGED_MODEL_DIR"
  TOKENIZER_PATH="$MERGED_MODEL_DIR"
  MODEL_SHORT="$MODEL_SHORT"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
  TB2_LIMIT="${TB2_LIMIT:-0}"
  bash
  "$FABLE_DIR/scripts/run_qwen35_9b_tb2_vllm_sharded_20260624.sh"
)

printf 'train_run_id=%s\nadapter_dir=%s\nmerged_model_dir=%s\neval_run_id=%s\n' \
  "$TRAIN_RUN_ID" "$ADAPTER_DIR" "$MERGED_MODEL_DIR" "$EVAL_RUN_ID"
print_cmd "merge command" "${merge_cmd[@]}"
print_cmd "eval command" "${eval_cmd[@]}"

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  exit 0
fi

start_ts="$(date +%s)"
while [[ ! -f "$ADAPTER_DIR/adapter_config.json" || ! -f "$TRAIN_OUTPUT_DIR/run_config.json" ]]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed > MAX_WAIT_SECONDS )); then
    echo "timed out waiting for adapter: $ADAPTER_DIR" >&2
    exit 2
  fi
  if ! pgrep -af "$TRAIN_OUTPUT_DIR" >/dev/null; then
    echo "training process is not running and final adapter is not complete: $ADAPTER_DIR" >&2
    exit 3
  fi
  date -u '+%Y-%m-%dT%H:%M:%SZ waiting for final_lora'
  sleep "$POLL_SECONDS"
done

if [[ ! -f "$MERGED_MODEL_DIR/merge_manifest.json" ]]; then
  "${merge_cmd[@]}" 2>&1 | tee "$LOG_DIR/merge.log"
else
  echo "skip merge; manifest exists: $MERGED_MODEL_DIR/merge_manifest.json" | tee "$LOG_DIR/merge.log"
fi

"${eval_cmd[@]}" 2>&1 | tee "$LOG_DIR/eval.log"
