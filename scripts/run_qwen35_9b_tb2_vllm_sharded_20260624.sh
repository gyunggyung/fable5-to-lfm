#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B TB2-lite vLLM fallback.
# This keeps GPUs busy when DiffusionGemma cannot initialize through the
# Docker-free Transformers path.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

RUN_NOW="${RUN_NOW:-0}"
RUN_ID="${RUN_ID:-20260624_qwen35_9b_tb2_vllm_sharded}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
MODEL_SHORT="${MODEL_SHORT:-qwen35-9b-base-vllm}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-$FABLE_DIR/benchmarks/$RUN_ID/results}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/benchmarks/$RUN_ID/logs}"
SHARD_COUNT="${SHARD_COUNT:-8}"
SHARD_GPUS="${SHARD_GPUS:-0,1,2,3,4,5,6,7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
TB2_LIMIT="${TB2_LIMIT:-0}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

print_cmd() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

IFS=',' read -r -a shard_gpus <<< "$SHARD_GPUS"
if [[ "${#shard_gpus[@]}" -lt "$SHARD_COUNT" ]]; then
  echo "SHARD_GPUS has fewer entries than SHARD_COUNT" >&2
  exit 1
fi

limit_args=()
if [[ "$TB2_LIMIT" != "0" ]]; then
  limit_args=(--limit "$TB2_LIMIT")
fi

echo "run_id=$RUN_ID model=$MODEL_PATH tokenizer=$TOKENIZER_PATH shards=$SHARD_COUNT gpus=$SHARD_GPUS"
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_model_short="$(printf '%s.part%02d' "$MODEL_SHORT" "$shard_idx")"
  shard_cmd=(
    env -u PYTHONPATH
    PYTHONNOUSERSITE=1
    PYTHONUNBUFFERED=1
    TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="${shard_gpus[$shard_idx]}"
    "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/replay_eval_vllm.py"
    --model "$MODEL_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --model-short "$shard_model_short"
    --gpu "${shard_gpus[$shard_idx]}"
    --eval-path "$EVAL_PATH"
    --output-dir "$RESULTS_DIR"
    --dtype bfloat16
    --tp 1
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-tokens "$MAX_TOKENS"
    --temperature 0.0
    --top-p 1.0
    --language-model-only
    --allow-raw-fallback
    --shard-index "$shard_idx"
    --shard-count "$SHARD_COUNT"
    "${limit_args[@]}"
  )
  print_cmd "shard $shard_idx" "${shard_cmd[@]}"
done

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  exit 0
fi

pids=()
: > "$LOG_DIR/vllm_shards.tsv"
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_model_short="$(printf '%s.part%02d' "$MODEL_SHORT" "$shard_idx")"
  shard_log="$(printf '%s/%s.part%02d.log' "$LOG_DIR" "$MODEL_SHORT" "$shard_idx")"
  env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${shard_gpus[$shard_idx]}" \
    "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/replay_eval_vllm.py" \
      --model "$MODEL_PATH" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --model-short "$shard_model_short" \
      --gpu "${shard_gpus[$shard_idx]}" \
      --eval-path "$EVAL_PATH" \
      --output-dir "$RESULTS_DIR" \
      --dtype bfloat16 \
      --tp 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-tokens "$MAX_TOKENS" \
      --temperature 0.0 \
      --top-p 1.0 \
      --language-model-only \
      --allow-raw-fallback \
      --shard-index "$shard_idx" \
      --shard-count "$SHARD_COUNT" \
      "${limit_args[@]}" \
    > "$shard_log" 2>&1 &
  pids+=("$!")
  echo -e "$shard_idx\t${shard_gpus[$shard_idx]}\t${pids[-1]}\t$shard_log" | tee -a "$LOG_DIR/vllm_shards.tsv"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "one or more Qwen vLLM shards failed" >&2
  exit 1
fi

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python" \
  "$FABLE_DIR/scripts/merge_diffusiongemma_transformers_shards.py" \
  --input-dir "$RESULTS_DIR" \
  --glob "${MODEL_SHORT}.part*.json" \
  --output-path "$RESULTS_DIR/${MODEL_SHORT}.json" \
  --model-short "$MODEL_SHORT" \
  2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.merge.log"
