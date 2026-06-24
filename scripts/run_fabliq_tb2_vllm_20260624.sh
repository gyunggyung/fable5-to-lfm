#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-20260624_tb2_vllm}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-$FABLE_DIR/benchmarks/$RUN_ID/results}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/benchmarks/$RUN_ID/logs}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
VLLM_PY="${VLLM_PY:-$VLLM_ENV/bin/python}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-12.9:/usr/local/cuda-12.9/include:/usr/include/x86_64-linux-gnu:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/runner.log"
}

MODELS=(
  "base|LiquidAI/LFM2.5-8B-A1B"
  "toolbench|/home/work/.data/hf_upload_stage/lfm25_8b_a1b_toolbench_full/epoch1"
  "phase1-fabliq|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623/final_model"
  "phase1b-frombase|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623/final_model"
  "phase2-reasoning|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623/final_model"
  "mega-combined|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623/final_model"
  "mega-lr2e6|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr2e6-FullSFT-20260623/final_model"
  "mega-lr5e7|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr5e7-FullSFT-20260623/final_model"
)

PIDS=()
SHORTS=()

run_one() {
  local gpu="$1"
  local short="$2"
  local model="$3"
  local log_path="$LOG_DIR/$short.log"

  if [[ -s "$RESULTS_DIR/$short.json" ]]; then
    log "skip_existing gpu=$gpu short=$short"
    return
  fi

  log "launch gpu=$gpu short=$short model=$model"
  (
    env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 \
      TOKENIZERS_PARALLELISM=false \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$VLLM_PY" "$FABLE_DIR/scripts/replay_eval_vllm.py" \
        --model "$model" \
        --tokenizer-path "$model" \
        --model-short "$short" \
        --gpu "$gpu" \
        --eval-path "$EVAL_PATH" \
        --output-dir "$RESULTS_DIR" \
        --dtype bfloat16 \
        --tp 1 \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0.0 \
        --top-p 1.0 \
        --language-model-only \
        --allow-raw-fallback \
        --skip-if-exists
  ) >"$log_path" 2>&1 &

  PIDS+=("$!")
  SHORTS+=("$short")
  printf '%s\t%s\t%s\t%s\n' "$gpu" "$!" "$short" "$model" >> "$RESULTS_DIR/pids.tsv"
}

: > "$RESULTS_DIR/pids.tsv"
log "start run_id=$RUN_ID eval_path=$EVAL_PATH results=$RESULTS_DIR"
log "vllm_env=$VLLM_ENV max_model_len=$MAX_MODEL_LEN max_tokens=$MAX_TOKENS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"

gpu=0
for spec in "${MODELS[@]}"; do
  short="${spec%%|*}"
  model="${spec#*|}"
  run_one "$gpu" "$short" "$model"
  gpu=$((gpu + 1))
done

fail=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  short="${SHORTS[$idx]}"
  if wait "$pid"; then
    log "done short=$short pid=$pid"
  else
    log "failed short=$short pid=$pid"
    fail=$((fail + 1))
  fi
done

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$ROOT_DIR/.liquid-sft-env/bin/python" \
  "$FABLE_DIR/scripts/summarize_replay_results.py" \
  --results-dir "$RESULTS_DIR" \
  --output-path "$RESULTS_DIR/SUMMARY.md" \
  > "$RESULTS_DIR/SUMMARY.stdout.md" || true

log "summary=$RESULTS_DIR/SUMMARY.md failures=$fail"
exit "$fail"
