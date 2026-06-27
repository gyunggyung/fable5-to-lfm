#!/usr/bin/env bash
set -euo pipefail

# Launch GLM-5.2-FP8 through the official vLLM-style path without Docker.
#
# Dry-run:
#   bash fable_distillation/scripts/run_glm52_fp8_vllm_server_20260627.sh
#
# Actual server:
#   RUN_NOW=1 bash fable_distillation/scripts/run_glm52_fp8_vllm_server_20260627.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

RUN_NOW="${RUN_NOW:-0}"
ENV_DIR="${ENV_DIR:-$FABLE_DIR/.venvs/glm52-vllm-cu128}"
MODEL_PATH="${MODEL_PATH:-zai-org/GLM-5.2-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2-fp8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/20260627_glm52_fp8_vllm_server}"

mkdir -p "$LOG_DIR"

cmd=(
  env
  -u PYTHONPATH
  PYTHONNOUSERSITE=1
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  HF_HOME="$HF_HOME"
  HF_HUB_CACHE="$HF_HUB_CACHE"
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE"
  VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
  "$ENV_DIR/bin/vllm" serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --kv-cache-dtype fp8
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --speculative-config.method mtp
  --speculative-config.num_speculative_tokens 5
  --tool-call-parser glm47
  --reasoning-parser glm45
  --enable-auto-tool-choice
)

printf 'model=%s\nserved_model_name=%s\nhost=%s\nport=%s\n' \
  "$MODEL_PATH" "$SERVED_MODEL_NAME" "$HOST" "$PORT"
printf 'env=%s\ngpus=%s tp=%s max_model_len=%s max_num_seqs=%s gpu_mem_util=%s\n' \
  "$ENV_DIR" "$CUDA_VISIBLE_DEVICES" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$GPU_MEMORY_UTILIZATION"
printf 'command:\n  '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to launch the vLLM server."
  exit 0
fi

if [[ ! -x "$ENV_DIR/bin/vllm" ]]; then
  echo "missing vLLM env: $ENV_DIR/bin/vllm" >&2
  echo "Run scripts/setup_glm52_vllm_uv_20260627.sh first." >&2
  exit 2
fi

"${cmd[@]}" 2>&1 | tee "$LOG_DIR/server.log"
