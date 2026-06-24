#!/usr/bin/env bash
set -euo pipefail

# Start the official DiffusionGemma-capable vLLM Docker image.
# Official vLLM recipe currently points to vllm/vllm-openai:gemma because PyPI
# wheels available in this environment stop at vLLM 0.23.0, which lacks the
# DiffusionGemma model code.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-diffusiongemma-vllm-gemma}"
IMAGE="${IMAGE:-vllm/vllm-openai:gemma}"
MODEL_PATH="${MODEL_PATH:-google/diffusiongemma-26B-A4B-it}"
PORT="${PORT:-8008}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
HF_HOME_HOST="${HF_HOME_HOST:-/home/work/.data/harness1/hf_home}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/20260624_diffusiongemma_vllm_docker}"

mkdir -p "$LOG_DIR" "$HF_HOME_HOST"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --ipc=host \
  --gpus "device=$GPU_DEVICE" \
  -p "$PORT:8000" \
  -v "$HF_HOME_HOST:/root/.cache/huggingface" \
  "$IMAGE" \
    --model "$MODEL_PATH" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --generation-config vllm \
    --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}' \
    --diffusion-config '{"canvas_length":256}' \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    --host 0.0.0.0 \
    --port 8000 \
  > "$LOG_DIR/container.id"

docker logs -f "$CONTAINER_NAME" > "$LOG_DIR/server.log" 2>&1 &
echo $! > "$LOG_DIR/docker_logs.pid"
cat "$LOG_DIR/container.id"
