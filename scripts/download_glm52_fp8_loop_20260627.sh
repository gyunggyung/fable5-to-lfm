#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/huggingface/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
mkdir -p logs

exec 9>logs/glm52_fp8_download_20260627.lock
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another GLM-5.2-FP8 download loop is already running"
  exit 0
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GLM-5.2-FP8 download attempt ${attempt}"
  /home/work/.local/bin/hf download zai-org/GLM-5.2-FP8 --cache-dir "${HF_HUB_CACHE}" && {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GLM-5.2-FP8 download complete"
    break
  }
  code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] download failed with exit ${code}; retrying in 60s"
  sleep 60
done
