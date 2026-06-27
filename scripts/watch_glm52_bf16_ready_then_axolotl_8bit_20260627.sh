#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
WATCH_ENV="${WATCH_ENV:-$FABLE_DIR/.venvs/glm52-vllm-cu129-release-driver570}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/20260627_glm52_bf16_ready_then_axolotl_8bit}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
mkdir -p "$LOG_DIR"

while true; do
  if env -u PYTHONPATH PYTHONNOUSERSITE=1 "$WATCH_ENV/bin/python" \
    scripts/check_glm52_bf16_snapshot_ready_20260627.py \
    --model-id zai-org/GLM-5.2 \
    --cache-dir "$HF_HUB_CACHE" 2>&1 | tee -a "$LOG_DIR/watch.log"; then
    break
  fi
  date -u '+[%Y-%m-%dT%H:%M:%SZ] GLM-5.2 BF16 snapshot not ready; sleeping' | tee -a "$LOG_DIR/watch.log"
  tail -n 10 logs/20260627_glm52_bf16_redownload/download.log 2>/dev/null | tee -a "$LOG_DIR/watch.log" || true
  sleep "$SLEEP_SECONDS"
done

AXOLOTL_ENV="${AXOLOTL_ENV:-$FABLE_DIR/.venvs/glm52-axolotl-8bit-moe}"
while [[ ! -x "$AXOLOTL_ENV/bin/accelerate" || ! -x "$AXOLOTL_ENV/bin/axolotl" ]]; do
  date -u "+[%Y-%m-%dT%H:%M:%SZ] Axolotl env not ready at $AXOLOTL_ENV; sleeping" | tee -a "$LOG_DIR/watch.log"
  tail -n 20 logs/20260627_glm52_axolotl_setup/setup.log 2>/dev/null | tee -a "$LOG_DIR/watch.log" || true
  sleep "$SLEEP_SECONDS"
done

echo "snapshot ready; launching Axolotl 8bit MoE LoRA" | tee -a "$LOG_DIR/watch.log"
RUN_NOW=1 bash scripts/run_glm52_axolotl_8bit_moe_lora_20260627.sh
