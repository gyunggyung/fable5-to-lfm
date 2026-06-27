#!/usr/bin/env bash
set -euo pipefail

# Wait until the BF16 GLM-5.2 snapshot is fully present in the HF cache, then
# optionally run QLoRA smoke -> pilot -> optional long run. This path is guarded
# because Transformers+BitsAndBytes does not quantize GLM's raw MoE expert
# parameters enough for local 8xH200 QLoRA in current testing.

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

if [[ -f "$FABLE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FABLE_DIR/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

WATCH_ENV="${WATCH_ENV:-$FABLE_DIR/.venvs/glm52-vllm-cu129-release-driver570}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-5.2}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
ALLOW_EXPERIMENTAL_GLM52_BF16_QLORA="${ALLOW_EXPERIMENTAL_GLM52_BF16_QLORA:-0}"
START_LONG_AFTER_PILOT="${START_LONG_AFTER_PILOT:-0}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/20260627_glm52_bf16_ready_then_qlora}"
mkdir -p "$LOG_DIR"

check_snapshot_cmd=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  HF_HOME="$HF_HOME"
  HF_HUB_CACHE="$HF_HUB_CACHE"
  "$WATCH_ENV/bin/python"
  "$FABLE_DIR/scripts/check_glm52_bf16_snapshot_ready_20260627.py"
  --model-id "$MODEL_ID"
  --cache-dir "$HF_HUB_CACHE"
)

echo "watching model=$MODEL_ID cache=$HF_HUB_CACHE sleep=${SLEEP_SECONDS}s" | tee "$LOG_DIR/watch.log"
while true; do
  if "${check_snapshot_cmd[@]}" 2>&1 | tee -a "$LOG_DIR/watch.log"; then
    break
  fi
  date -u '+[%Y-%m-%dT%H:%M:%SZ] snapshot not ready; sleeping' | tee -a "$LOG_DIR/watch.log"
  tail -n 5 "$FABLE_DIR/logs/20260627_glm52_bf16_download/download.log" 2>/dev/null | tee -a "$LOG_DIR/watch.log" || true
  sleep "$SLEEP_SECONDS"
done

if [[ "$ALLOW_EXPERIMENTAL_GLM52_BF16_QLORA" != "1" ]]; then
  {
    echo "snapshot ready, but GLM-5.2 BF16 QLoRA auto-run is disabled."
    echo "Reason: local Transformers+BitsAndBytes testing OOMed because GLM MoE expert weights are raw Parameters."
    echo "Set ALLOW_EXPERIMENTAL_GLM52_BF16_QLORA=1 only if intentionally rerunning the risky smoke."
    echo "Inspect memory with: python scripts/inspect_glm52_bf16_qlora_memory_20260627.py"
  } | tee -a "$LOG_DIR/watch.log"
  exit 0
fi

echo "snapshot ready; starting 1-step smoke" | tee -a "$LOG_DIR/watch.log"
RUN_NOW=1 \
RUN_ID=20260627_glm52_bf16_qlora_smoke \
OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-smoke-20260627 \
MAX_STEPS=1 MAX_TRAIN_ROWS=2 MAX_SEQ_LENGTH=512 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
bash "$FABLE_DIR/scripts/run_glm52_bf16_qlora_device_map_20260627.sh"

echo "smoke succeeded; starting 25-step pilot" | tee -a "$LOG_DIR/watch.log"
RUN_NOW=1 \
RUN_ID=20260627_glm52_bf16_qlora_pilot25 \
OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-pilot25-20260627 \
MAX_STEPS=25 MAX_SEQ_LENGTH=1024 SAVE_STEPS=5 SAVE_TOTAL_LIMIT=2 \
bash "$FABLE_DIR/scripts/run_glm52_bf16_qlora_device_map_20260627.sh"

if [[ "$START_LONG_AFTER_PILOT" == "1" ]]; then
  echo "pilot succeeded; starting long resumable QLoRA run" | tee -a "$LOG_DIR/watch.log"
  RUN_NOW=1 bash "$FABLE_DIR/scripts/run_glm52_bf16_qlora_device_map_20260627.sh"
else
  echo "pilot succeeded; START_LONG_AFTER_PILOT=$START_LONG_AFTER_PILOT so stopping" | tee -a "$LOG_DIR/watch.log"
fi
