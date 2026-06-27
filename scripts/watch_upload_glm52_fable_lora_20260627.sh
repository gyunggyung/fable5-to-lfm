#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

if [[ -f "$FABLE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FABLE_DIR/.env"
  set +a
fi

SWIFT_ENV="${SWIFT_ENV:-/home/work/.cache/fable_distillation/venvs/glm52-swift-megatron}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/work/.data/harness1/models}"
OUTPUT_GLOB="${OUTPUT_GLOB:-zai-org__GLM-5.2__GLM-5.2*Fable5*LoRA*}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/zai-org__GLM-5.2__GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-20260627}"
HF_REPO_ID="${HF_REPO_ID:-LLM-OS-Models/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"
STATE_FILE="${STATE_FILE:-$FABLE_DIR/logs/20260627_glm52_fable_upload_seen.txt}"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

if [[ "${UPLOAD_CARD_ON_START:-0}" == "1" ]]; then
  "$SWIFT_ENV/bin/python" scripts/upload_glm52_fable_lora_adapter_20260627.py \
    --folder "$OUTPUT_DIR" \
    --repo-id "$HF_REPO_ID" \
    --allow-empty \
    --commit-message "Update GLM-5.2 Fable LoRA model card"
fi

while true; do
  latest_checkpoint="$(
    find "$OUTPUT_ROOT" -maxdepth 3 -path "$OUTPUT_ROOT/$OUTPUT_GLOB/*" \
      -type f \( -name adapter_model.safetensors -o -name adapter_model.bin -o -name adapter_config.json \) \
      -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
  )"
  if [[ -n "${latest_checkpoint:-}" ]] && ! grep -Fxq "$latest_checkpoint" "$STATE_FILE"; then
    "$SWIFT_ENV/bin/python" scripts/upload_glm52_fable_lora_adapter_20260627.py \
      --folder "$latest_checkpoint" \
      --repo-id "$HF_REPO_ID" \
      --commit-message "Upload GLM-5.2 Fable LoRA checkpoint $(basename "$latest_checkpoint")"
    echo "$latest_checkpoint" >> "$STATE_FILE"
  fi
  sleep "$SLEEP_SECONDS"
done
