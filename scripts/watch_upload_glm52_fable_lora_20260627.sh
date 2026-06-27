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
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2-Agentic-Fable5-Composer2.5-TP8-LoRA-20260627}"
HF_REPO_ID="${HF_REPO_ID:-LLM-OS-Models/GLM-5.2-Agentic-Fable5-Composer2.5-TP8-LoRA}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"
STATE_FILE="${STATE_FILE:-$FABLE_DIR/logs/20260627_glm52_fable_upload_seen.txt}"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

while true; do
  latest_checkpoint="$(find "$OUTPUT_DIR" -type f \( -name adapter_model.safetensors -o -name adapter_model.bin -o -name adapter_config.json \) -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
  if [[ -n "${latest_checkpoint:-}" ]] && ! grep -Fxq "$latest_checkpoint" "$STATE_FILE"; then
    "$SWIFT_ENV/bin/python" scripts/upload_glm52_fable_lora_adapter_20260627.py \
      --folder "$latest_checkpoint" \
      --repo-id "$HF_REPO_ID" \
      --commit-message "Upload GLM-5.2 Fable LoRA checkpoint $(basename "$latest_checkpoint")"
    echo "$latest_checkpoint" >> "$STATE_FILE"
  fi
  sleep "$SLEEP_SECONDS"
done
