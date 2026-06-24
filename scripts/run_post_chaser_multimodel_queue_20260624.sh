#!/usr/bin/env bash
set -euo pipefail

# Wait for the current GLM-5.2 chaser launcher to exit, then run multi-model
# jobs sequentially so GPUs do not sit idle after the long SFT/eval.
#
# Dry-run by default:
#   bash fable_distillation/scripts/run_post_chaser_multimodel_queue_20260624.sh
# Actual watcher:
#   mkdir -p fable_distillation/logs/20260624_post_chaser_multimodel_queue
#   setsid env RUN_NOW=1 bash fable_distillation/scripts/run_post_chaser_multimodel_queue_20260624.sh \
#     > fable_distillation/logs/20260624_post_chaser_multimodel_queue/nohup.log 2>&1 &

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

RUN_NOW="${RUN_NOW:-0}"
RUN_ID="${RUN_ID:-20260624_post_chaser_multimodel_queue}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
WAIT_PID_FILE="${WAIT_PID_FILE:-$FABLE_DIR/logs/20260624_glm52_chaser_mix/launcher.pid}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/queue.log"
}

wait_for_chaser() {
  if [[ ! -f "$WAIT_PID_FILE" ]]; then
    log "pid file not found: $WAIT_PID_FILE; continuing without wait"
    return
  fi
  local pid
  pid="$(tr -dc '0-9' < "$WAIT_PID_FILE")"
  if [[ -z "$pid" ]]; then
    log "pid file is empty: $WAIT_PID_FILE; continuing without wait"
    return
  fi
  while kill -0 "$pid" 2>/dev/null; do
    log "waiting for chaser launcher pid=$pid"
    sleep "$SLEEP_SECONDS"
  done
  log "chaser launcher pid=$pid has exited"
}

run_step() {
  local name="$1"
  shift
  log "starting $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  log "finished $name"
}

log "run_id=$RUN_ID"
log "wait_pid_file=$WAIT_PID_FILE"

if [[ "$RUN_NOW" != "1" ]]; then
  log "DRY-RUN. Set RUN_NOW=1 to execute after wait."
  log "planned steps: diffusiongemma base eval -> diffusiongemma smoke -> gemma4_12b_it smoke -> qwen35_9b smoke"
  exit 0
fi

wait_for_chaser

run_step diffusiongemma_26b_a4b_base_eval \
  env RUN_NOW=1 \
  bash "$FABLE_DIR/scripts/run_diffusiongemma_dllm_eval_20260624.sh"

run_step diffusiongemma_26b_a4b_smoke \
  env RUN_NOW=1 \
  bash "$FABLE_DIR/scripts/run_diffusiongemma_fable_lora_20260624.sh"

run_step gemma4_12b_it_smoke \
  env MODEL_PRESET=gemma4_12b_it RUN_NOW=1 \
  bash "$FABLE_DIR/scripts/run_multifamily_sft_smoke_20260624.sh"

run_step qwen35_9b_smoke \
  env MODEL_PRESET=qwen35_9b RUN_NOW=1 \
  bash "$FABLE_DIR/scripts/run_multifamily_sft_smoke_20260624.sh"

log "done"
