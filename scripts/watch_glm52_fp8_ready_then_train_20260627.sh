#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

HF_BIN="${HF_BIN:-/home/work/.local/bin/hf}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-5.2-FP8}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/20260627_glm52_fp8_ready_then_train}"
DOWNLOAD_LOG="$LOG_DIR/download.log"
SETUP_LOG="$LOG_DIR/setup_env.log"
SMOKE_LOG_DIR="$FABLE_DIR/logs/20260627_glm52_fp8_smoke_lora"
FULL_RUN_ID="${FULL_RUN_ID:-20260627_glm52_fp8_fable_official_agentic_lora}"

RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"
RUN_FULL_TRAIN="${RUN_FULL_TRAIN:-1}"
STOP_OTHER_GPU_JOBS_ON_READY="${STOP_OTHER_GPU_JOBS_ON_READY:-1}"
DOWNLOAD_RETRY_SLEEP="${DOWNLOAD_RETRY_SLEEP:-60}"

mkdir -p "$LOG_DIR" "$HF_HUB_CACHE"
exec 9>"$LOG_DIR/watch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher already running"
  exit 0
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

stop_other_gpu_jobs() {
  local pids
  pids="$(ps -eo pid=,args= | awk '/[t]rain_multifamily_chat_sft\\.py|[t]orch\\.distributed\\.run/ && !/GLM-5\\.2-FP8/ {print $1}')"
  if [[ -n "$pids" ]]; then
    log "stopping non-GLM GPU jobs: $pids"
    kill $pids || true
    sleep 10
    pids="$(ps -eo pid=,args= | awk '/[t]rain_multifamily_chat_sft\\.py|[t]orch\\.distributed\\.run/ && !/GLM-5\\.2-FP8/ {print $1}')"
    if [[ -n "$pids" ]]; then
      log "force stopping non-GLM GPU jobs: $pids"
      kill -9 $pids || true
    fi
  fi
}

log "starting env setup in background: TRAIN_ENV=$TRAIN_ENV"
(
  set -uo pipefail
  TRAIN_ENV="$TRAIN_ENV" "$FABLE_DIR/scripts/setup_glm52_lora_env_20260627.sh"
) >"$SETUP_LOG" 2>&1 &
SETUP_PID=$!
log "env setup pid=$SETUP_PID log=$SETUP_LOG"

attempt=0
while true; do
  attempt=$((attempt + 1))
  {
    log "download attempt $attempt model=$MODEL_ID cache=$HF_HUB_CACHE"
    df -h /home/work/.data || true
    du -sh "$HF_HUB_CACHE/models--zai-org--GLM-5.2-FP8" 2>/dev/null || true
  } >>"$DOWNLOAD_LOG" 2>&1

  "$HF_BIN" download "$MODEL_ID" --cache-dir "$HF_HUB_CACHE" >>"$DOWNLOAD_LOG" 2>&1
  code=$?
  if [[ "$code" == "0" ]]; then
    log "download complete for $MODEL_ID" | tee -a "$DOWNLOAD_LOG"
    break
  fi
  log "download failed with exit $code; retrying in ${DOWNLOAD_RETRY_SLEEP}s" | tee -a "$DOWNLOAD_LOG"
  sleep "$DOWNLOAD_RETRY_SLEEP"
done

log "waiting for env setup pid=$SETUP_PID"
wait "$SETUP_PID"
setup_code=$?
if [[ "$setup_code" != "0" ]]; then
  log "env setup failed with exit $setup_code; see $SETUP_LOG"
  exit "$setup_code"
fi
log "env setup complete"

if [[ "$STOP_OTHER_GPU_JOBS_ON_READY" == "1" ]]; then
  stop_other_gpu_jobs
fi

if [[ "$RUN_SMOKE_FIRST" == "1" ]]; then
  log "starting GLM smoke run"
  RUN_ID=20260627_glm52_fp8_smoke_lora \
  TRAIN_ENV="$TRAIN_ENV" \
  SKIP_BUILD_DATASET=1 \
  RUN_NOW=1 \
  MAX_STEPS=1 \
  MAX_TRAIN_ROWS=16 \
  MAX_SEQ_LENGTH=2048 \
  SAVE_STEPS=1 \
  SAVE_TOTAL_LIMIT=1 \
  OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2-FP8__Fable-OfficialAgentic-LoRA-smoke-20260627 \
  "$FABLE_DIR/scripts/run_glm52_official_agentic_lora_20260627.sh" >"$SMOKE_LOG_DIR.train.log" 2>&1
  smoke_code=$?
  if [[ "$smoke_code" != "0" ]]; then
    log "GLM smoke failed with exit $smoke_code; see $SMOKE_LOG_DIR.train.log"
    exit "$smoke_code"
  fi
  log "GLM smoke complete"
fi

if [[ "$RUN_FULL_TRAIN" == "1" ]]; then
  log "starting full GLM LoRA train run_id=$FULL_RUN_ID"
  RUN_ID="$FULL_RUN_ID" \
  TRAIN_ENV="$TRAIN_ENV" \
  SKIP_BUILD_DATASET=1 \
  RUN_NOW=1 \
  "$FABLE_DIR/scripts/run_glm52_official_agentic_lora_20260627.sh"
else
  log "RUN_FULL_TRAIN=$RUN_FULL_TRAIN; not starting full train"
fi
