#!/usr/bin/env bash
set -euo pipefail
# Phase-2M: Mega → + reasoning expansion
# Mega (raw base → 모든 데이터 4,328 rows) 모델에 WithinUs+Helio reasoning 추가 학습
# 이미 Mega에 WithinUs+Helio가 포함되어 있지만, reasoning 강화를 위해 추가 epoch 돌림

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
RUN_NOW="${RUN_NOW:-0}"

BASE_MODEL="${BASE_MODEL:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623/final_model}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/fable_distillation/datasets/phase2_reasoning_lfm_sft_20260623.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Phase2M-Reasoning-FullSFT-20260623}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

run_or_print() {
  local name="$1"; shift
  echo; echo "### $name"; printf '%q ' "$@"; echo
  if [[ "$RUN_NOW" == "1" ]]; then "$@"; fi
}

# Mega 모델 이미 4,328 rows 학습 (WithinUs 135 + Helio 146 포함).
# Phase-2M은 reasoning 부분만 다시 한 번 강조해서 추가 학습 (2 epoch, 낮은 LR).
run_or_print "train LFM2.5-8B Phase-2M Mega reasoning (8 GPU H200)" \
  env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
    "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
      harness-1/training/train_lfm25_rlvr_json_sft.py \
        --model-path "$BASE_MODEL" \
        --train-jsonl "$TRAIN_JSONL" \
        --output-dir "$OUTPUT_DIR" \
        --finetune-mode full \
        --max-seq-length 8192 \
        --epochs 2 \
        --learning-rate 2e-7 \
        --per-device-train-batch-size 2 \
        --gradient-accumulation-steps 4 \
        --save-steps 5 \
        --save-total-limit 2 \
        --logging-steps 1 \
        --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
