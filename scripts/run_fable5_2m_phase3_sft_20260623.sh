#!/usr/bin/env bash
set -euo pipefail
# Phase-3: 2M traces scale-out agentic SFT
# Phase-1B(FromBase) → + 2M traces 3,866 rows × 1 epoch
# 목적: agentic trajectory 확대로 phase-1(4K) 대비 일반화 성능 향상

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
RUN_NOW="${RUN_NOW:-0}"

BASE_MODEL="${BASE_MODEL:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623/final_model}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/fable_distillation/datasets/fable5_2m_lfm_sft_20260623.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-2M-Phase3-FullSFT-20260623}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

run_or_print() {
  local name="$1"; shift
  echo; echo "### $name"; printf '%q ' "$@"; echo
  if [[ "$RUN_NOW" == "1" ]]; then "$@"; fi
}

# 2M 데이터 3,866 rows × 1 epoch (이미 Phase-1B에서 4,047 rows 학습했으므로 추가분만)
# LR 5e-7 (Phase-1B보다 약간 높게, 새 데이터 적응 위해)
run_or_print "train LFM2.5-8B Phase-3 2M agentic scale-out (8 GPU H200)" \
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
        --learning-rate 5e-7 \
        --per-device-train-batch-size 2 \
        --gradient-accumulation-steps 4 \
        --save-steps 30 \
        --save-total-limit 2 \
        --logging-steps 1 \
        --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
