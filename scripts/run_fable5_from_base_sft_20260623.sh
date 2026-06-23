#!/usr/bin/env bash
set -euo pipefail
# Phase-1B: Raw LFM2.5-8B-A1B → Fable-5 agentic SFT (ToolBench 생략 경로)
# 목적: ToolBench foundation 효과를 ablation 하기 위해 base 모델에서 Fable-5 바로 SFT
# 비교: Fabliq-8B-Agent (ToolBench → Fable-5) vs Fabliq-8B-Agent-FromBase (이 스크립트 결과)

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
RUN_NOW="${RUN_NOW:-0}"

# raw LiquidAI base
BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

run_or_print() {
  local name="$1"; shift
  echo; echo "### $name"; printf '%q ' "$@"; echo
  if [[ "$RUN_NOW" == "1" ]]; then "$@"; fi
}

# base에서 직접 SFT → epochs 3, LR 1e-6 (ToolBench가 이미 정제 안 됐으니 조금 더 높게)
# Phase-1과 동일한 8192 / batch 2 / accum 4 / 8GPU = global 64
run_or_print "train LFM2.5-8B-A1B base → Fable-5 SFT (8 GPU H200)" \
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
        --epochs 3 \
        --learning-rate 1e-6 \
        --per-device-train-batch-size 2 \
        --gradient-accumulation-steps 4 \
        --save-steps 50 \
        --save-total-limit 2 \
        --logging-steps 1 \
        --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
