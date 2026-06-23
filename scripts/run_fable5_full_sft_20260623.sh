#!/usr/bin/env bash
set -euo pipefail
# Fable-5 agentic distill → LFM2.5-8B-A1B Terminal-ToolBench-Full-SFT-1Epoch 위에 full SFT
# 데이터: fable5_lfm_sft_20260623.jsonl (4,047 rows, 약 10M LFM tokens)
# 베이스: ToolBench-Full-SFT-1Epoch (16.9GB full model)
# H200 8대, FSDP full_shard + activation_checkpointing, bf16

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
RUN_NOW="${RUN_NOW:-0}"

BASE_MODEL="${BASE_MODEL:-/home/work/.data/hf_upload_stage/lfm25_8b_a1b_toolbench_full/epoch1}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

run_or_print() {
  local name="$1"; shift
  echo; echo "### $name"; printf '%q ' "$@"; echo
  if [[ "$RUN_NOW" == "1" ]]; then "$@"; fi
}

# 기존 ToolBench 학습 config (run_config.json 기준) 준용:
#   max_seq=4096 → 8192 (Fable traces 길이 때문)
#   epochs=0.25 → 3 (데이터 작음)
#   LR=8e-7 → 5e-7 (이어받으니 낮춤, forgetting 방지)
#   batch=1, grad_accum=4 → batch=2, grad_accum=4 (8192 seq에 H200 141GB면 가능)
run_or_print "train LFM2.5-8B Fable-5 agentic full SFT (8 GPU H200)" \
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
        --learning-rate 5e-7 \
        --per-device-train-batch-size 2 \
        --gradient-accumulation-steps 4 \
        --save-steps 50 \
        --save-total-limit 2 \
        --logging-steps 1 \
        --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
