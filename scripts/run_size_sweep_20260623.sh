#!/usr/bin/env bash
set -euo pipefail
# Size sweep: LFM2.5-1.2B-Instruct (small) + LFM2-24B-A2B (XL) with Fable-5 SFT
# 밤샘 size scaling 실험

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl}"

# 1. LFM2.5-1.2B-Instruct → Fable-5 (Fabliq-1.2B-Agent)
echo "=============================================="
echo "Size sweep: LFM2.5-1.2B-Instruct → Fable-5"
echo "=============================================="

OUTPUT_DIR_12B="/home/work/.data/harness1/models/LFM2.5-1.2B-Instruct__Terminal-Fable5-FullSFT-20260623"

# 1.2B 모델은 작으니 batch size 올리고 max_seq 유지
env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    harness-1/training/train_lfm25_rlvr_json_sft.py \
      --model-path LiquidAI/LFM2.5-1.2B-Instruct \
      --train-jsonl "$TRAIN_JSONL" \
      --output-dir "$OUTPUT_DIR_12B" \
      --finetune-mode full \
      --max-seq-length 8192 \
      --epochs 3 \
      --learning-rate 2e-6 \
      --per-device-train-batch-size 4 \
      --gradient-accumulation-steps 2 \
      --save-steps 50 \
      --save-total-limit 1 \
      --logging-steps 1 \
      --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate

echo "=============================================="
echo "Size sweep: LFM2-24B-A2B → Fable-5 (Fabliq-24B-Agent)"
echo "=============================================="

OUTPUT_DIR_24B="/home/work/.data/harness1/models/LFM2-24B-A2B__Terminal-Fable5-FullSFT-20260623"

# 24B는 크니 batch 낮추고 gradient checkpointing
env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    harness-1/training/train_lfm25_rlvr_json_sft.py \
      --model-path LiquidAI/LFM2-24B-A2B \
      --train-jsonl "$TRAIN_JSONL" \
      --output-dir "$OUTPUT_DIR_24B" \
      --finetune-mode full \
      --max-seq-length 4096 \
      --epochs 1 \
      --learning-rate 5e-7 \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --save-steps 50 \
      --save-total-limit 1 \
      --logging-steps 1 \
      --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate

echo "=== Size sweep done ==="
