#!/usr/bin/env bash
# Mega 5ep - 5 epoch ablation
set -euo pipefail
cd /home/work/.projects/LLM-OS-Models/Terminal
OUTPUT_DIR=/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-5ep-FullSFT-20260623 \
  env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  .liquid-sft-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
    harness-1/training/train_lfm25_rlvr_json_sft.py \
    --model-path LiquidAI/LFM2.5-8B-A1B \
    --train-jsonl fable_distillation/datasets/mega_combined_lfm_sft_20260623.jsonl \
    --output-dir /home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-5ep-FullSFT-20260623 \
    --finetune-mode full --max-seq-length 8192 --epochs 5 --learning-rate 1e-6 \
    --per-device-train-batch-size 2 --gradient-accumulation-steps 4 \
    --save-steps 100 --save-total-limit 1 --logging-steps 1 \
    --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
