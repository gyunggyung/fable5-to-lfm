#!/usr/bin/env bash
set -euo pipefail
# Mega ablation: 다양한 epoch / LR 조합 실험
# - 1 epoch, 5 epoch, 10 epoch (epoch scan)
# - LR 5e-7, 2e-6 (LR scan)

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/fable_distillation/datasets/mega_combined_lfm_sft_20260623.jsonl}"

run_mega() {
  local NAME="$1" EPOCHS="$2" LR="$3"
  local OUTPUT_DIR="/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-${NAME}-FullSFT-20260623"

  echo "=============================================="
  echo "Mega ${NAME}: epochs=${EPOCHS}, LR=${LR}"
  echo "=============================================="

  env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 \
      harness-1/training/train_lfm25_rlvr_json_sft.py \
        --model-path "$BASE_MODEL" \
        --train-jsonl "$TRAIN_JSONL" \
        --output-dir "$OUTPUT_DIR" \
        --finetune-mode full \
        --max-seq-length 8192 \
        --epochs "$EPOCHS" \
        --learning-rate "$LR" \
        --per-device-train-batch-size 2 \
        --gradient-accumulation-steps 4 \
        --save-steps 100 \
        --save-total-limit 1 \
        --logging-steps 1 \
        --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
}

# Epoch scan
run_mega "1ep" 1 1e-6
run_mega "5ep" 5 1e-6
run_mega "10ep" 10 5e-7  # 10 epoch는 LR 낮춰 overfitting 방지

# LR scan (3 epoch 기준)
run_mega "lr5e7" 3 5e-7
run_mega "lr2e6" 3 2e-6

echo "=== All Mega ablations done ==="
