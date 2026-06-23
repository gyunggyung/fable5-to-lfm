#!/usr/bin/env bash
# 남은 밤샘 ablation 자동 진행 - Mega 5ep 완료 후 실행
# Mega 10ep → Mega lr5e-7 → Mega lr2e-6 → LFM2.5-1.2B small → LFM2-24B-A2B large

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
MEGA_DATA="$ROOT_DIR/fable_distillation/datasets/mega_combined_lfm_sft_20260623.jsonl"
FABLE5_DATA="$ROOT_DIR/fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl"

wait_for_free_gpu() {
  # GPU 0이 free 될 때까지 대기
  echo "Waiting for GPU 0 to be free..."
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "$used" -lt 5000 ]]; then
      break
    fi
    sleep 30
  done
}

run_one() {
  local NAME="$1" BASE="$2" DATA="$3" EPOCHS="$4" LR="$5" SEQ="$6" BATCH="$7" ACCUM="$8"
  local OUTPUT_DIR="/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-${NAME}-FullSFT-20260623"

  if [[ -e "$OUTPUT_DIR/final_model/model.safetensors" ]]; then
    echo "SKIP $NAME (already done)"
    return
  fi

  echo "=============================================="
  echo "Training $NAME: epochs=$EPOCHS, LR=$LR, seq=$SEQ"
  echo "=============================================="

  env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 \
      harness-1/training/train_lfm25_rlvr_json_sft.py \
      --model-path "$BASE" \
      --train-jsonl "$DATA" \
      --output-dir "$OUTPUT_DIR" \
      --finetune-mode full \
      --max-seq-length "$SEQ" \
      --epochs "$EPOCHS" \
      --learning-rate "$LR" \
      --per-device-train-batch-size "$BATCH" \
      --gradient-accumulation-steps "$ACCUM" \
      --save-steps 100 \
      --save-total-limit 1 \
      --logging-steps 1 \
      --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
}

# Mega 10ep
run_one "Mega-10ep" "LiquidAI/LFM2.5-8B-A1B" "$MEGA_DATA" 10 5e-7 8192 2 4

# Mega lr5e-7
run_one "Mega-lr5e7" "LiquidAI/LFM2.5-8B-A1B" "$MEGA_DATA" 3 5e-7 8192 2 4

# Mega lr2e-6
run_one "Mega-lr2e6" "LiquidAI/LFM2.5-8B-A1B" "$MEGA_DATA" 3 2e-6 8192 2 4

# LFM2.5-1.2B-Instruct small variant
run_one "Fable5-1.2B" "LiquidAI/LFM2.5-1.2B-Instruct" "$FABLE5_DATA" 3 2e-6 8192 4 2

# LFM2-24B-A2B large variant (max_seq 4096 for VRAM)
run_one "Fable5-24B" "LiquidAI/LFM2-24B-A2B" "$FABLE5_DATA" 1 5e-7 4096 1 8

echo "=== All remaining ablations done ==="
