#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$FABLE_DIR"

if [[ -f "$FABLE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FABLE_DIR/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/work/.data/modelscope}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

MODEL_SLUG="${MODEL_SLUG:-GLM-5.2-Agentic-Fable5-Composer2.5-TP8-LoRA}"
RUN_ID="${RUN_ID:-20260627_glm52_swift_megatron_tp8_lora}"
SWIFT_ENV="${SWIFT_ENV:-/home/work/.cache/fable_distillation/venvs/glm52-swift-megatron}"
SOURCE_JSONL="${SOURCE_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.jsonl}"
SWIFT_JSONL="${SWIFT_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.swift_agent.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/zai-org__GLM-5.2__${MODEL_SLUG}-20260627}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
MODEL_CARD="${MODEL_CARD:-$FABLE_DIR/model_cards/GLM-5.2-Agentic-Fable5-Composer2.5-TP8-LoRA-README.md}"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")" "$MODELSCOPE_CACHE"

if [[ ! -x "$SWIFT_ENV/bin/swift" && ! -x "$SWIFT_ENV/bin/megatron" ]]; then
  echo "missing swift/megatron env: $SWIFT_ENV" >&2
  echo "run scripts/setup_glm52_swift_megatron_env_20260627.sh first" >&2
  exit 2
fi

if [[ ! -s "$SWIFT_JSONL" ]]; then
  "$SWIFT_ENV/bin/python" scripts/prepare_swift_glm52_agent_jsonl_20260627.py \
    --input "$SOURCE_JSONL" \
    --output "$SWIFT_JSONL"
fi

if [[ "${RUN_NOW:-0}" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  echo "RUN_ID=$RUN_ID"
  echo "MODEL_SLUG=$MODEL_SLUG"
  echo "SWIFT_ENV=$SWIFT_ENV"
  echo "SWIFT_JSONL=$SWIFT_JSONL"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  MODELSCOPE_CACHE="$MODELSCOPE_CACHE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  "$SWIFT_ENV/bin/megatron" sft \
    --model zai-org/GLM-5.2 \
    --save_safetensors true \
    --dataset "$SWIFT_JSONL" \
    --tensor_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --cross_entropy_loss_fusion true \
    --lr 8e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 8e-7 \
    --max_steps "${MAX_STEPS:-200}" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "${SAVE_STEPS:-25}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --template "${SWIFT_TEMPLATE:-default}" \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8 \
    --gradient_accumulation_fusion false \
    --model_author LLM-OS-Models \
    --model_name "$MODEL_SLUG" 2>&1 | tee "$LOG_DIR/train.log"
