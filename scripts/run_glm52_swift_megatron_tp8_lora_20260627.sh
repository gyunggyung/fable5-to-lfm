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
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-2}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-4}"

MODEL_SLUG="${MODEL_SLUG:-GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA}"
RUN_ID="${RUN_ID:-20260627_glm52_swift_megatron_tp8_lora}"
SWIFT_ENV="${SWIFT_ENV:-/home/work/.cache/fable_distillation/venvs/glm52-swift-megatron}"
SWIFT_SITE="$SWIFT_ENV/lib/python3.12/site-packages"
GLM52_MODEL="${GLM52_MODEL:-/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
QUANT_METHOD="${QUANT_METHOD:-fp8}"
QUANT_BITS="${QUANT_BITS:-float8}"
FP8_FORMAT="${FP8_FORMAT:-hybrid}"
FP8_PARAM_GATHER="${FP8_PARAM_GATHER:-true}"
USE_CPU_INITIALIZATION="${USE_CPU_INITIALIZATION:-false}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-false}"
OFFLOAD_BRIDGE="${OFFLOAD_BRIDGE:-false}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-true}"
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-true}"
MOE_SHARED_EXPERT_OVERLAP="${MOE_SHARED_EXPERT_OVERLAP:-true}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGET_MODULES="${TARGET_MODULES:-q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj o_proj}"
LR="${LR:-8e-6}"
MIN_LR="${MIN_LR:-8e-7}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
SOURCE_JSONL="${SOURCE_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.jsonl}"
SWIFT_JSONL="${SWIFT_JSONL:-$FABLE_DIR/datasets/official_agentic_sft_mix_20260627.swift_agent.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/zai-org__GLM-5.2__${MODEL_SLUG}-20260627}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
MODEL_CARD="${MODEL_CARD:-$FABLE_DIR/model_cards/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-README.md}"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")" "$MODELSCOPE_CACHE"
export LD_LIBRARY_PATH="$SWIFT_SITE/nvidia/cublas/lib:$SWIFT_SITE/nvidia/cudnn/lib:$SWIFT_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

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
  echo "GLM52_MODEL=$GLM52_MODEL"
  echo "SWIFT_ENV=$SWIFT_ENV"
  echo "SWIFT_JSONL=$SWIFT_JSONL"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "PARALLELISM=TP${TENSOR_MODEL_PARALLEL_SIZE}_PP${PIPELINE_MODEL_PARALLEL_SIZE}_EP${EXPERT_MODEL_PARALLEL_SIZE}"
  echo "TARGET_MODULES=$TARGET_MODULES"
  echo "OFFLOAD_MODEL=$OFFLOAD_MODEL OFFLOAD_BRIDGE=$OFFLOAD_BRIDGE OPTIMIZER_CPU_OFFLOAD=$OPTIMIZER_CPU_OFFLOAD USE_CPU_INITIALIZATION=$USE_CPU_INITIALIZATION"
  exit 0
fi

read -r -a TARGET_MODULES_ARR <<< "$TARGET_MODULES"

env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  MODELSCOPE_CACHE="$MODELSCOPE_CACHE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  "$SWIFT_ENV/bin/megatron" sft \
    --model "$GLM52_MODEL" \
    --use_hf true \
    --torch_dtype "$TORCH_DTYPE" \
    --quant_method "$QUANT_METHOD" \
    --quant_bits "$QUANT_BITS" \
    --fp8_format "$FP8_FORMAT" \
    --fp8_param_gather "$FP8_PARAM_GATHER" \
    --use_cpu_initialization "$USE_CPU_INITIALIZATION" \
    --offload_model "$OFFLOAD_MODEL" \
    --offload_bridge "$OFFLOAD_BRIDGE" \
    --optimizer_cpu_offload "$OPTIMIZER_CPU_OFFLOAD" \
    --save_safetensors true \
    --merge_lora false \
    --dataset "$SWIFT_JSONL" \
    --tensor_model_parallel_size "$TENSOR_MODEL_PARALLEL_SIZE" \
    --pipeline_model_parallel_size "$PIPELINE_MODEL_PARALLEL_SIZE" \
    --expert_model_parallel_size "$EXPERT_MODEL_PARALLEL_SIZE" \
    --sequence_parallel true \
    --moe_permute_fusion "$MOE_PERMUTE_FUSION" \
    --moe_grouped_gemm "$MOE_GROUPED_GEMM" \
    --moe_shared_expert_overlap "$MOE_SHARED_EXPERT_OVERLAP" \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --tuner_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "${TARGET_MODULES_ARR[@]}" \
    --cross_entropy_loss_fusion true \
    --lr "$LR" \
    --lr_warmup_fraction 0.05 \
    --min_lr "$MIN_LR" \
    --train_iters "${MAX_STEPS:-200}" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "${SAVE_STEPS:-25}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --template "${SWIFT_TEMPLATE:-glm5_2}" \
    --agent_template "${SWIFT_AGENT_TEMPLATE:-glm5_1}" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --gradient_accumulation_fusion false \
    --model_author LLM-OS-Models \
    --model_name "$MODEL_SLUG" 2>&1 | tee "$LOG_DIR/train.log"
