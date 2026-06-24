#!/usr/bin/env bash
set -euo pipefail

# DiffusionGemma strength-task LoRA runner.
# Dry-run by default:
#   bash fable_distillation/scripts/run_diffusiongemma_strength_lora_20260624.sh
# Actual run after GPUs are free:
#   RUN_NOW=1 bash fable_distillation/scripts/run_diffusiongemma_strength_lora_20260624.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

RUN_NOW="${RUN_NOW:-0}"
SKIP_BUILD_DATASET="${SKIP_BUILD_DATASET:-0}"
RUN_ID="${RUN_ID:-20260624_diffusiongemma_strength_lora_fable_structured_repair_nooptim}"
TRAIN_ENV="${TRAIN_ENV:-/home/work/.data/harness1/envs/diffusiongemma-nemo}"
DG_WORK_DIR="${DG_WORK_DIR:-/home/work/.data/harness1/diffusiongemma_retriever}"
AUTOMODEL_DIR="${AUTOMODEL_DIR:-$DG_WORK_DIR/Automodel}"
CONFIG_PATH="${CONFIG_PATH:-$FABLE_DIR/configs/diffusiongemma_26b_a4b_strength_lora_20260624.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$FABLE_DIR/scripts/diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py}"
TRAIN_JSONL="${TRAIN_JSONL:-$FABLE_DIR/datasets/diffusiongemma_strength_mix_20260624.jsonl}"
TRAIN_META="${TRAIN_META:-$FABLE_DIR/datasets/diffusiongemma_strength_mix_20260624.meta.json}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"

mkdir -p "$LOG_DIR"

build_dataset_cmd=(
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$TRAIN_ENV/bin/python"
  "$FABLE_DIR/scripts/build_diffusiongemma_strength_mix_20260624.py"
  --output "$TRAIN_JSONL"
  --meta "$TRAIN_META"
)

train_cmd=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  HF_HOME="${HF_HOME:-/home/work/.data/harness1/hf_home}"
  HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/harness1/hf_home/hub}"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  "$TRAIN_ENV/bin/python" -m torch.distributed.run
  --standalone --nproc-per-node="$NPROC_PER_NODE"
  "$TRAIN_SCRIPT"
  -c "$CONFIG_PATH"
)

print_cmd() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

printf 'run_id=%s\nconfig=%s\ntrain_script=%s\ntrain_jsonl=%s\nlog_dir=%s\n' "$RUN_ID" "$CONFIG_PATH" "$TRAIN_SCRIPT" "$TRAIN_JSONL" "$LOG_DIR"
print_cmd "dataset command" "${build_dataset_cmd[@]}"
print_cmd "train command" "${train_cmd[@]}"

if [[ "$RUN_NOW" == "1" ]]; then
  if [[ "$SKIP_BUILD_DATASET" == "1" ]]; then
    echo "skip dataset build: TRAIN_JSONL=$TRAIN_JSONL" | tee "$LOG_DIR/build_dataset.log"
  else
    "${build_dataset_cmd[@]}" 2>&1 | tee "$LOG_DIR/build_dataset.log"
  fi
  "${train_cmd[@]}" 2>&1 | tee "$LOG_DIR/train.log"
else
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
fi
