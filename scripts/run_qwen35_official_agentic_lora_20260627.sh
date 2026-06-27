#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation"
cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/huggingface/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON_BIN="${PYTHON_BIN:-/home/work/.projects/LLM-OS-Models/Terminal/.liquid-sft-env/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/datasets/official_agentic_sft_mix_20260627.jsonl}"
RUN_ID="${RUN_ID:-20260627_qwen35_official_agentic_lora_qkvo_b2_200step}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/Qwen3.5-9B__Fable-OfficialAgentic-LoRA-qkvo-b2-200step-20260627}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-${ROOT_DIR}/.cache/tokenized/20260627_qwen35_official_agentic_lora_200step}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_ID}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-6000}"
MAX_STEPS="${MAX_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "$(dirname "${TOKENIZED_CACHE_DIR}")"

if [[ "${RUN_NOW:-0}" != "1" ]]; then
  cat <<EOF
Dry run. Set RUN_NOW=1 to launch.
RUN_ID=${RUN_ID}
MODEL_PATH=${MODEL_PATH}
DATASET_PATH=${DATASET_PATH}
OUTPUT_DIR=${OUTPUT_DIR}
TOKENIZED_CACHE_DIR=${TOKENIZED_CACHE_DIR}
LOG_DIR=${LOG_DIR}
NPROC_PER_NODE=${NPROC_PER_NODE}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS}
MAX_STEPS=${MAX_STEPS}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH}
GRAD_ACCUM=${GRAD_ACCUM}
LORA_RANK=${LORA_RANK}
TARGET_MODULES=${TARGET_MODULES}
EOF
  exit 0
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  training/train_multifamily_chat_sft.py \
  --model-path "${MODEL_PATH}" \
  --model-class causal-lm \
  --train-jsonl "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --finetune-mode lora \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --max-train-rows "${MAX_TRAIN_ROWS}" \
  --max-steps "${MAX_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --per-device-train-batch-size "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --save-steps "${SAVE_STEPS}" \
  --logging-steps 1 \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --target-modules "${TARGET_MODULES}" \
  --place-model-on-current-device-before-lora \
  --chat-serialization simple-chatml \
  --ddp-find-unused-parameters false \
  --chat-template-kwargs-json '{}' \
  --resume-from-checkpoint auto \
  --tokenized-cache-dir "${TOKENIZED_CACHE_DIR}"
