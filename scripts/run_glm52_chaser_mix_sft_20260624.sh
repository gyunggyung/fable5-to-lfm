#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"

RUN_ID="${RUN_ID:-20260624_glm52_chaser_mix}"
BASE_MODEL="${BASE_MODEL:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623/final_model}"
TRAIN_JSONL="${TRAIN_JSONL:-$FABLE_DIR/datasets/glm52_chaser_terminal_toolmix_20260624.jsonl}"
TRAIN_META="${TRAIN_META:-$FABLE_DIR/datasets/glm52_chaser_terminal_toolmix_20260624.meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/logs/$RUN_ID}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
EPOCHS="${EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-7}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/run.log"
}

log "run_id=$RUN_ID"
log "base_model=$BASE_MODEL"
log "output_dir=$OUTPUT_DIR"
log "train_jsonl=$TRAIN_JSONL"
log "gpus=$TRAIN_GPUS max_seq_length=$MAX_SEQ_LENGTH epochs=$EPOCHS lr=$LEARNING_RATE batch=$PER_DEVICE_BATCH accum=$GRAD_ACCUM"

log "building dataset"
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$TRAIN_ENV/bin/python" \
  "$FABLE_DIR/scripts/build_glm52_chaser_mix_20260624.py" \
  --output "$TRAIN_JSONL" \
  --meta "$TRAIN_META" \
  2>&1 | tee "$LOG_DIR/build_dataset.log"

log "starting full SFT"
env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
  TOKENIZERS_PARALLELISM=false \
  "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "$FABLE_DIR/training/train_lfm25_rlvr_json_sft.py" \
      --model-path "$BASE_MODEL" \
      --train-jsonl "$TRAIN_JSONL" \
      --output-dir "$OUTPUT_DIR" \
      --finetune-mode full \
      --max-seq-length "$MAX_SEQ_LENGTH" \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --per-device-train-batch-size "$PER_DEVICE_BATCH" \
      --gradient-accumulation-steps "$GRAD_ACCUM" \
      --save-steps "$SAVE_STEPS" \
      --save-total-limit "$SAVE_TOTAL_LIMIT" \
      --logging-steps 1 \
      --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate \
      2>&1 | tee "$LOG_DIR/train.log"

log "training complete"

if [[ "$RUN_EVAL" == "1" ]]; then
  RESULTS_DIR="$FABLE_DIR/benchmarks/${RUN_ID}_tb2_vllm/results"
  EVAL_LOG_DIR="$FABLE_DIR/benchmarks/${RUN_ID}_tb2_vllm/logs"
  mkdir -p "$RESULTS_DIR" "$EVAL_LOG_DIR"

  VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
  VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-12.9:/usr/local/cuda-12.9/include:/usr/include/x86_64-linux-gnu:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

  log "starting vLLM TB2-lite eval for glm52-chaser-mix"
  env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" \
    "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/replay_eval_vllm.py" \
      --model "$OUTPUT_DIR/final_model" \
      --tokenizer-path "$OUTPUT_DIR/final_model" \
      --model-short "glm52-chaser-mix" \
      --gpu "${EVAL_GPU:-0}" \
      --eval-path "${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}" \
      --output-dir "$RESULTS_DIR" \
      --dtype bfloat16 \
      --tp 1 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
      --max-model-len "${MAX_MODEL_LEN:-32768}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
      --max-tokens "${MAX_TOKENS:-1024}" \
      --temperature 0.0 \
      --top-p 1.0 \
      --language-model-only \
      --allow-raw-fallback \
      2>&1 | tee "$EVAL_LOG_DIR/glm52-chaser-mix.log"

  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$TRAIN_ENV/bin/python" \
    "$FABLE_DIR/scripts/summarize_replay_results.py" \
    --results-dir "$RESULTS_DIR" \
    --output-path "$RESULTS_DIR/SUMMARY.md" \
    > "$RESULTS_DIR/SUMMARY.stdout.md" || true
  log "eval summary=$RESULTS_DIR/SUMMARY.md"
fi

log "done"
