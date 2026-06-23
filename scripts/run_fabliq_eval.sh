#!/usr/bin/env bash
set -euo pipefail
# Fabliq eval 파이프라인 (transformers backend, vllm ABI 깨져서 fallback)
# 한 모델에 대해: MMLU + HumanEval (lm-eval-harness transformers)

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

MODEL="${MODEL:?MODEL required (path or HF id)}"
MODEL_SHORT="${MODEL_SHORT:-$(basename "$MODEL")}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/fabliq_eval/$MODEL_SHORT}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.vllm-env}"
PYTHON="${PYTHON:-$TRAIN_ENV/bin/python}"

mkdir -p "$OUTPUT_DIR"

export NUMEXPR_MAX_THREADS=64
export TOKENIZERS_PARALLELISM=false
# Add system dist-packages for colorama, sacrebleu, etc.

# 1. MMLU (vLLM backend)
echo "=== [1/2] MMLU eval ($MODEL_SHORT) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m lm_eval \
  --model vllm \
  --model_args pretrained="$MODEL",dtype=bfloat16,max_model_len="$MAX_MODEL_LEN",gpu_memory_utilization=0.85,tensor_parallel_size=1 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "$OUTPUT_DIR/mmlu" \
  > "$OUTPUT_DIR/mmlu.log" 2>&1 || echo "MMLU failed (log: $OUTPUT_DIR/mmlu.log)"

# 2. HumanEval (vLLM backend)
echo "=== [2/2] HumanEval ($MODEL_SHORT) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m lm_eval \
  --model vllm \
  --model_args pretrained="$MODEL",dtype=bfloat16,max_model_len=4096,gpu_memory_utilization=0.85,tensor_parallel_size=1,trust_remote_code \
  --tasks humaneval \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path "$OUTPUT_DIR/humaneval" \
  > "$OUTPUT_DIR/humaneval.log" 2>&1 || echo "HumanEval failed (log: $OUTPUT_DIR/humaneval.log)"

echo ""
echo "=== Done: $MODEL_SHORT ==="
ls -la "$OUTPUT_DIR/"
