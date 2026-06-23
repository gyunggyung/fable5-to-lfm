#!/usr/bin/env bash
set -euo pipefail
# Fabliq eval 파이프라인 (밤샘 평가용)
# 한 모델에 대해: tb2_lite terminal + MMLU + HumanEval
# vLLM 직접 로드 방식 (HTTP 서버 불필요)

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

MODEL="${MODEL:?MODEL required (path or HF id)}"
MODEL_SHORT="${MODEL_SHORT:-$(basename "$MODEL")}"
GPU="${GPU:-0,1,2,3,4,5,6,7}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/fabliq_eval/$MODEL_SHORT}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

# tb2_lite 터미널 task 평가 (LFM vLLM 전용 스크립트)
echo "=== [1/3] tb2_lite terminal eval ($MODEL_SHORT) ==="
CUDA_VISIBLE_DEVICES="$GPU" python tb2_lite/scripts/replay_eval_lfm_vllm.py \
  --model "$MODEL" \
  --model-short "$MODEL_SHORT" \
  --eval-path tb2_lite/data/replay_dev_20.jsonl \
  --output-dir "$OUTPUT_DIR/tb2_lite" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.85 \
  --max-tokens 1024 \
  --temperature 0.0 \
  --tp 1 \
  ${LIMIT:+--limit "$LIMIT"} \
  > "$OUTPUT_DIR/tb2_lite.log" 2>&1 || echo "tb2_lite failed (log: $OUTPUT_DIR/tb2_lite.log)"

# MMLU (lm-eval-harness, vLLM backend)
echo "=== [2/3] MMLU eval ($MODEL_SHORT) ==="
CUDA_VISIBLE_DEVICES="$GPU" lm_eval \
  --model vllm \
  --model_args pretrained="$MODEL",dtype=bfloat16,max_model_len="$MAX_MODEL_LEN",gpu_memory_utilization=0.85,tensor_parallel_size=1 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "$OUTPUT_DIR/mmlu" \
  > "$OUTPUT_DIR/mmlu.log" 2>&1 || echo "MMLU failed (log: $OUTPUT_DIR/mmlu.log)"

# HumanEval (lm-eval-harness, vLLM backend)
echo "=== [3/3] HumanEval ($MODEL_SHORT) ==="
CUDA_VISIBLE_DEVICES="$GPU" lm_eval \
  --model vllm \
  --model_args pretrained="$MODEL",dtype=bfloat16,max_model_len="$MAX_MODEL_LEN",gpu_memory_utilization=0.85,tensor_parallel_size=1,trust_remote_code \
  --tasks humaneval \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path "$OUTPUT_DIR/humaneval" \
  > "$OUTPUT_DIR/humaneval.log" 2>&1 || echo "HumanEval failed (log: $OUTPUT_DIR/humaneval.log)"

echo ""
echo "=== Done: $MODEL_SHORT ==="
ls -la "$OUTPUT_DIR/"
