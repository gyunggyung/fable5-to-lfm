#!/usr/bin/env bash
set -euo pipefail
# Fabliq-8B-Agent → GGUF 변환 (Phase-1 모델)
# llama.cpp-mtp 빌드 사용: /home/work/.cache/llama.cpp-mtp/build/bin/
# GPU 불필요 (CPU 변환) → 학습과 병렬 실행

LLAMA_DIR="/home/work/.cache/llama.cpp-mtp"
CONVERT="${LLAMA_DIR}/convert_hf_to_gguf.py"
QUANTIZE="${LLAMA_DIR}/build/bin/llama-quantize"
LLAMA_PYTHON="${LLAMA_DIR}-env/bin/python"  # 가상환환 (필요시 수정)

# Phase-1 모델 (Fabliq-8B-Agent 원본)
HF_MODEL="/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623/final_model"
OUT_DIR="/home/work/.data/gguf/fabliq-8b-agent"
mkdir -p "$OUT_DIR"

cd "$LLAMA_DIR"

# 1. F16 (비양자화) GGUF 생성
echo "=== Converting HF → F16 GGUF ==="
python3 "$CONVERT" "$HF_MODEL" --outtype f16 --outfile "$OUT_DIR/fabliq-8b-agent.F16.gguf" 2>&1 | tail -20

# 2. 양자화 버전들 생성
echo "=== Quantizing to Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M ==="
for QUANT in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M; do
  echo "--- $QUANT ---"
  "$QUANTIZE" "$OUT_DIR/fabliq-8b-agent.F16.gguf" "$OUT_DIR/fabliq-8b-agent.${QUANT}.gguf" "$QUANT" 2>&1 | tail -5
done

# 3. 결과 확인
echo "=== Final GGUF files ==="
ls -lh "$OUT_DIR/"

# 4. F16 삭제 (용량 큼, 이미 Q8_0가 near-lossless)
rm -f "$OUT_DIR/fabliq-8b-agent.F16.gguf"
echo "=== After F16 cleanup ==="
ls -lh "$OUT_DIR/"
