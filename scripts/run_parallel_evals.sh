#!/usr/bin/env bash
# 8 모델 병렬 eval (GPU 0-7 각각 1모델)
# simple_fabliq_eval.py (transformers, single GPU) 사용

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

EVAL_ROOT="/home/work/.data/fabliq_eval"
mkdir -p "$EVAL_ROOT"

PYTHON="$ROOT_DIR/.liquid-sft-env/bin/python"

# 8개 모델 + GPU 매핑 (model_name|model_path|gpu)
MODELS=(
  "base|LiquidAI/LFM2.5-8B-A1B|0"
  "phase1-fabliq|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623/final_model|1"
  "phase1b-frombase|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623/final_model|2"
  "phase2-reasoning|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623/final_model|3"
  "mega-combined|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623/final_model|4"
  "mega-lr2e6|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr2e6-FullSFT-20260623/final_model|5"
  "mega-lr5e7|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr5e7-FullSFT-20260623/final_model|6"
  "fabliq-1.2b|/home/work/.data/harness1/models/LFM2.5-1.2B-Instruct__Terminal-Fable5-FullSFT-20260623/final_model|7"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name path gpu <<< "$entry"
  echo "Starting eval: $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u fable_distillation/eval_scripts/simple_fabliq_eval.py \
    --model "$path" \
    --model-name "$name" \
    --output "$EVAL_ROOT/$name/eval.json" \
    > "$EVAL_ROOT/$name.log" 2>&1 &
done

echo "All 8 evals started in parallel. Waiting..."
wait
echo "=== All evals done ==="

# 결과 취합
echo ""
echo "=== Results Summary ==="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name path gpu <<< "$entry"
  if [[ -f "$EVAL_ROOT/$name/eval.json" ]]; then
    "$PYTHON" -c "
import json
with open('$EVAL_ROOT/$name/eval.json') as f:
    d = json.load(f)
print(f'{d[\"model_name\"]:<25} tool_rate={d[\"tool_call_rate\"]:.2f} think_rate={d[\"think_rate\"]:.2f} tool_correct={d[\"tool_correct_rate\"]:.2f} avg_lat={d[\"avg_latency\"]:.1f}s')
"
  fi
done
