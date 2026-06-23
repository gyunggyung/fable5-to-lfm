#!/usr/bin/env bash
set -euo pipefail
# 전 모델 eval 마스터 러너 (밤샘 자동화)
# 7개 모델 순차 eval: base, ToolBench, Phase-1, Phase-1B, Phase-2, Phase-2B, Mega

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

EVAL_ROOT="${EVAL_ROOT:-/home/work/.data/fabliq_eval}"
mkdir -p "$EVAL_ROOT"

# 모델 정의 (name|path)
MODELS=(
  "base|LiquidAI/LFM2.5-8B-A1B"
  "toolbench|/home/work/.data/hf_upload_stage/lfm25_8b_a1b_toolbench_full/epoch1"
  "phase1-fabliq|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623/final_model"
  "phase1b-frombase|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623/final_model"
  "phase2-reasoning|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623/final_model"
  "phase2b-frombase-reasoning|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-Phase2B-Reasoning-FullSFT-20260623/final_model"
  "mega-combined|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623/final_model"
)

START_FROM="${START_FROM:-0}"

for i in "${!MODELS[@]}"; do
  [[ "$i" -lt "$START_FROM" ]] && continue
  entry="${MODELS[$i]}"
  name="${entry%%|*}"
  path="${entry##*|}"

  if [[ ! -e "$path" && ! "$path" == */* ]]; then
    echo "SKIP $name: path $path not found"
    continue
  fi
  if [[ ! -e "$path" && "$path" != */* ]]; then
    echo "SKIP $name: path $path not found"
    continue
  fi

  echo ""
  echo "=============================================="
  echo "[$((i+1))/${#MODELS[@]}] $name → $path"
  echo "=============================================="

  MODEL="$path" \
  MODEL_SHORT="$name" \
  OUTPUT_DIR="$EVAL_ROOT/$name" \
  bash fable_distillation/scripts/run_fabliq_eval.sh \
    > "$EVAL_ROOT/$name.master.log" 2>&1 || echo "  $name eval failed (see log)"
done

echo ""
echo "=== All evals done. Aggregating... ==="
python fable_distillation/scripts/aggregate_eval_results.py \
  --eval-root "$EVAL_ROOT" \
  --output-md fable_distillation/EVAL_RESULTS_20260623.md \
  --output-json fable_distillation/EVAL_RESULTS_20260623.json
