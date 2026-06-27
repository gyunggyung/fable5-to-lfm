#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

BENCH_ROOT="${BENCH_ROOT:-$(cat /home/work/.data/fabliq_benchmarks/latest_run.txt)}"
TB2_DIR="${TB2_DIR:-$BENCH_ROOT/tb2_full}"
OUT_DIR="${OUT_DIR:-$BENCH_ROOT/lm_eval_wave2}"
PYTHON="${PYTHON:-$ROOT_DIR/.liquid-sft-env/bin/python}"

mkdir -p "$OUT_DIR/logs"

wait_for_tb2() {
  local pids_file="$TB2_DIR/pids.tsv"
  if [[ ! -s "$pids_file" ]]; then
    echo "No TB2 pids file found: $pids_file"
    return
  fi

  echo "Waiting for TB2 pids from $pids_file"
  while true; do
    local running=0
    while IFS=$'\t' read -r pid gpu name log; do
      [[ -z "${pid:-}" ]] && continue
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      fi
    done < "$pids_file"
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    echo "TB2 still running: $running jobs"
    sleep 60
  done
}

summarize_tb2() {
  "$PYTHON" tb2_lite/scripts/summarize_replay_results.py \
    --results-dir "$TB2_DIR" \
    --output-path "$BENCH_ROOT/tb2_full_summary.md" \
    > "$BENCH_ROOT/tb2_full_summary.stdout.md" || true
}

launch_lm_eval() {
  # GLM-5.2 benchmark-adjacent local tasks:
  # - leaderboard_gpqa_diamond: GPQA-Diamond family
  # - ifeval: instruction following
  # - mmlu_pro_computer_science: coding/reasoning proxy without code execution
  local tasks="${TASKS:-leaderboard_gpqa_diamond,ifeval,mmlu_pro_computer_science}"
  local limit_arg=()
  if [[ -n "${LIMIT:-}" ]]; then
    limit_arg=(--limit "$LIMIT")
  fi

  local models=(
    "base|LiquidAI/LFM2.5-8B-A1B|0"
    "toolbench|/home/work/.data/hf_upload_stage/lfm25_8b_a1b_toolbench_full/epoch1|1"
    "phase1-fabliq|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-ToolBench-Fable5-Agentic-FullSFT-20260623/final_model|2"
    "phase1b-frombase|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-FromBase-FullSFT-20260623/final_model|3"
    "phase2-reasoning|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Fable5-Phase2-Reasoning-FullSFT-20260623/final_model|4"
    "mega-combined|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-Combined-FullSFT-20260623/final_model|5"
    "mega-lr2e6|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr2e6-FullSFT-20260623/final_model|6"
    "mega-lr5e7|/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-Mega-lr5e7-FullSFT-20260623/final_model|7"
  )

  : > "$OUT_DIR/pids.tsv"
  for entry in "${models[@]}"; do
    IFS='|' read -r name model gpu <<< "$entry"
    local log="$OUT_DIR/logs/$name.log"
    mkdir -p "$OUT_DIR/$name"
    : > "$log"
    setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON" -m lm_eval run \
        --model hf \
        --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True" \
        --tasks "$tasks" \
        --apply_chat_template \
        --batch_size auto \
        --device cuda \
        --output_path "$OUT_DIR/$name" \
        --log_samples \
        "${limit_arg[@]}" \
        > "$log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$name" "$log" | tee -a "$OUT_DIR/pids.tsv"
  done
}

wait_for_tb2
summarize_tb2
launch_lm_eval
