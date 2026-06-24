#!/usr/bin/env bash
set -euo pipefail

# DiffusionGemma-first dLLM evaluation without Docker.
# Default path: local uv env + HF Transformers DiffusionGemmaForBlockDiffusion.
# Runs two checks:
#   1. TB2-lite replay quality, sharded across GPUs.
#   2. Small long-output probe for throughput/shape.
#
# Dry-run by default:
#   bash fable_distillation/scripts/run_diffusiongemma_dllm_eval_20260624.sh
# Actual:
#   RUN_NOW=1 bash fable_distillation/scripts/run_diffusiongemma_dllm_eval_20260624.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="$ROOT_DIR/fable_distillation"
cd "$ROOT_DIR"

RUN_NOW="${RUN_NOW:-0}"
RUN_ID="${RUN_ID:-20260624_diffusiongemma_dllm_base_transformers}"
DEFAULT_VLLM_ENV="$FABLE_DIR/.venvs/diffusiongemma-transformers-cu128"
if [[ ! -x "$DEFAULT_VLLM_ENV/bin/python" ]]; then
  DEFAULT_VLLM_ENV="$FABLE_DIR/.venvs/diffusiongemma-vllm"
fi
if [[ ! -x "$DEFAULT_VLLM_ENV/bin/python" ]]; then
  DEFAULT_VLLM_ENV="$ROOT_DIR/.vllm-lfm-cu12"
fi
VLLM_ENV="${VLLM_ENV:-$DEFAULT_VLLM_ENV}"
MODEL_PATH="${MODEL_PATH:-google/diffusiongemma-26B-A4B-it}"
MODEL_SHORT="${MODEL_SHORT:-diffusiongemma-26b-a4b-it-base}"
RESULTS_DIR="${RESULTS_DIR:-$FABLE_DIR/benchmarks/$RUN_ID/results}"
LOG_DIR="${LOG_DIR:-$FABLE_DIR/benchmarks/$RUN_ID/logs}"
PROMPT_JSONL="${PROMPT_JSONL:-$FABLE_DIR/benchmarks/20260624_dllm_probe/prompts.jsonl}"
BACKEND="${BACKEND:-transformers}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8008/v1}"
API_KEY="${API_KEY:-EMPTY}"
START_DOCKER_SERVER="${START_DOCKER_SERVER:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-diffusiongemma-vllm-gemma}"

GPU="${GPU:-0}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TB2_LIMIT="${TB2_LIMIT:-0}"
RUN_TB2="${RUN_TB2:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
PREFETCH_MODEL="${PREFETCH_MODEL:-1}"
TRANSFORMERS_SHARD_COUNT="${TRANSFORMERS_SHARD_COUNT:-8}"
TRANSFORMERS_GPUS="${TRANSFORMERS_GPUS:-0,1,2,3,4,5,6,7}"

HF_OVERRIDES_JSON="${HF_OVERRIDES_JSON:-{\"diffusion_sampler\":\"entropy_bound\",\"diffusion_entropy_bound\":0.1}}"
ENGINE_KWARGS_JSON="${ENGINE_KWARGS_JSON:-{\"diffusion_config\":{\"canvas_length\":256}}}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="transformers"
fi

VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-12.9:/usr/local/cuda-12.9/include:/usr/include/x86_64-linux-gnu:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

base_env=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  TOKENIZERS_PARALLELISM=false
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH"
  CUDA_VISIBLE_DEVICES="$GPU"
)
transformers_base_env=(
  env -u PYTHONPATH
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  TOKENIZERS_PARALLELISM=false
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  HF_DEACTIVATE_ASYNC_LOAD=1
  LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH"
)

build_probe_cmd=(
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python"
  "$FABLE_DIR/scripts/build_dllm_probe_prompts_20260624.py"
  --output "$PROMPT_JSONL"
)

prefetch_cmd=(
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python"
  "$FABLE_DIR/scripts/prefetch_diffusiongemma_hf_20260624.py"
  --model "$MODEL_PATH"
)

tb2_cmd=(
  "${base_env[@]}"
  "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/replay_eval_vllm.py"
  --model "$MODEL_PATH"
  --tokenizer-path "$MODEL_PATH"
  --model-short "$MODEL_SHORT"
  --gpu "$GPU"
  --eval-path "${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
  --output-dir "$RESULTS_DIR"
  --dtype bfloat16
  --tp "$TP"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-tokens "$MAX_TOKENS"
  --temperature 0.0
  --top-p 1.0
  --language-model-only
  --thinking-mode off
  --allow-raw-fallback
  --hf-overrides-json "$HF_OVERRIDES_JSON"
  --engine-kwargs-json "$ENGINE_KWARGS_JSON"
)
if [[ "$TB2_LIMIT" != "0" ]]; then
  tb2_cmd+=(--limit "$TB2_LIMIT")
fi

probe_cmd=(
  "${base_env[@]}"
  "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/vllm_prompt_probe.py"
  --model "$MODEL_PATH"
  --tokenizer-path "$MODEL_PATH"
  --prompt-jsonl "$PROMPT_JSONL"
  --output-path "$RESULTS_DIR/${MODEL_SHORT}.probe.json"
  --dtype bfloat16
  --tp "$TP"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-tokens "$MAX_TOKENS"
  --temperature 0.0
  --top-p 1.0
  --chat-template-kwargs-json '{"enable_thinking": false}'
  --engine-kwargs-json "$ENGINE_KWARGS_JSON"
)

print_cmd() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

print_cmd "build probe" "${build_probe_cmd[@]}"
print_cmd "prefetch model" "${prefetch_cmd[@]}"
if [[ "$BACKEND" == "offline_vllm" ]]; then
  print_cmd "tb2 quality" "${tb2_cmd[@]}"
  print_cmd "long probe" "${probe_cmd[@]}"
elif [[ "$BACKEND" == "transformers" ]]; then
  transformers_cmd=(
    "${transformers_base_env[@]}" CUDA_VISIBLE_DEVICES="$GPU"
    "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/diffusiongemma_transformers_eval.py"
    --model "$MODEL_PATH"
    --model-short "$MODEL_SHORT"
    --output-dir "$RESULTS_DIR"
    --max-new-tokens "$MAX_TOKENS"
  )
  if [[ "$RUN_TB2" == "1" ]]; then
    transformers_cmd+=(--eval-path "${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}")
    if [[ "$TB2_LIMIT" != "0" ]]; then
      transformers_cmd+=(--limit "$TB2_LIMIT")
    fi
  fi
  if [[ "$RUN_PROBE" == "1" ]]; then
    transformers_cmd+=(--prompt-jsonl "$PROMPT_JSONL")
  fi
  print_cmd "transformers fallback" "${transformers_cmd[@]}"
  if [[ "$TRANSFORMERS_SHARD_COUNT" != "1" && "$RUN_TB2" == "1" ]]; then
    echo "transformers sharded TB2: shard_count=$TRANSFORMERS_SHARD_COUNT gpus=$TRANSFORMERS_GPUS"
  fi
else
  openai_tb2_cmd=(
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python"
    "$FABLE_DIR/scripts/replay_eval_openai_chat.py"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --model "$MODEL_PATH"
    --model-short "$MODEL_SHORT"
    --eval-path "${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
    --output-dir "$RESULTS_DIR"
    --max-tokens "$MAX_TOKENS"
    --temperature 0.0
    --top-p 1.0
    --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'
  )
  if [[ "$TB2_LIMIT" != "0" ]]; then
    openai_tb2_cmd+=(--limit "$TB2_LIMIT")
  fi
  openai_probe_cmd=(
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python"
    "$FABLE_DIR/scripts/openai_prompt_probe.py"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --model "$MODEL_PATH"
    --prompt-jsonl "$PROMPT_JSONL"
    --output-path "$RESULTS_DIR/${MODEL_SHORT}.probe.json"
    --max-tokens "$MAX_TOKENS"
    --temperature 0.0
    --top-p 1.0
    --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'
  )
  print_cmd "tb2 quality" "${openai_tb2_cmd[@]}"
  print_cmd "long probe" "${openai_probe_cmd[@]}"
fi

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to execute."
  exit 0
fi

"${build_probe_cmd[@]}" 2>&1 | tee "$LOG_DIR/build_probe.log"
if [[ "$PREFETCH_MODEL" == "1" ]]; then
  "${prefetch_cmd[@]}" 2>&1 | tee "$LOG_DIR/prefetch_model.log"
fi
if [[ "$BACKEND" == "offline_vllm" ]]; then
  if [[ "$RUN_TB2" == "1" ]]; then
    "${tb2_cmd[@]}" 2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.tb2.log"
  fi
  if [[ "$RUN_PROBE" == "1" ]]; then
    "${probe_cmd[@]}" 2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.probe.log"
  fi
elif [[ "$BACKEND" == "transformers" ]]; then
  if [[ "$RUN_TB2" == "1" && "$TRANSFORMERS_SHARD_COUNT" != "1" ]]; then
    IFS=',' read -r -a shard_gpus <<< "$TRANSFORMERS_GPUS"
    if [[ "${#shard_gpus[@]}" -lt "$TRANSFORMERS_SHARD_COUNT" ]]; then
      echo "TRANSFORMERS_GPUS has fewer entries than TRANSFORMERS_SHARD_COUNT" >&2
      exit 1
    fi
    limit_args=()
    if [[ "$TB2_LIMIT" != "0" ]]; then
      limit_args=(--limit "$TB2_LIMIT")
    fi
    pids=()
    for shard_idx in $(seq 0 $((TRANSFORMERS_SHARD_COUNT - 1))); do
      shard_model_short="$(printf '%s.part%02d' "$MODEL_SHORT" "$shard_idx")"
      shard_log="$(printf '%s/%s.transformers.part%02d.log' "$LOG_DIR" "$MODEL_SHORT" "$shard_idx")"
      "${transformers_base_env[@]}" CUDA_VISIBLE_DEVICES="${shard_gpus[$shard_idx]}" \
        "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/diffusiongemma_transformers_eval.py" \
          --model "$MODEL_PATH" \
          --model-short "$shard_model_short" \
          --output-dir "$RESULTS_DIR" \
          --max-new-tokens "$MAX_TOKENS" \
          --eval-path "${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}" \
          --shard-index "$shard_idx" \
          --shard-count "$TRANSFORMERS_SHARD_COUNT" \
          "${limit_args[@]}" \
        > "$shard_log" 2>&1 &
      pids+=("$!")
      echo -e "$shard_idx\t${shard_gpus[$shard_idx]}\t${pids[-1]}\t$shard_log" | tee -a "$LOG_DIR/transformers_shards.tsv"
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" != "0" ]]; then
      echo "one or more transformers shards failed" >&2
      exit 1
    fi
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python" \
      "$FABLE_DIR/scripts/merge_diffusiongemma_transformers_shards.py" \
      --input-dir "$RESULTS_DIR" \
      --glob "${MODEL_SHORT}.part*.transformers.json" \
      --output-path "$RESULTS_DIR/${MODEL_SHORT}.json" \
      --model-short "$MODEL_SHORT" \
      2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.merge.log"
    if [[ "$RUN_PROBE" == "1" ]]; then
      "${transformers_base_env[@]}" CUDA_VISIBLE_DEVICES="${shard_gpus[0]}" \
        "$VLLM_ENV/bin/python" "$FABLE_DIR/scripts/diffusiongemma_transformers_eval.py" \
          --model "$MODEL_PATH" \
          --model-short "${MODEL_SHORT}.probe" \
          --output-dir "$RESULTS_DIR" \
          --max-new-tokens "$MAX_TOKENS" \
          --prompt-jsonl "$PROMPT_JSONL" \
        2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.probe.transformers.log"
    fi
  else
    "${transformers_cmd[@]}" 2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.transformers.log"
  fi
else
  if [[ "$START_DOCKER_SERVER" == "1" ]]; then
    PORT="${BASE_URL#http://127.0.0.1:}"
    PORT="${PORT%%/*}"
    PORT="${PORT%%:*}"
    PORT="$PORT" CONTAINER_NAME="$CONTAINER_NAME" GPU_DEVICE="$GPU" \
      bash "$FABLE_DIR/scripts/run_diffusiongemma_vllm_docker_server_20260624.sh" \
      2>&1 | tee "$LOG_DIR/docker_start.log"
    for _ in $(seq 1 120); do
      if curl -fsS "$BASE_URL/models" >/dev/null 2>&1; then
        break
      fi
      sleep 10
    done
    curl -fsS "$BASE_URL/models" >/dev/null
  fi
  if [[ "$RUN_TB2" == "1" ]]; then
    "${openai_tb2_cmd[@]}" 2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.tb2.log"
  fi
  if [[ "$RUN_PROBE" == "1" ]]; then
    "${openai_probe_cmd[@]}" 2>&1 | tee "$LOG_DIR/${MODEL_SHORT}.probe.log"
  fi
fi

if [[ "$RUN_TB2" == "1" && "$BACKEND" != "transformers" ]]; then
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VLLM_ENV/bin/python" \
    "$FABLE_DIR/scripts/summarize_replay_results.py" \
    --results-dir "$RESULTS_DIR" \
    --output-path "$RESULTS_DIR/SUMMARY.md" \
    > "$RESULTS_DIR/SUMMARY.stdout.md" || true
fi
