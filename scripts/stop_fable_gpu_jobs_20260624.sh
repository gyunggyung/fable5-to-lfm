#!/usr/bin/env bash
set -euo pipefail

# Stop only fable_distillation GPU/eval jobs. Dry-run by default.
# Usage:
#   bash fable_distillation/scripts/stop_fable_gpu_jobs_20260624.sh
#   RUN_NOW=1 bash fable_distillation/scripts/stop_fable_gpu_jobs_20260624.sh

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
FABLE_DIR="${FABLE_DIR:-$ROOT_DIR/fable_distillation}"
RUN_NOW="${RUN_NOW:-0}"
TERM_WAIT_SEC="${TERM_WAIT_SEC:-5}"

collect_pids() {
  ps -eo pid=,args= | while read -r pid args; do
    [[ -z "${pid:-}" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    case "$args" in
      *"$FABLE_DIR"*torch.distributed.run*|\
      *"$FABLE_DIR"*diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py*|\
      *"$FABLE_DIR"*run_diffusiongemma_strength_lora_20260624.sh*|\
      *"$FABLE_DIR"*replay_eval_vllm.py*|\
      *"$FABLE_DIR"*run_qwen35_9b_tb2_vllm_sharded_20260624.sh*|\
      *"$FABLE_DIR"*run_glm52_chaser_mix_sft_20260624.sh*|\
      *"$FABLE_DIR"*lm_eval*|\
      *"$FABLE_DIR"*vllm*)
        printf '%s\n' "$pid"
        ;;
    esac
  done | sort -u
}

print_gpu_state() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU compute apps:"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
    echo "GPU memory/utilization:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  fi
}

mapfile -t pids < <(collect_pids)

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No fable_distillation GPU/eval jobs matched."
  print_gpu_state
  exit 0
fi

echo "Matched fable_distillation job PIDs:"
printf '  %s\n' "${pids[@]}"

if [[ "$RUN_NOW" != "1" ]]; then
  echo "DRY-RUN. Set RUN_NOW=1 to stop these jobs."
  print_gpu_state
  exit 0
fi

kill "${pids[@]}" 2>/dev/null || true
sleep "$TERM_WAIT_SEC"

mapfile -t remaining < <(collect_pids)
if [[ "${#remaining[@]}" -gt 0 ]]; then
  echo "Still running after SIGTERM; sending SIGKILL:"
  printf '  %s\n' "${remaining[@]}"
  kill -9 "${remaining[@]}" 2>/dev/null || true
  sleep 2
fi

print_gpu_state
