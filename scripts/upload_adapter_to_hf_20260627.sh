#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation"
cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/huggingface/hub}"

HF_BIN="${HF_BIN:-/home/work/.local/bin/hf}"
ENV_FILE="${ENV_FILE:-/home/work/.projects/LLM-OS-Models/Terminal/.env}"
PREFER_ENV_FILE_TOKEN="${PREFER_ENV_FILE_TOKEN:-1}"
HF_ORG="${HF_ORG:-LLM-OS-Models}"
REPO_NAME="${REPO_NAME:-}"
ARTIFACT_DIR="${ARTIFACT_DIR:-}"
MODEL_CARD="${MODEL_CARD:-}"
PRIVATE="${PRIVATE:-0}"

if [[ "${PREFER_ENV_FILE_TOKEN}" == "1" && -f "${ENV_FILE}" ]]; then
  HF_TOKEN_VALUE="$(awk -F= '{key=$1; sub(/^export[[:space:]]+/, "", key)} key == "HF_TOKEN" {print substr($0, index($0, "=") + 1); exit}' "${ENV_FILE}")"
  HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\"}"
  HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\"}"
  HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\'}"
  HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\'}"
  if [[ -n "${HF_TOKEN_VALUE}" ]]; then
    export HF_TOKEN="${HF_TOKEN_VALUE}"
  fi
fi

if [[ -z "${REPO_NAME}" || -z "${ARTIFACT_DIR}" ]]; then
  cat <<EOF
Usage:
  REPO_NAME=<repo-name> ARTIFACT_DIR=<adapter-or-model-dir> RUN_NOW=1 $0

Optional:
  HF_ORG=${HF_ORG}
  MODEL_CARD=<README.md path to upload first>
  PRIVATE=1

Example:
  REPO_NAME=Qwen3.5-9B-Fable-OfficialAgentic-LoRA-20260627 \\
  ARTIFACT_DIR=/home/work/.data/harness1/models/Qwen3.5-9B__Fable-OfficialAgentic-LoRA-qkvo-b2-200step-20260627/final_lora \\
  MODEL_CARD=docs/hf_cards/Qwen3.5-9B-Fable-OfficialAgentic-LoRA-20260627.md \\
  RUN_NOW=1 $0
EOF
  exit 2
fi

if [[ ! -d "${ARTIFACT_DIR}" ]]; then
  echo "artifact dir does not exist: ${ARTIFACT_DIR}" >&2
  exit 2
fi

REPO_ID="${HF_ORG}/${REPO_NAME}"
PRIVATE_FLAG=()
if [[ "${PRIVATE}" == "1" ]]; then
  PRIVATE_FLAG=(--private)
fi
TOKEN_FLAG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_FLAG=(--token "${HF_TOKEN}")
  "${HF_BIN}" auth login --token "${HF_TOKEN}" --force >/dev/null
fi

echo "HF identity:"
"${HF_BIN}" auth whoami || {
  echo "Not logged in under HF_HOME=${HF_HOME}. Set a token with write access to ${HF_ORG}." >&2
  exit 1
}

if [[ "${RUN_NOW:-0}" != "1" ]]; then
  echo "Dry run. Set RUN_NOW=1 to create/upload ${REPO_ID} from ${ARTIFACT_DIR}."
  exit 0
fi

"${HF_BIN}" repo create "${REPO_ID}" --repo-type model "${PRIVATE_FLAG[@]}" "${TOKEN_FLAG[@]}" --exist-ok

if [[ -n "${MODEL_CARD}" ]]; then
  if [[ ! -f "${MODEL_CARD}" ]]; then
    echo "model card does not exist: ${MODEL_CARD}" >&2
    exit 2
  fi
  "${HF_BIN}" upload "${REPO_ID}" "${MODEL_CARD}" README.md --repo-type model "${TOKEN_FLAG[@]}"
fi

"${HF_BIN}" upload "${REPO_ID}" "${ARTIFACT_DIR}" . --repo-type model "${TOKEN_FLAG[@]}"
echo "Uploaded: https://huggingface.co/${REPO_ID}"
