#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
OUT_DIR="${OUT_DIR:-/home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/work/.cache/llama.cpp-mtp}"
QUANTS="${QUANTS:-Q4_K_M Q5_K_M Q8_0}"
MAKE_BF16="${MAKE_BF16:-1}"
UPLOAD_REPO_ID="${UPLOAD_REPO_ID:-}"

mkdir -p "$OUT_DIR"

MODEL_DIR="$(
python - "$MODEL_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1]))
PY
)"

PATCHED_CONVERTER="$OUT_DIR/convert_hf_to_gguf_lfm25_terminal.py"
python - "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$PATCHED_CONVERTER" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
text = src.read_text()
needle = '''        if chkhsh == "169bf0296a13c4d9b7672313f749eb36501d931022de052aad6e36f2bf34dd51":
            # ref: https://huggingface.co/LiquidAI/LFM2-Tokenizer
            res = "lfm2"
'''
patch = needle + '''        if chkhsh == "9e454714343b69b99b71795c1d27a68c2a1d15dab111f4d353109f966af29da7":
            # ref: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
            res = "lfm2"
'''
if "9e454714343b69b99b71795c1d27a68c2a1d15dab111f4d353109f966af29da7" not in text:
    text = text.replace(needle, patch)
dst.write_text(text)
PY

export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"
export PYTHONPATH="$LLAMA_CPP_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}"

BF16_GGUF="$OUT_DIR/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.BF16.gguf"
if [[ "$MAKE_BF16" == "1" && ! -s "$BF16_GGUF" ]]; then
  python "$PATCHED_CONVERTER" "$MODEL_DIR" \
    --outtype bf16 \
    --outfile "$BF16_GGUF" \
    > "$OUT_DIR/convert_bf16.log" 2>&1
fi

if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j "${BUILD_JOBS:-8}"
fi

for quant in $QUANTS; do
  case "$quant" in
    BF16)
      continue
      ;;
    Q8_0)
      target="$OUT_DIR/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q8_0.gguf"
      if [[ ! -s "$target" ]]; then
        python "$PATCHED_CONVERTER" "$MODEL_DIR" \
          --outtype q8_0 \
          --outfile "$target" \
          > "$OUT_DIR/convert_q8_0.log" 2>&1
      fi
      ;;
    *)
      target="$OUT_DIR/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.${quant}.gguf"
      if [[ ! -s "$target" ]]; then
        "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$BF16_GGUF" "$target" "$quant" "${QUANT_THREADS:-16}" \
          > "$OUT_DIR/quantize_${quant}.log" 2>&1
      fi
      ;;
  esac
done

python - "$OUT_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.gguf")):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    rows.append({"file": path.name, "size_bytes": path.stat().st_size, "sha256": h.hexdigest()})
(out_dir / "checksums.json").write_text(json.dumps(rows, indent=2) + "\n")
(out_dir / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in rows))
PY

if [[ -n "$UPLOAD_REPO_ID" ]]; then
  python - "$OUT_DIR" "$UPLOAD_REPO_ID" <<'PY'
import os
import pathlib
import sys

from huggingface_hub import HfApi, create_repo, upload_folder

out_dir = pathlib.Path(sys.argv[1])
repo_id = sys.argv[2]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(out_dir), token=token)
PY
fi
