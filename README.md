[**English**](README.md) | [**한국어**](README.ko.md)

# fable5-to-lfm

> **fable5-to-lfm** — Distilling [Fable-5](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) agentic terminal traces into [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) (8B MoE, ~1B active) to build the **Fabliq** family of terminal-agent models.

The name says exactly what the project does: **Fable-5 → LFM**. Reference lineups: [Qwable-9B](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5), [Qwythos-9B](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M), [gemma-4-12B-agentic-fable5](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).

---

## 🌊 Fabliq Model Lineup

| Model | Stage | Base | Data | Status | HF Link |
| --- | --- | --- | --- | --- | --- |
| **Fabliq-8B-Agent** | Phase-1 | ToolBench-Full-SFT-1Epoch | Fable-5 4,047 rows × 3 epoch | ✅ Done | [LLM-OS-Models/Fabliq-8B-Agent](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent) |
| **Fabliq-8B-Agent-Reasoning** | Phase-2 | Fabliq-8B-Agent | + WithinUs 135 + Helio 146 = 281 rows × 4 epoch | ✅ Done | [LLM-OS-Models/Fabliq-8B-Agent-Reasoning](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Reasoning) |
| **Fabliq-8B-Agent-FromBase** | Phase-1B | raw LiquidAI/LFM2.5-8B-A1B | Fable-5 4,047 rows × 3 epoch | ✅ Done | [LLM-OS-Models/Fabliq-8B-Agent-FromBase](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-FromBase) |
| **Fabliq-8B-Agent-FromBase-Reasoning** | Phase-2B | Fabliq-8B-Agent-FromBase | + WithinUs+Helio 281 rows × 4 epoch | 🔄 Training | - |
| **Fabliq-8B-Agent-Large** | Phase-3 | (Phase-2 or base) | Complete-FABLE.5-traces-2M 3,866 rows × 2 epoch | 📋 Queued | - |
| **Fabliq-8B-Agent-GGUF** | Quantized | Fabliq-8B-Agent | Q4_K_M/Q5_K_M/Q6_K/Q8_0 | ✅ Done | [LLM-OS-Models/Fabliq-8B-Agent-GGUF](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-GGUF) |

---

## 📂 File Inventory

### Top-level markdown

| File | Description |
| --- | --- |
| `README.md` | **This file** (English). |
| `README.ko.md` | Korean version. |
| `PROGRESS_20260623.ko.md` | Timeline of progress (Phase-1/2/3 + GPU usage log). |
| `DATA_SOURCES_20260623.ko.md` | Detailed analysis of 6 datasets (source structure, preprocessing, current usage). |
| `NEXT_STEPS_20260623.ko.md` | Follow-up work candidates (priority 1-7). |
| `TRAINING_USAGE.md` | Initial dataset analysis (Korean, 6-dataset comparison + scenarios A-E). |
| `README.old.md` | Previous README (datasets-only early version). |

### Data preprocessing scripts (`*.py` in root)

| File | Input | Output | Description |
| --- | --- | --- | --- |
| `build_fable5_to_lfm_sft.py` | `datasets/Fable-5-traces/fable5_cot_merged.jsonl` (4,665 rows) | `datasets/fable5_lfm_sft_20260623.jsonl` (4,047 rows) | **Phase-1 main builder.** Parses Glint `context` → multi-turn messages, `{tool, input}` → `<\|tool_call_start\|>...<\|tool_call_end\|>`, `cot` → `<think>...</think>`. Drops 618 short rows. |
| `build_withinus_lfm_sft.py` | `datasets/claude_mythos_distilled_25k/...` (25k rows) | `datasets/withinus_lfm_sft_20260623.jsonl` (135 rows) | **Phase-2 WithinUs builder.** Category-balanced sampling, SHA-256 dedup (25k → 135 unique), removes "Drawing from the autonomous..." template first sentence. |
| `build_helio_lfm_sft.py` | `datasets/Fable-5-Distill-Reasoning-462x/...` (462 rows) | `datasets/helio_lfm_sft_20260623.jsonl` (146 rows) | **Phase-2 Helio builder.** Cyrillic ratio <30% filter (drops Russian-dominant), `<think>` wrapping, line 192 corruption skip. |
| `build_fable5_2m_to_lfm_sft.py` | `datasets/Complete-FABLE.5-traces-2M/data/train.parquet` (2M events) | `datasets/fable5_2m_lfm_sft_20260623.jsonl` (3,866 rows) | **Phase-3 builder.** Filters Parquet `row_json` for Glint structured rows (`output_type` key present), `seen_count<=100`. Reuses `build_fable5_to_lfm_sft.convert_row`. |

### `training/` — LFM2.5 training scripts (copied from harness-1)

| File | Description |
| --- | --- |
| `training/train_lfm25_rlvr_json_sft.py` | **Main training script.** Full-parameter SFT + LoRA. FSDP `full_shard`, `Lfm2MoeDecoderLayer` auto-wrap, activation_checkpointing. Tokenizes via `apply_chat_template(messages)`. Saves to `final_model/` (full) or `final_lora/` (LoRA). |
| `training/build_lfm25_agentic_sft.py` | Agentic SFT data builder (terminal tool-use JSONL). |
| `training/build_lfm25_rlvr_json_sft.py` | RLVR SFT data builder. |
| `training/mix_lfm25_sft_jsonl.py` | Utility to mix/shuffle multiple JSONLs. |

### `eval_scripts/` — LFM2.5 eval scripts (copied from harness-1)

| File | Description |
| --- | --- |
| `eval_scripts/eval_lfm25_agentic_vllm.py` | **Agentic tool-use eval.** Runs multi-turn terminal trajectories via vLLM HTTP. Uses Harness-1 retrieval-curation system prompt. |
| `eval_scripts/eval_lfm25_rlvr_retrieval_vllm.py` | Retrieval-curation eval (single-turn). |

### `scripts/` — Shell runners

| File | Description |
| --- | --- |
| `scripts/run_fable5_full_sft_20260623.sh` | **Phase-1 runner.** ToolBench-Full-SFT-1Epoch → Fable-5 SFT. 8 H200, max_seq 8192, epochs 3, LR 5e-7, batch 2×accum 4. |
| `scripts/run_phase2_reasoning_sft_20260623.sh` | **Phase-2 runner.** Fabliq-8B-Agent → + WithinUs+Helio reasoning. 8 H200, max_seq 8192, epochs 4, LR 3e-7. |
| `scripts/run_fable5_from_base_sft_20260623.sh` | **Phase-1B runner (ablation).** raw LiquidAI/LFM2.5-8B-A1B → Fable-5 SFT. Isolates ToolBench-foundation effect. LR 1e-6 (higher since from base). |
| `scripts/run_phase2b_reasoning_sft_20260623.sh` | **Phase-2B runner.** Fabliq-8B-Agent-FromBase + reasoning expansion (mirror of Phase-2 on the FromBase variant). |
| `scripts/run_fable5_2m_phase3_sft_20260623.sh` | **Phase-3 runner (queued).** Phase-1B/Phase-2 + 2M traces scale-out. |
| `scripts/convert_lfm25_gguf.sh` | GGUF converter copied from Liquid-CLI. Includes LFM2.5 tokenizer hash (9e45...) patch. |
| `scripts/convert_fabliq_gguf.sh` | **Fabliq-specific GGUF converter.** Based on `convert_lfm25_gguf.sh`, parameterized model name/output path. Generates Q4_K_M/Q5_K_M/Q6_K/Q8_0 + HF upload option. |
| `scripts/convert_fabliq_to_gguf.sh` | Initial simple converter (fails without tokenizer patch - kept for reference). |

### `datasets/` — Source + preprocessed data

| Directory | Source | Rows | Notes |
| --- | --- | --- | --- |
| `datasets/Fable-5-traces/` | [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | 4,665 | Phase-1 input. `fable5_cot_merged.jsonl` plus pi-traces. |
| `datasets/claude_mythos_distilled_25k/` | WithinUsAI | 25,000 | Phase-2 WithinUs source. |
| `datasets/Fable-5-Distill-Reasoning-462x/` | HelioAI | 462 | Phase-2 Helio source. |
| `datasets/Complete-FABLE.5-traces-2M/` | Glint-Research | 2,006,487 | Phase-3 input. Parquet + jsonl.gz. |
| `datasets/agentic-distill-fable-5-sft/` | lordx64 | 4,659 | Unused (less clean than Glint original). |
| `datasets/claude-fable-5-claude-code/` | armand0e | 18,370 | Unused (Phase-3 alternative). |

**Preprocessed JSONL outputs:**

| File | Rows | Used by |
| --- | --- | --- |
| `datasets/fable5_lfm_sft_20260623.jsonl` | 4,047 | Phase-1 |
| `datasets/withinus_lfm_sft_20260623.jsonl` | 135 | Phase-2 (WithinUs part) |
| `datasets/helio_lfm_sft_20260623.jsonl` | 146 | Phase-2 (Helio part) |
| `datasets/phase2_reasoning_lfm_sft_20260623.jsonl` | 281 | Phase-2 (WithinUs+Helio combined) |
| `datasets/fable5_mixed_lfm_sft_20260623.jsonl` | 4,328 | Phase-1+2 combined (alternative) |
| `datasets/fable5_2m_lfm_sft_20260623.jsonl` | 3,866 | Phase-3 |

---

## 🔧 Environment Setup

```bash
# Training env (.liquid-sft-env) — at project root
cd /home/work/.projects/LLM-OS-Models/Terminal
source .liquid-sft-env/bin/activate

# Dependencies (already installed)
# - torch 2.10+cu128 (FSDP full_shard)
# - transformers 5.5+ (Lfm2MoeForCausalLM)
# - trl 0.16+, datasets, accelerate
# - lm-eval 0.4.11, vllm (for evaluation)
```

**Hardware:** 8× NVIDIA H200 141GB (full_shard distributed)

---

## 🚀 Reproducing Training

### Phase-1: Fable-5 Agentic Foundation (✅ Done)

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_full_sft_20260623.sh
```

### Phase-2: Reasoning Expansion (✅ Done)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_phase2_reasoning_sft_20260623.sh
```

### Phase-1B: Base → Fable-5 Ablation (✅ Done)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_from_base_sft_20260623.sh
```

### Phase-2B: FromBase + Reasoning (🔄 Training)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_phase2b_reasoning_sft_20260623.sh
```

### Phase-3: 2M Traces Scale-Out (📋 Queued)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_2m_phase3_sft_20260623.sh
```

---

## 📊 Data Preprocessing Pipeline

### Phase-1 data build

```bash
python fable_distillation/build_fable5_to_lfm_sft.py \
  --input fable_distillation/datasets/Fable-5-traces/fable5_cot_merged.jsonl \
  --output fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl
# Result: 4,047 rows (618 short-context drops)
```

### Phase-2 data build

```bash
python fable_distillation/build_withinus_lfm_sft.py
python fable_distillation/build_helio_lfm_sft.py

# Combine
head -n 135 fable_distillation/datasets/withinus_lfm_sft_20260623.jsonl > phase2.jsonl
cat fable_distillation/datasets/helio_lfm_sft_20260623.jsonl >> phase2.jsonl
mv phase2.jsonl fable_distillation/datasets/phase2_reasoning_lfm_sft_20260623.jsonl
```

### Phase-3 data build

```bash
python fable_distillation/build_fable5_2m_to_lfm_sft.py \
  --max-rows 50000 --seen-count-max 100
# Result: 3,866 rows from 2M events
```

---

## 🔍 Evaluation

### tb2_lite terminal task eval

```bash
# Serve model with vLLM
bash Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh &

# Run terminal task replay eval
python tb2_lite/scripts/replay_eval_lfm_vllm.py \
  --model-path $MODEL_PATH \
  --vllm-base-url http://127.0.0.1:8137/v1 \
  --dataset-path tb2_lite/data/replay_dev_20.jsonl \
  --output-jsonl /tmp/eval.jsonl
```

### Standard benchmarks (MMLU, GPQA, HumanEval) via lm-eval-harness

```bash
vllm serve $MODEL_PATH --max-model-len 8192 --dtype bfloat16 --port 8000 &

lm_eval --model vllm \
  --model_args pretrained=$MODEL_PATH,base_url=http://localhost:8000/v1,dtype=bfloat16 \
  --tasks mmlu,gpqa_diamond,humaneval \
  --batch_size auto
```

---

## 📦 GGUF Conversion (CPU parallel)

```bash
bash fable_distillation/scripts/convert_fabliq_gguf.sh
# Output: /home/work/.data/gguf/Fabliq-8B-Agent/Fabliq-8B-Agent.{Q4_K_M,Q5_K_M,Q6_K,Q8_0}.gguf
# Auto-upload: UPLOAD_REPO_ID=LLM-OS-Models/Fabliq-8B-Agent-GGUF
```

---

## 📈 Training Results Summary

### Phase-1 (Fable-5 Agentic, 4,047 rows × 3 epoch)
- Final train_loss: **1.277**
- Train runtime: 831s (~14 min)
- Global step: 192
- LR: 5e-7 constant, batch 2×accum 4×8 GPU = global 64

### Phase-2 (Reasoning Expansion, 281 rows × 4 epoch)
- Final train_loss: **~1.6** (small data + harder reasoning = higher loss, expected)
- Train runtime: ~6 min
- Global step: 20
- LR: 3e-7 (lower than Phase-1 to avoid forgetting)

### Phase-1B (Base → Fable-5, ablation)
- Same data as Phase-1 but starting from raw LiquidAI/LFM2.5-8B-A1B (no ToolBench foundation)
- LR 1e-6 (higher since base model is less specialized)

---

## 🔗 Related Docs

- [Progress Timeline](./PROGRESS_20260623.ko.md) (Korean)
- [Data Sources Detail](./DATA_SOURCES_20260623.ko.md) (Korean)
- [Next Steps](./NEXT_STEPS_20260623.ko.md) (Korean)
- [TRAINING_USAGE.md (initial dataset analysis)](./TRAINING_USAGE.md)
- [Korean README](./README.ko.md)

---

## 🏷️ License

Apache 2.0 (inherited from LFM2.5-8B-A1B base). The Fable-5 training traces are distilled from Claude, original Anthropic usage policy applies.
