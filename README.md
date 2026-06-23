[**English**](README.md) | [**한국어**](README.ko.md)

# fable5-to-lfm

> **fable5-to-lfm** — Distilling [Fable-5](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) agentic terminal traces into LiquidAI's LFM family (1.2B / 2.6B / 8B / 24B-A2B) to build the **Fabliq** terminal-agent model line.

The name says what the project does: **Fable-5 → LFM**. 12 model variants trained across 3 model sizes, all published on HuggingFace.

Reference lineups: [Qwable-9B](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5), [Qwythos-9B](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M), [gemma-4-12B-agentic-fable5](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).

---

## 🌊 Fabriq Model Lineup (12 variants + 12 GGUF = 24 repos)

### 8B Class (10 variants)

| Model | Stage | Base | Final Loss | HF |
| --- | --- | --- | --- | --- |
| **Fabliq-8B-Agent** | Phase-1 | ToolBench-Full-SFT-1Epoch | 1.277 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent) |
| **Fabliq-8B-Agent-Reasoning** | Phase-2 | Fabliq-8B-Agent | ~1.6 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Reasoning) |
| Fabliq-8B-Agent-FromBase | Phase-1B | raw LFM2.5-8B-A1B | - | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-FromBase) |
| Fabliq-8B-Agent-FromBase-Reasoning | Phase-2B | Phase-1B | - | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-FromBase-Reasoning) |
| **Fabliq-8B-Agent-Mega** | Mega | raw LFM2.5-8B-A1B | 1.236 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega) |
| Fabliq-8B-Agent-Mega-Reasoning | Phase-2M | Mega | - | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-Reasoning) |
| Fabliq-8B-Agent-Mega-1ep | Ablation | raw LFM2.5-8B-A1B | 1.35 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-1ep) |
| Fabliq-8B-Agent-Mega-5ep | Ablation | raw LFM2.5-8B-A1B | 1.379 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-5ep) |
| Fabliq-8B-Agent-Mega-10ep | Ablation | raw LFM2.5-8B-A1B | - | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-10ep) |
| Fabliq-8B-Agent-Mega-lr5e7 | Ablation | raw LFM2.5-8B-A1B | 1.415 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-lr5e7) |
| 🏆 **Fabliq-8B-Agent-Mega-lr2e6** | Ablation | raw LFM2.5-8B-A1B | **1.169** | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-lr2e6) |

### 1.2B Class (small variant)

| Model | Base | Final Loss | HF |
| --- | --- | --- | --- |
| Fabliq-1.2B-Agent | LFM2.5-1.2B-Instruct | 1.55 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-1.2B-Agent) |

### GGUF quants (all variants)

Each model has 4 GGUF quants uploaded: Q4_K_M, Q5_K_M, Q6_K, Q8_0. Search `LLM-OS-Models/*-GGUF` on HuggingFace.

---

## 🎯 Key Findings

### LR Sweep (3 epoch, Mega dataset)
| LR | Final Loss |
| --- | --- |
| 5e-7 | 1.415 |
| 1e-6 | 1.236 |
| **2e-6** | **1.169** 🏆 |

**LR 2e-6 is optimal** for 4,328 rows × 3 epoch.

### Mega (single-pass) vs Phase-1+2 (multi-phase)
- Mega (all data 4,328 rows × 3 epoch): loss 1.236
- Phase-1 (Fable-5) + Phase-2 (WithinUs+Helio): two-stage curriculum

Single-pass Mega is simpler with comparable results.

### ToolBench Foundation Effect
Phase-1 (with ToolBench foundation) vs Phase-1B (raw LFM2.5 base). ToolBench pre-exposes the model to terminal tool usage.

---

## 📊 Data Inventory

| Dataset | Original | Preprocessed | Used |
| --- | --- | --- | --- |
| [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | 4,665 | 4,047 | All variants |
| WithinUsAI/claude_mythos_distilled_25k | 25,000 | 135 (SHA-256 dedup) | Phase-2/2B/2M, Mega |
| HelioAI/Fable-5-Distill-Reasoning-462x | 462 | 146 (Russian filter) | Phase-2/2B/2M, Mega |
| Glint-Research/Complete-FABLE.5-traces-2M | 2,006,487 | 0 (all dup of Fable-5) | Unused |
| armand0e/claude-fable-5-claude-code | 18,370 | raw events | Unused |
| lordx64/agentic-distill-fable-5-sft | 4,659 | dup of Glint | Unused |

**Total unique training data: 4,328 rows**

---

## 📂 File Inventory

See [DATA_SOURCES_20260623.ko.md](./DATA_SOURCES_20260623.ko.md) for detailed dataset breakdown and [FINAL_REPORT_20260624.ko.md](./FINAL_REPORT_20260624.ko.md) for complete results.

### Top-level scripts
- `build_fable5_to_lfm_sft.py` — Fable-5 → LFM JSONL (Phase-1 builder)
- `build_withinus_lfm_sft.py` — WithinUs → LFM JSONL (Phase-2 part 1)
- `build_helio_lfm_sft.py` — Helio → LFM JSONL (Phase-2 part 2)
- `build_fable5_2m_to_lfm_sft.py` — 2M traces → LFM JSONL (Phase-3 builder, all dup)
- `build_mega_combined_lfm_sft.py` — combine all sources with dedup

### Training scripts (`scripts/`)
- `run_fable5_full_sft_20260623.sh` — Phase-1
- `run_phase2_reasoning_sft_20260623.sh` — Phase-2
- `run_fable5_from_base_sft_20260623.sh` — Phase-1B (ablation)
- `run_phase2b_reasoning_sft_20260623.sh` — Phase-2B
- `run_mega_combined_sft_20260623.sh` — Mega
- `run_phase2m_mega_reasoning_sft_20260623.sh` — Phase-2M
- `run_mega_5ep.sh`, `run_mega_10ep.sh` — epoch ablations
- `run_remaining_ablations.sh` — LR scan + size sweep
- `run_size_sweep_20260623.sh` — small (1.2B) + XL (24B)
- `convert_fabliq_gguf.sh` — GGUF conversion with LFM tokenizer hash patch
- `run_fabliq_eval.sh` — eval runner
- `aggregate_eval_results.py` — results compiler

### Training code (`training/`)
- `train_lfm25_rlvr_json_sft.py` — main training (full SFT + LoRA, FSDP)
- `build_lfm25_agentic_sft.py`, `build_lfm25_rlvr_json_sft.py`, `mix_lfm25_sft_jsonl.py`

### Eval code (`eval_scripts/`)
- `eval_lfm25_agentic_vllm.py` — agentic tool-use eval
- `eval_lfm25_rlvr_retrieval_vllm.py` — retrieval-curation eval
- `simple_fabliq_eval.py` — simple direct eval (transformers backend)

---

## 🚀 Reproducing Training

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
source .liquid-sft-env/bin/activate

# Phase-1 (Fabliq-8B-Agent)
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_full_sft_20260623.sh

# Mega (all data)
RUN_NOW=1 bash fable_distillation/scripts/run_mega_combined_sft_20260623.sh

# Mega-lr2e6 (best loss)
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  .liquid-sft-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
    harness-1/training/train_lfm25_rlvr_json_sft.py \
    --model-path LiquidAI/LFM2.5-8B-A1B \
    --train-jsonl fable_distillation/datasets/mega_combined_lfm_sft_20260623.jsonl \
    --output-dir /path/to/output \
    --finetune-mode full --max-seq-length 8192 --epochs 3 --learning-rate 2e-6 \
    --per-device-train-batch-size 2 --gradient-accumulation-steps 4 \
    --target-modules q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate
```

---

## 📜 License

Apache 2.0 (inherited from LiquidAI LFM base).

## 🔗 Related Docs

- [Final Report](./FINAL_REPORT_20260624.ko.md) (Korean, comprehensive results)
- [Progress Timeline](./PROGRESS_20260623.ko.md)
- [Data Sources](./DATA_SOURCES_20260623.ko.md)
- [Next Steps](./NEXT_STEPS_20260623.ko.md)
- [TRAINING_USAGE.md](./TRAINING_USAGE.md)
- [Korean README](./README.ko.md)
- [GitHub: gyunggyung/fable5-to-lfm](https://github.com/gyunggyung/fable5-to-lfm)
