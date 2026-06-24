[**English**](README.md) | [**한국어**](README.ko.md)

# fable5-to-lfm

> **fable5-to-lfm** — Distilling [Fable-5](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) agentic terminal traces into LiquidAI's LFM family (1.2B / 2.6B / 8B / 24B-A2B) to build the **Fabliq** terminal-agent model line.

The name says what the project does: **Fable-5 → LFM**. 12 model variants trained across 3 model sizes, all published on HuggingFace.

Reference lineups: [Qwable-9B](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5), [Qwythos-9B](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M), [gemma-4-12B-agentic-fable5](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).

---

## Latest vLLM Benchmark (2026-06-24)

TB2-lite replay evaluation is now the primary local regression benchmark. It is not the public Terminal-Bench 2.1 harness; it is a 303-sample terminal next-action replay benchmark run with vLLM.

| Rank | Model | Score | Cmd F1 | First Cmd | Valid JSON |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **Fabliq-8B-Agent-Reasoning** (`phase2-reasoning`) | **51.59** | 0.5193 | 50.8% | 76.2% |
| 2 | ToolBench baseline | 51.46 | 0.5230 | 49.5% | 76.9% |
| 3 | Fabliq-8B-Agent (`phase1-fabliq`) | 49.31 | 0.5022 | 47.2% | 76.6% |
| 9 | Raw LFM2.5-8B-A1B base | 40.51 | 0.3992 | 41.9% | 59.1% |
| 13 | Mega-lr2e6, lowest train loss | 39.68 | 0.3929 | 40.6% | 59.1% |

Main takeaway: **ToolBench foundation + Fable terminal traces beat raw-base Mega training on terminal next-action behavior.** Lowest training loss did not produce the best agentic score.

Current 2026-06-24 experiment status:

- Running script: `scripts/run_glm52_chaser_mix_sft_20260624.sh`
- Training data: `datasets/glm52_chaser_terminal_toolmix_20260624.jsonl` (11,416 rows)
- Base model: `Fabliq-8B-Agent-Reasoning`
- Status: full SFT completed at `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624`; final model scored `51.13` on sharded vLLM TB2-lite.
- Goal: beat the current TB2-lite vLLM score `51.59`.
- Multi-model setup: Gemma 4 12B IT, Qwen3.5 9B, and DiffusionGemma 26B-A4B smoke runners are staged under `scripts/` and `configs/`.
- Qwen3.5-9B base vLLM fallback completed with score `36.75` on TB2-lite 32k sharded eval.
- DiffusionGemma dLLM base eval succeeded without Docker through the local `diffusiongemma-transformers-cu128` uv env plus Transformers `DiffusionGemmaForBlockDiffusion`. A decode/prompt-stripping bug was fixed after a 0-score first run; the corrected full run scored `25.12` with `97.88 tok/s` on the probe. The status is execution success but benchmark failure.
- GLM-5.2 chaser `checkpoint-1400` vLLM sharded TB2-lite eval completed at `50.56`, below both final `51.13` and the current best `51.59`.
- Qwen3.5-9B LoRA SFT300 was fixed with `simple-chatml` after the native Qwen chat template rejected assistant-only Fable replay samples, then restarted with `DDP_FIND_UNUSED_PARAMETERS=true` and a shared tokenized cache after the VLM text-only DDP unused-parameter failure. Current run id: `20260624_qwen35_9b_glm52_terminalmix_lora_sft300_chatml_ddptrue`.
- Docs: [current experiment status](./CURRENT_EXPERIMENT_STATUS_20260624.ko.md), [TB2 vLLM benchmark](./TB2_VLLM_BENCHMARK_20260624.ko.md), [GLM-5.2 chaser experiment](./GLM52_CHASER_EXPERIMENT_20260624.ko.md), [multi-model GLM-5.2 chaser plan](./MULTI_MODEL_GLM52_CHASER_PLAN_20260624.ko.md), [DiffusionGemma dLLM eval plan](./DIFFUSIONGEMMA_DLLM_EVAL_PLAN_20260624.ko.md)

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
| **Fabliq-8B-Agent-Mega-lr2e6** | Ablation | raw LFM2.5-8B-A1B | **1.169** | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-lr2e6) |

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

**LR 2e-6 is optimal by training loss** for 4,328 rows × 3 epoch. It is **not** the best TB2-lite agentic model.

### Mega (single-pass) vs Phase-1+2 (multi-phase)
- Mega (all data 4,328 rows × 3 epoch): loss 1.236
- Phase-1 (Fable-5) + Phase-2 (WithinUs+Helio): two-stage curriculum

Single-pass Mega is simpler by training setup, but it did not match the ToolBench + Fable curriculum on TB2-lite terminal replay.

### ToolBench Foundation Effect
Phase-1 (with ToolBench foundation) vs Phase-1B (raw LFM2.5 base). ToolBench pre-exposes the model to terminal tool usage.

TB2-lite vLLM evidence:

- ToolBench baseline: `51.46`
- Phase-1 Fabliq from ToolBench: `49.31`
- Phase-2 Reasoning from Phase-1: `51.59`
- Raw LFM2.5 base: `40.51`
- FromBase/Mega variants: mostly `39.68-41.47`

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
- `run_fabliq_tb2_vllm_20260624.sh` — TB2-lite vLLM wave 1
- `run_fabliq_tb2_vllm_wave2_20260624.sh` — TB2-lite vLLM wave 2
- `build_glm52_chaser_mix_20260624.py` — terminal/tool-call mix builder for GLM-5.2 chaser run
- `run_glm52_chaser_mix_sft_20260624.sh` — long full-SFT chaser run + automatic vLLM eval
- `build_hermes_agent_traces_mix_20260624.py` — Hermes Kimi/GLM agent traces → chat SFT JSONL
- `build_dllm_probe_prompts_20260624.py` — small long/code/tool-call prompt suite for dLLM behavior checks
- `vllm_prompt_probe.py` — vLLM long-output speed/shape probe
- `run_diffusiongemma_dllm_eval_20260624.sh` — DiffusionGemma base TB2-lite + long-output Transformers dLLM eval
- `run_multifamily_sft_smoke_20260624.sh` — Gemma/Qwen LoRA smoke SFT queue
- `run_diffusiongemma_fable_lora_20260624.sh` — DiffusionGemma NeMo AutoModel LoRA smoke queue
- `run_post_chaser_multimodel_queue_20260624.sh` — waits for the current chaser run, then launches DiffusionGemma eval/smoke before Gemma/Qwen jobs
- `setup_diffusiongemma_transformers_uv_20260624.sh` — isolated Docker-free uv env for DiffusionGemma Transformers eval with torch cu128
- `setup_diffusiongemma_vllm_uv_20260624.sh` — vLLM wheel env kept for future direct support; not used for the current no-Docker dLLM run
- `replay_eval_vllm.py`, `replay_metrics.py`, `summarize_replay_results.py` — local replay evaluator

### Training code (`training/`)
- `train_lfm25_rlvr_json_sft.py` — main training (full SFT + LoRA, FSDP)
- `train_multifamily_chat_sft.py` — generic Gemma/Qwen chat SFT trainer for LoRA smoke runs
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
- [TB2 vLLM Benchmark](./TB2_VLLM_BENCHMARK_20260624.ko.md)
- [GLM-5.2 Chaser Experiment](./GLM52_CHASER_EXPERIMENT_20260624.ko.md)
- [Multi-model GLM-5.2 Chaser Plan](./MULTI_MODEL_GLM52_CHASER_PLAN_20260624.ko.md)
- [DiffusionGemma dLLM Eval Plan](./DIFFUSIONGEMMA_DLLM_EVAL_PLAN_20260624.ko.md)
- [Current Experiment Status](./CURRENT_EXPERIMENT_STATUS_20260624.ko.md)
- [Progress Timeline](./PROGRESS_20260623.ko.md)
- [Data Sources](./DATA_SOURCES_20260623.ko.md)
- [Next Steps](./NEXT_STEPS_20260623.ko.md)
- [TRAINING_USAGE.md](./TRAINING_USAGE.md)
- [Korean README](./README.ko.md)
- [GitHub: gyunggyung/fable5-to-lfm](https://github.com/gyunggyung/fable5-to-lfm)
