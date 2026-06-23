[**English**](README.md) | [**한국어**](README.ko.md)

# fable5-to-lfm

> **fable5-to-lfm** — [Fable-5](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) agentic 터미널 트레이스를 LiquidAI LFM family (1.2B / 2.6B / 8B / 24B-A2B) 에 증류하여 터미널 에이전트 **Fabliq** 모델 라인 구축.

이름 그대로 **Fable-5 → LFM** 방향의 증류. 3가지 모델 크기에서 12개 variant 학습, 모두 HuggingFace에 게시.

참고 라인: [Qwable-9B](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5), [Qwythos-9B](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M), [gemma-4-12B-agentic-fable5](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).

---

## 🌊 Fabliq 모델 라인업 (12 variants + 12 GGUF = 24 레포)

### 8B 클래스 (10 variants)

| 모델 | 단계 | 베이스 | Final Loss | HF |
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

### 1.2B 클래스 (작은 variant)

| 모델 | 베이스 | Final Loss | HF |
| --- | --- | --- | --- |
| Fabliq-1.2B-Agent | LFM2.5-1.2B-Instruct | 1.55 | [link](https://huggingface.co/LLM-OS-Models/Fabliq-1.2B-Agent) |

### GGUF 양자화 (모든 variants)

각 모델마다 4개 GGUF quant 업로드: Q4_K_M, Q5_K_M, Q6_K, Q8_0. HuggingFace에서 `LLM-OS-Models/*-GGUF` 검색.

---

## 🎯 주요 발견

### LR 스캔 (3 epoch, Mega 데이터)
| LR | Final Loss |
| --- | --- |
| 5e-7 | 1.415 |
| 1e-6 | 1.236 |
| **2e-6** | **1.169** 🏆 |

**LR 2e-6이 최적** (4,328 rows × 3 epoch 기준).

### Mega (한 번에) vs Phase-1+2 (단계적)
- Mega (모든 데이터 4,328 rows × 3 epoch): loss 1.236
- Phase-1 (Fable-5) + Phase-2 (WithinUs+Helio): 2단계 curriculum

단일 phase Mega가 더 단순하면서 비슷한 성능.

### ToolBench Foundation 효과
- Phase-1 (ToolBench 기반) vs Phase-1B (raw LFM2.5)
- ToolBench가 터미널 툴 사용에 미리 노출되어 foundation 역할

---

## 📊 데이터 인벤토리

| 데이터셋 | 원본 | 전처리 후 | 사용처 |
| --- | --- | --- | --- |
| [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | 4,665 | 4,047 | 모든 variants |
| WithinUsAI/claude_mythos_distilled_25k | 25,000 | 135 (SHA-256 dedup) | Phase-2/2B/2M, Mega |
| HelioAI/Fable-5-Distill-Reasoning-462x | 462 | 146 (Russian filter) | Phase-2/2B/2M, Mega |
| Glint-Research/Complete-FABLE.5-traces-2M | 2,006,487 | 0 (모두 Fable-5 중복) | 미사용 |
| armand0e/claude-fable-5-claude-code | 18,370 | (raw events, 미처리) | 미사용 |
| lordx64/agentic-distill-fable-5-sft | 4,659 | (Glint 중복) | 미사용 |

**총 unique 학습 데이터: 4,328 rows**

---

## 📂 파일 구성

데이터셋 상세는 [DATA_SOURCES_20260623.ko.md](./DATA_SOURCES_20260623.ko.md), 전체 결과는 [FINAL_REPORT_20260624.ko.md](./FINAL_REPORT_20260624.ko.md) 참조.

### 최상위 스크립트
- `build_fable5_to_lfm_sft.py` — Fable-5 → LFM JSONL (Phase-1 빌더)
- `build_withinus_lfm_sft.py` — WithinUs → LFM JSONL (Phase-2 1/2)
- `build_helio_lfm_sft.py` — Helio → LFM JSONL (Phase-2 2/2)
- `build_fable5_2m_to_lfm_sft.py` — 2M traces → LFM JSONL (모두 중복)
- `build_mega_combined_lfm_sft.py` — 모든 소스 통합 (dedup)

### 학습 스크립트 (`scripts/`)
- `run_fable5_full_sft_20260623.sh` — Phase-1
- `run_phase2_reasoning_sft_20260623.sh` — Phase-2
- `run_fable5_from_base_sft_20260623.sh` — Phase-1B (ablation)
- `run_phase2b_reasoning_sft_20260623.sh` — Phase-2B
- `run_mega_combined_sft_20260623.sh` — Mega
- `run_phase2m_mega_reasoning_sft_20260623.sh` — Phase-2M
- `run_mega_5ep.sh`, `run_mega_10ep.sh` — epoch ablations
- `run_remaining_ablations.sh` — LR 스캔 + size sweep
- `run_size_sweep_20260623.sh` — 작은 (1.2B) + XL (24B)
- `convert_fabliq_gguf.sh` — GGUF 변환 (LFM tokenizer patch 포함)
- `run_fabliq_eval.sh` — eval runner
- `aggregate_eval_results.py` — 결과 취합

### 학습 코드 (`training/`)
- `train_lfm25_rlvr_json_sft.py` — 메인 학습 (full SFT + LoRA, FSDP)
- 그 외 데이터 빌더, JSONL 믹서

### 평가 코드 (`eval_scripts/`)
- `eval_lfm25_agentic_vllm.py` — agentic tool-use 평가
- `eval_lfm25_rlvr_retrieval_vllm.py` — retrieval-curation 평가
- `simple_fabliq_eval.py` — 단순 직접 평가 (transformers backend)

---

## 🚀 학습 재현

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
source .liquid-sft-env/bin/activate

# Phase-1 (Fabliq-8B-Agent)
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_full_sft_20260623.sh

# Mega (모든 데이터)
RUN_NOW=1 bash fable_distillation/scripts/run_mega_combined_sft_20260623.sh

# Mega-lr2e6 (최저 loss)
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

## 📜 라이선스

Apache 2.0 (LiquidAI LFM base 상속).

## 🔗 관련 문서

- [최종 리포트](./FINAL_REPORT_20260624.ko.md)
- [진행 상황 타임라인](./PROGRESS_20260623.ko.md)
- [데이터 소스 상세](./DATA_SOURCES_20260623.ko.md)
- [후속 작업 계획](./NEXT_STEPS_20260623.ko.md)
- [TRAINING_USAGE.md](./TRAINING_USAGE.md)
- [영문 README](./README.md)
- [GitHub: gyunggyung/fable5-to-lfm](https://github.com/gyunggyung/fable5-to-lfm)
