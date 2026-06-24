[**English**](README.md) | [**한국어**](README.ko.md)

# fable5-to-lfm

> **fable5-to-lfm** — [Fable-5](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) agentic 터미널 트레이스를 LiquidAI LFM family (1.2B / 2.6B / 8B / 24B-A2B) 에 증류하여 터미널 에이전트 **Fabliq** 모델 라인 구축.

이름 그대로 **Fable-5 → LFM** 방향의 증류. 3가지 모델 크기에서 12개 variant 학습, 모두 HuggingFace에 게시.

참고 라인: [Qwable-9B](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5), [Qwythos-9B](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M), [gemma-4-12B-agentic-fable5](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).

---

## 최신 vLLM 벤치마크 (2026-06-24)

현재 로컬 회귀 기준은 TB2-lite replay 평가다. public Terminal-Bench 2.1 harness와 같은 평가는 아니고, 303개 터미널 next-action prompt를 vLLM으로 빠르게 돌려 command/action 품질을 보는 내부 벤치마크다.

| 순위 | 모델 | Score | Cmd F1 | First Cmd | Valid JSON |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **Fabliq-8B-Agent-Reasoning** (`phase2-reasoning`) | **51.59** | 0.5193 | 50.8% | 76.2% |
| 2 | ToolBench baseline | 51.46 | 0.5230 | 49.5% | 76.9% |
| 3 | Fabliq-8B-Agent (`phase1-fabliq`) | 49.31 | 0.5022 | 47.2% | 76.6% |
| 9 | Raw LFM2.5-8B-A1B base | 40.51 | 0.3992 | 41.9% | 59.1% |
| 13 | Mega-lr2e6, 최저 train loss | 39.68 | 0.3929 | 40.6% | 59.1% |

핵심 결론: **ToolBench foundation + Fable terminal traces 조합이 raw-base Mega 학습보다 터미널 next-action에서 훨씬 낫다.** 최저 train loss가 최고 agentic 점수를 의미하지 않았다.

현재 2026-06-24 실험 상태:

- 실행 스크립트: `scripts/run_glm52_chaser_mix_sft_20260624.sh`
- 학습 데이터: `datasets/glm52_chaser_terminal_toolmix_20260624.jsonl` (11,416 rows)
- 베이스 모델: `Fabliq-8B-Agent-Reasoning`
- 상태: full SFT 완료. 출력 모델은 `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624`; final model은 sharded vLLM TB2-lite에서 `51.13`.
- 목표: 현재 TB2-lite vLLM 점수 `51.59` 돌파.
- 멀티모델 준비: Gemma 4 12B IT, Qwen3.5 9B, DiffusionGemma 26B-A4B smoke runner를 `scripts/`, `configs/` 아래에 준비
- Qwen3.5-9B base vLLM fallback은 TB2-lite 32k sharded 평가에서 `36.75`로 완료.
- DiffusionGemma 우선순위: dLLM base 평가는 Docker 없이 로컬 `diffusiongemma-transformers-cu128` uv 가상환경의 Transformers `DiffusionGemmaForBlockDiffusion` backend로 성공. 첫 0점 run 이후 decode/prompt stripping 버그를 고쳤고, 수정 full run은 score `25.12`, probe `97.88 tok/s`. 결론은 실행 성공/성능 실패다.
- GLM-5.2 chaser `checkpoint-1400` vLLM sharded TB2-lite 평가는 완료. score `50.56`으로 final `51.13`보다 낮아 현 1위 `51.59`는 못 넘었다.
- Qwen3.5-9B LoRA SFT300은 train loss `0.6287`로 완료했고 adapter 병합도 끝났다. 다만 merged checkpoint가 현재 vLLM 로드에서 VLM/text wrapper weight-key mismatch로 실패해 TB2-lite 점수는 아직 없다. 이 export 문제는 DiffusionGemma 1차 학습 뒤 처리한다.
- 다음 DiffusionGemma run은 raw next-action이 아니라 강점 태스크로 바꾼다. Fable terminal/tool-call trace를 유지하고 structured JSON/tool-call repair를 추가하며, `scripts/build_diffusiongemma_strength_mix_20260624.py`와 `configs/diffusiongemma_26b_a4b_strength_lora_20260624.yaml`을 사용한다. 첫 retry는 step 199까지 정상 학습 후 NeMo PEFT optimizer safetensors checkpoint 버그로 실패했고, 이제 `scripts/diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py`로 adapter checkpoint는 유지하고 optimizer checkpoint만 건너뛴다.
- 문서: [현재 실험 상태](./CURRENT_EXPERIMENT_STATUS_20260624.ko.md), [TB2 vLLM benchmark](./TB2_VLLM_BENCHMARK_20260624.ko.md), [GLM-5.2 chaser experiment](./GLM52_CHASER_EXPERIMENT_20260624.ko.md), [multi-model GLM-5.2 chaser plan](./MULTI_MODEL_GLM52_CHASER_PLAN_20260624.ko.md), [DiffusionGemma dLLM eval plan](./DIFFUSIONGEMMA_DLLM_EVAL_PLAN_20260624.ko.md), [DiffusionGemma strength tasks](./DIFFUSIONGEMMA_STRENGTH_TASKS_20260624.ko.md)

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
| **Fabliq-8B-Agent-Mega-lr2e6** | Ablation | raw LFM2.5-8B-A1B | **1.169** | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega-lr2e6) |

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

**LR 2e-6은 train loss 기준 최적** (4,328 rows × 3 epoch 기준). TB2-lite agentic 성능 기준 1위는 아니다.

### Mega (한 번에) vs Phase-1+2 (단계적)
- Mega (모든 데이터 4,328 rows × 3 epoch): loss 1.236
- Phase-1 (Fable-5) + Phase-2 (WithinUs+Helio): 2단계 curriculum

단일 phase Mega는 학습 구성은 단순하지만, TB2-lite 터미널 replay에서는 ToolBench + Fable curriculum을 따라오지 못했다.

### ToolBench Foundation 효과
- Phase-1 (ToolBench 기반) vs Phase-1B (raw LFM2.5)
- ToolBench가 터미널 툴 사용에 미리 노출되어 foundation 역할

TB2-lite vLLM 근거:

- ToolBench baseline: `51.46`
- Phase-1 Fabliq from ToolBench: `49.31`
- Phase-2 Reasoning from Phase-1: `51.59`
- Raw LFM2.5 base: `40.51`
- FromBase/Mega variants: 대체로 `39.68-41.47`

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
- `run_fabliq_tb2_vllm_20260624.sh` — TB2-lite vLLM wave 1
- `run_fabliq_tb2_vllm_wave2_20260624.sh` — TB2-lite vLLM wave 2
- `build_glm52_chaser_mix_20260624.py` — GLM-5.2 추격용 terminal/tool-call mix builder
- `run_glm52_chaser_mix_sft_20260624.sh` — 장시간 full-SFT chaser run + 자동 vLLM 평가
- `build_hermes_agent_traces_mix_20260624.py` — Hermes Kimi/GLM agent traces → chat SFT JSONL 변환
- `build_dllm_probe_prompts_20260624.py` — dLLM 긴 출력/code/tool-call 확인용 작은 prompt suite
- `vllm_prompt_probe.py` — vLLM 긴 출력 속도/형태 probe
- `run_diffusiongemma_dllm_eval_20260624.sh` — DiffusionGemma base TB2-lite + long-output Transformers dLLM 평가
- `build_diffusiongemma_strength_mix_20260624.py` — DiffusionGemma용 Fable + structured repair SFT mix 생성
- `diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py` — PEFT optimizer checkpoint 직렬화는 건너뛰고 LoRA adapter 저장은 유지하는 로컬 NeMo launcher wrapper
- `run_diffusiongemma_strength_lora_20260624.sh` — DiffusionGemma 강점 태스크용 NeMo AutoModel LoRA runner
- `run_multifamily_sft_smoke_20260624.sh` — Gemma/Qwen LoRA smoke SFT queue
- `merge_multifamily_lora_for_vllm.py` — Gemma/Qwen 계열 HF checkpoint를 vLLM 평가 전에 병합하는 generic PEFT LoRA merge helper
- `watch_qwen35_lora_merge_eval_20260624.sh` — 현재 Qwen3.5 LoRA SFT run을 기다렸다가 `final_lora` 병합 후 8-shard TB2-lite vLLM 평가를 실행하는 watcher
- `run_diffusiongemma_fable_lora_20260624.sh` — DiffusionGemma NeMo AutoModel LoRA smoke queue
- `run_post_chaser_multimodel_queue_20260624.sh` — 현재 chaser run 종료 후 DiffusionGemma 평가/스모크를 먼저 실행하고 Gemma/Qwen을 이어서 실행하는 watcher
- `setup_diffusiongemma_transformers_uv_20260624.sh` — DiffusionGemma Transformers 평가용 torch cu128 isolated uv env 준비 스크립트
- `setup_diffusiongemma_vllm_uv_20260624.sh` — 향후 vLLM 직접 지원 대비용 env 스크립트. 현재 무도커 dLLM run에는 쓰지 않음
- `replay_eval_vllm.py`, `replay_metrics.py`, `summarize_replay_results.py` — local replay evaluator

### 학습 코드 (`training/`)
- `train_lfm25_rlvr_json_sft.py` — 메인 학습 (full SFT + LoRA, FSDP)
- `train_multifamily_chat_sft.py` — Gemma/Qwen 계열 LoRA smoke용 generic chat SFT trainer
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
- [TB2 vLLM 벤치마크](./TB2_VLLM_BENCHMARK_20260624.ko.md)
- [GLM-5.2 추격 실험](./GLM52_CHASER_EXPERIMENT_20260624.ko.md)
- [Multi-model GLM-5.2 추격 계획](./MULTI_MODEL_GLM52_CHASER_PLAN_20260624.ko.md)
- [DiffusionGemma dLLM 평가 계획](./DIFFUSIONGEMMA_DLLM_EVAL_PLAN_20260624.ko.md)
- [현재 실험 상태](./CURRENT_EXPERIMENT_STATUS_20260624.ko.md)
- [진행 상황 타임라인](./PROGRESS_20260623.ko.md)
- [데이터 소스 상세](./DATA_SOURCES_20260623.ko.md)
- [후속 작업 계획](./NEXT_STEPS_20260623.ko.md)
- [TRAINING_USAGE.md](./TRAINING_USAGE.md)
- [영문 README](./README.md)
- [GitHub: gyunggyung/fable5-to-lfm](https://github.com/gyunggyung/fable5-to-lfm)
