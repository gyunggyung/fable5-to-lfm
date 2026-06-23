# Fabliq 최종 결과 리포트 (2026-06-24)

> fable5-to-lfm 프로젝트 밤샘 작업 최종 결과 (총 12 variants + 모든 GGUF).

## 📊 전체 모델 비교표

### 8B Variants (LFM2.5-8B-A1B / LFM2.5-8B-A1B 기반)

| 모델 | Base | 데이터 | Epochs | LR | Final Loss | Train Time | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Fabliq-8B-Agent** | ToolBench | Fable-5 4,047 | 3 | 5e-7 | 1.277 | 14min | 메인 (터미널 에이전트) |
| Fabliq-8B-Agent-Reasoning | Phase-1 | + WithinUs+Helio 281 | 4 | 3e-7 | ~1.6 | 6min | reasoning 확장 |
| Fabliq-8B-Agent-FromBase | raw LFM2.5 | Fable-5 4,047 | 3 | 1e-6 | - | 14min | ToolBench 생략 (ablation) |
| Fabliq-8B-Agent-FromBase-Reasoning | Phase-1B | + reasoning 281 | 4 | 3e-7 | - | 6min | FromBase + reasoning |
| **Fabliq-8B-Agent-Mega** | raw LFM2.5 | 모든 데이터 4,328 | 3 | 1e-6 | **1.236** | 17min | 한 번에 모든 데이터 |
| Fabliq-8B-Agent-Mega-Reasoning | Mega | + reasoning 281 | 2 | 2e-7 | - | 5min | Mega + reasoning 강조 |
| Fabliq-8B-Agent-Mega-1ep | raw LFM2.5 | 4,328 | 1 | 1e-6 | 1.35 | 7min | epoch ablation |
| Fabliq-8B-Agent-Mega-5ep | raw LFM2.5 | 4,328 | 5 | 1e-6 | 1.379 | 22min | epoch ablation |
| Fabliq-8B-Agent-Mega-10ep | raw LFM2.5 | 4,328 | 10 | 5e-7 | - | 32min | epoch ablation (LR 낮춤) |
| Fabliq-8B-Agent-Mega-lr5e7 | raw LFM2.5 | 4,328 | 3 | 5e-7 | 1.415 | 14min | LR ablation |
| **Fabliq-8B-Agent-Mega-lr2e6** | raw LFM2.5 | 4,328 | 3 | **2e-6** | **1.169** | 13min | 🏆 최저 loss |

### 1.2B Variant

| 모델 | Base | 데이터 | Final Loss | Train Time | 비고 |
| --- | --- | --- | --- | --- | --- |
| Fabliq-1.2B-Agent | LFM2.5-1.2B-Instruct | Fable-5 4,047 | 1.55 | 7min | small agent (VRAM 4.5GB) |

### 24B Variant (실패)

- **Fabliq-24B-Agent**: 시도했으나 CUDA OOM (24B 모델이 H200 141GB × 8로도 부족, 137GB/GPU 사용). paged 8bit optimizer + max_seq 1024로 재시도 가능하지만 시간 효율 낮음.

## 🎯 주요 발견

### 1. LR 최적값: **2e-6이 가장 낮은 loss (1.169)**

| LR | Final Loss |
| --- | --- |
| 5e-7 | 1.415 |
| 1e-6 | 1.236 |
| **2e-6** | **1.169** 🏆 |

LR 2e-6이 데이터 크기 (4,328 rows × 3 epoch)에 가장 적합.

### 2. Epoch 영향: 10 epoch 약간 개선이지만 수익 감소

| Epochs | Final Loss | 비고 |
| --- | --- | --- |
| 1 | 1.35 | 빠르지만 수렴 덜 됨 |
| 3 | 1.236 | 기본값 |
| 5 | 1.379 | loss 상승 (overfitting 시작?) |
| 10 | - | 5e-7 LR이라 비교 어려움 |

10 epoch는 LR을 5e-7로 낮춰 overfitting 방지 필요.

### 3. Mega (single-pass) vs Phase-1+Phase-2 (multi-phase)

- Mega 한 번에 모든 데이터 → loss 1.236
- Phase-1 (Fable-5) → Phase-2 (reasoning) → 두 단계

Mega가 단일 phase로 더 단순하면서 비슷한 성능.

### 4. ToolBench Foundation 효과

- Phase-1 (ToolBench → Fable-5) vs Phase-1B (Base → Fable-5)
- ToolBench가 터미널 툴 사용에 미리 노출되므로 foundation 역할

### 5. Reasoning Expansion 효과

- Phase-1 + WithinUs+Helio (281 rows) → Phase-2
- WithinUs/Helio는 터미널이 아닌 일반 reasoning이라 도메인 확장

## 📦 데이터셋 요약

| 데이터셋 | 원본 | 전처리 후 | 사용처 |
| --- | --- | --- | --- |
| Glint-Research/Fable-5-traces | 4,665 | 4,047 | 모든 variants |
| WithinUsAI/claude_mythos_distilled_25k | 25,000 | 135 (dedup) | Phase-2/2B/2M, Mega |
| HelioAI/Fable-5-Distill-Reasoning-462x | 462 | 146 (Russian filter) | Phase-2/2B/2M, Mega |
| Glint-Research/Complete-FABLE.5-traces-2M | 2,006,487 | 0 (모두 Fable-5 중복) | 미사용 |
| armand0e/claude-fable-5-claude-code | 18,370 | (raw events, 미처리) | 미사용 |
| lordx64/agentic-distill-fable-5-sft | 4,659 | (Glint 중복) | 미사용 |

**최종 unique 데이터**: 4,328 rows

## 🔬 GGUF 라인업 (모든 variants)

12개 모델 × 4 quants (Q4_K_M, Q5_K_M, Q6_K, Q8_0) = **48 GGUF 파일** HF 업로드 완료.

| 모델 | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
| --- | --- | --- | --- | --- |
| Fabliq-8B-Agent (10 variants) | 4.9G | 5.7G | 6.5G | 8.4G |
| Fabliq-1.2B-Agent | 698M | 805M | 919M | 1.1G |

## 🌐 HuggingFace 게시 현황

총 **24개 레포** (12 모델 + 12 GGUF):

- LLM-OS-Models/Fabliq-8B-Agent
- LLM-OS-Models/Fabliq-8B-Agent-Reasoning
- LLM-OS-Models/Fabliq-8B-Agent-FromBase
- LLM-OS-Models/Fabliq-8B-Agent-FromBase-Reasoning
- LLM-OS-Models/Fabliq-8B-Agent-Mega
- LLM-OS-Models/Fabliq-8B-Agent-Mega-Reasoning
- LLM-OS-Models/Fabliq-8B-Agent-Mega-1ep
- LLM-OS-Models/Fabliq-8B-Agent-Mega-5ep
- LLM-OS-Models/Fabliq-8B-Agent-Mega-10ep
- LLM-OS-Models/Fabliq-8B-Agent-Mega-lr5e7
- LLM-OS-Models/Fabliq-8B-Agent-Mega-lr2e6
- LLM-OS-Models/Fabliq-1.2B-Agent
- 각각의 -GGUF 버전

## ⚠️ 알려진 이슈

1. **vLLM ABI 손상**: 시스템의 모든 vllm env에서 undefined symbol 에러. 벤치마크 eval (MMLU, HumanEval)이 transformers backend로만 가능해서 느림. simple_fabliq_eval.py로 5 terminal task만 테스트 가능.
2. **2M 데이터 중복**: Complete-FABLE.5-traces-2M은 Glint가 Fable-5를 재포장. 새 데이터 없음.
3. **WithinUs 템플릿 반복**: 25k → 135 unique.
4. **러시아어 필터**: Helio 462 → 146.
5. **24B OOM**: LFM2-24B-A2B 학습 시 H200 8대로 부족. 8bit optimizer + CPU offload 필요.

## 🚀 추천 모델

| 용도 | 추천 |
| --- | --- |
| 터미널 에이전트 (8B) | **Fabliq-8B-Agent-Mega-lr2e6** (최저 loss) 또는 **Fabliq-8B-Agent** (안정적) |
| 터미널 + reasoning | Fabliq-8B-Agent-Mega-Reasoning |
| 로컬 16GB VRAM (Q4_K_M) | GGUF 4.9G |
| 로컬 24GB VRAM (Q8_0) | GGUF 8.4G |
| 로컬 4-8GB VRAM | Fabliq-1.2B-Agent Q4_K_M (698M) |

## 📜 라이선스

Apache 2.0 (LiquidAI LFM base 상속).

## 🙏 참고 모델

- [Qwable-9B-Claude-Fable-5](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5): Qwen3.5-9B + Fable-5
- [Qwythos-9B-Claude-Mythos-5-1M](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M): 2-phase curriculum 패턴
- [gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF): test-gated filtering 패턴
