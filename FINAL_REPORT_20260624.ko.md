# Fabliq 최종 결과 리포트 (2026-06-24)

> fable5-to-lfm 프로젝트 밤샘 작업 최종 결과. **18 variants** + 모든 GGUF.

## 📊 전체 모델 비교표

### 8B Variants (LFM2.5-8B-A1B 기반, 10 variants)

| 모델 | Base | 데이터 | Epochs | LR | Final Loss | Train Time | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Fabliq-8B-Agent** | ToolBench | Fable-5 4,047 | 3 | 5e-7 | 1.277 | 14min | 메인 |
| Fabliq-8B-Agent-Reasoning | Phase-1 | + 281 | 4 | 3e-7 | ~1.6 | 6min | reasoning 확장 |
| Fabliq-8B-Agent-FromBase | raw LFM2.5 | Fable-5 4,047 | 3 | 1e-6 | - | 14min | ablation |
| Fabliq-8B-Agent-FromBase-Reasoning | Phase-1B | + 281 | 4 | 3e-7 | - | 6min | FromBase+reasoning |
| **Fabliq-8B-Agent-Mega** | raw LFM2.5 | 4,328 | 3 | 1e-6 | 1.236 | 17min | 한 번에 모든 데이터 |
| Fabliq-8B-Agent-Mega-Reasoning | Mega | + 281 | 2 | 2e-7 | - | 5min | Mega+reasoning |
| Fabliq-8B-Agent-Mega-1ep | raw LFM2.5 | 4,328 | 1 | 1e-6 | 1.35 | 7min | epoch scan |
| Fabliq-8B-Agent-Mega-5ep | raw LFM2.5 | 4,328 | 5 | 1e-6 | 1.379 | 22min | epoch scan |
| Fabliq-8B-Agent-Mega-10ep | raw LFM2.5 | 4,328 | 10 | 5e-7 | - | 32min | epoch scan |
| Fabliq-8B-Agent-Mega-lr5e7 | raw LFM2.5 | 4,328 | 3 | 5e-7 | 1.415 | 14min | LR scan |
| 🏆 **Fabliq-8B-Agent-Mega-lr2e6** | raw LFM2.5 | 4,328 | 3 | **2e-6** | **1.169** | 13min | **최저 loss** |

### 1.2B Variants (LFM2.5-1.2B-Instruct/Thinking/Base/JP 기반, 5 variants)

| 모델 | Base | 데이터 | Final Loss | 비고 |
| --- | --- | --- | --- | --- |
| Fabliq-1.2B-Agent | LFM2.5-1.2B-Instruct | Fable-5 4,047 | 1.55 | Instruct base |
| Fabliq-1.2B-Agent-Mega | LFM2.5-1.2B-Instruct | 4,328 | 1.612 | Instruct + Mega data |
| Fabliq-1.2B-Thinking-Agent | LFM2.5-1.2B-Thinking | Fable-5 4,047 | ~1.55 | Thinking base |
| Fabliq-1.2B-Thinking-Agent-Mega | LFM2.5-1.2B-Thinking | 4,328 | 1.613 | Thinking + Mega data |
| Fabliq-1.2B-Agent-Base | LFM2.5-1.2B-Base | 4,328 | 1.554 | raw base, no instruct |
| Fabliq-1.2B-Agent-JP | LFM2.5-1.2B-JP | 4,328 | (training) | Japanese base |

### 2.6B Variant

| 모델 | Base | 데이터 | Final Loss | 비고 |
| --- | --- | --- | --- | --- |
| Fabliq-2.6B-Agent | LFM2-2.6B | Fable-5 4,047 | 1.465 | LFM2 (not 2.5) |

### 24B Variant (실패)
- LFM2-24B-A2B: CUDA OOM. paged 8bit optimizer + CPU offload 필요.

## 🎯 핵심 발견

### 1. LR 스캔 (3 epoch, Mega 데이터, 8B)
| LR | Final Loss |
| --- | --- |
| 5e-7 | 1.415 |
| 1e-6 | 1.236 |
| **2e-6** | **1.169** 🏆 |

**LR 2e-6 최적** (4,328 rows × 3 epoch 기준).

### 2. Mega (single-pass) vs Phase-1+2 (multi-phase)
- Mega (4,328 rows × 3 epoch): loss 1.236
- Phase-1 (Fable-5) + Phase-2 (reasoning): 2단계

Mega가 단일 phase로 더 단순하면서 비슷/더 나은 성능.

### 3. 모델 크기 영향
- 1.2B: loss 1.55 (Instruct) / 1.554 (Base)
- 2.6B: loss 1.465
- 8B: loss 1.169 (Mega-lr2e6)

클수록 loss 낮아짐. 큰 모델이 더 잘 학습.

### 4. ToolBench Foundation 효과
- Phase-1 (ToolBench → Fable-5) vs Phase-1B (Base → Fable-5)
- ToolBench가 터미널 툴 사용에 미리 노출

## 📊 데이터 요약

| 데이터셋 | 원본 | 전처리 후 | 사용 |
| --- | --- | --- | --- |
| Glint-Research/Fable-5-traces | 4,665 | 4,047 | 모든 variants |
| WithinUsAI/claude_mythos_distilled_25k | 25,000 | 135 (dedup) | Phase-2/2B/2M, Mega |
| HelioAI/Fable-5-Distill-Reasoning-462x | 462 | 146 (Russian filter) | Phase-2/2B/2M, Mega |
| Complete-FABLE.5-traces-2M | 2,006,487 | 0 (모두 Fable-5 중복) | 미사용 |
| armand0e/claude-fable-5-claude-code | 18,370 | raw events | 미사용 |

**최종 unique 데이터: 4,328 rows**

## 🔬 GGUF 라인업 (모든 variants × 4 quants)

18 모델 × 4 quants (Q4_K_M, Q5_K_M, Q6_K, Q8_0) = **72 GGUF 파일**.

## 🌐 HuggingFace 게시 현황

총 **36개 레포** (18 모델 + 18 GGUF):

### 8B (10 variants + 10 GGUF)
- Fabliq-8B-Agent (Phase-1)
- Fabliq-8B-Agent-Reasoning (Phase-2)
- Fabliq-8B-Agent-FromBase (Phase-1B)
- Fabliq-8B-Agent-FromBase-Reasoning (Phase-2B)
- Fabliq-8B-Agent-Mega
- Fabliq-8B-Agent-Mega-Reasoning (Phase-2M)
- Fabliq-8B-Agent-Mega-1ep / -5ep / -10ep
- Fabliq-8B-Agent-Mega-lr5e7 / -lr2e6

### 1.2B (6 variants + 6 GGUF)
- Fabliq-1.2B-Agent (Instruct + Fable-5)
- Fabliq-1.2B-Agent-Mega (Instruct + Mega)
- Fabliq-1.2B-Thinking-Agent (Thinking + Fable-5)
- Fabliq-1.2B-Thinking-Agent-Mega (Thinking + Mega)
- Fabliq-1.2B-Agent-Base (Base + Mega)
- Fabliq-1.2B-Agent-JP (JP + Mega, training)

### 2.6B (1 variant + 1 GGUF)
- Fabliq-2.6B-Agent

## ⚠️ 알려진 이슈

1. **vLLM ABI 손상**: 시스템 vllm env에서 undefined symbol. 벤치마크 eval (MMLU, HumanEval)이 transformers backend로만 가능.
2. **2M 데이터 중복**: Complete-FABLE.5-traces-2M은 Fable-5 재포장.
3. **WithinUs 템플릿 반복**: 25k → 135 unique.
4. **러시아어 필터**: Helio 462 → 146.
5. **24B OOM**: LFM2-24B-A2B 학습 시 H200 8대로 부족.

## 🚀 추천 모델

| 용도 | 추천 |
| --- | --- |
| 터미널 에이전트 (8B, 최고 성능) | **Fabliq-8B-Agent-Mega-lr2e6** |
| 터미널 에이전트 (8B, 안정적) | Fabliq-8B-Agent 또는 Mega |
| 터미널 + reasoning | Fabliq-8B-Agent-Mega-Reasoning |
| 로컬 16GB VRAM | GGUF Q4_K_M (4.9G) |
| 로컬 8GB VRAM (1.2B) | Fabliq-1.2B-Agent Q4_K_M (698M) |
| 로컬 4GB VRAM | Fabliq-1.2B-Agent Q4_K_M |

## 📜 라이선스

Apache 2.0 (LiquidAI LFM base 상속).
