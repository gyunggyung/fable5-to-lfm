# fable5-to-lfm 결과 종합 (2026-06-24)

> 밤샘 학습 + 평가 최종 결과. **18 variants + 36 HF 레포 + 실제 성능 평가 포함.**

---

## 🎯 TL;DR

- **18개 variant 학습**: 8B×10, 1.2B×6, 2.6B×1 (24B는 OOM으로 스킵)
- **터미널 에이전트 성능 평가 (5 task × 모델)**: phase1-fabliq / phase2-reasoning / mega-combined 3종이 **80% 정확도** (base 60% 대비 +20%)
- **학습 loss 최저**: Mega-lr2e6 (1.169) — 하지만 실제 터미널 task 성능은 phase1/2/mega가 더 좋음
- **ToolBench foundation 효과 확인**: phase1 (60% tool_call) ≫ phase1b (20%)
- **36개 HF 레포 게시 완료**: 18 모델 + 18 GGUF (Q4_K_M/Q5_K_M/Q6_K/Q8_0)

---

## 📊 터미널 에이전트 성능 평가 (8B + 1.2B)

각 모델에 5개 터미널 task (ls/find/read/grep/write/edit) 프롬프트 → tool_call 정확도 측정.

| 모델 | Tool Call 사용률 | Think Block 사용률 | Tool 정확도 | 평균 지연 | 비고 |
| --- | --- | --- | --- | --- | --- |
| base LFM2.5-8B-A1B | 40% | 60% | 60% | 41.2s | baseline |
| 🥇 **phase1-fabliq** | 60% | 60% | **80%** ✅ | 38.9s | ToolBench 기반, 안정적 |
| phase1b-frombase | 20% | 20% | 60% | 41.6s | ToolBench 생략 → tool_call 약함 |
| 🥇 **phase2-reasoning** | 40% | **80%** | **80%** ✅ | 39.9s | reasoning 강화 |
| 🥇 **mega-combined** | 40% | 60% | **80%** ✅ | 41.5s | 한 번에 모든 데이터 |
| mega-lr2e6 (loss 최저) | 60% | 60% | 40% ⚠️ | 38.9s | 학습 loss와 task 성능 비례 X |
| mega-lr5e7 | 0% | 40% | 60% | 41.8s | 너무 낮은 LR → 수렴 덜 됨 |
| fabliq-1.2b | 40% | 0% | 20% | 8.9s | 1.2B는 8B 대비 현저히 낮음 |

### 핵심 인사이트

1. **터미널 task 정확도 80% 달성 (phase1/2/mega 3종)**: base 60% 대비 +20% 향상
2. **ToolBench foundation 결정적**: phase1 (60% tool_call) ≫ phase1b (20%, ToolBench 생략)
3. **Reasoning 강화 효과**: phase2는 think_block 80%로 reasoning 폭 확장
4. **학습 loss ≠ task 성능**: mega-lr2e6 (loss 1.169) 오히려 40% 정확도 → overfitting 가능성
5. **1.2B 한계**: 8B 대비 tool 정확도 현저히 낮음. 터미널 에이전트는 8B 이상 필요

---

## 📈 학습 Loss 비교 (8B Mega variants, 3 epoch)

| LR | Final Loss | Tool 정확도 |
| --- | --- | --- |
| 5e-7 | 1.415 | 60% |
| 1e-6 | 1.236 | 80% |
| **2e-6** | **1.169** (최저) | 40% ⚠️ |

**LR 1e-6이 실제 task 성능 최적**. LR 2e-6은 loss는 낮지만 overfitting으로 추정.

### Epoch 영향 (LR 1e-6)

| Epochs | Final Loss |
| --- | --- |
| 1 | 1.35 |
| **3** | **1.236** ✅ |
| 5 | 1.379 (overfitting 시작?) |
| 10 | - (LR 5e-7로 비교 불가) |

3 epoch이 loss 대비 시간 효율 최적.

---

## 🌊 Mega (single-pass) vs Phase-1+2 (multi-stage)

| 접근법 | 설명 | 결과 |
| --- | --- | --- |
| **Mega** | 모든 데이터 4,328 rows를 한 번에 3 epoch SFT | loss 1.236, **정확도 80%** ✅ |
| Phase-1 → Phase-2 | Fable-5 먼저 → reasoning 추가 2단계 | loss ~1.6, **정확도 80%** ✅ |

**Mega가 더 단순하면서 같은 정확도**. 추천: Mega로 단일 학습.

---

## 🏆 추천 모델 (용도별)

| 용도 | 추천 모델 | HF 링크 |
| --- | --- | --- |
| **터미널 에이전트 (8B, 안정적)** | Fabliq-8B-Agent (Phase-1) | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent) |
| **터미널 + reasoning (8B)** | Fabliq-8B-Agent-Reasoning (Phase-2) | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Reasoning) |
| **단일 학습으로 최고 (8B)** | Fabliq-8B-Agent-Mega | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Mega) |
| **로컬 16GB VRAM (Q4_K_M)** | GGUF 4.9G | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-GGUF) |
| **로컬 24GB VRAM (Q8_0)** | GGUF 8.4G | [link](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-GGUF) |
| **로컬 4-8GB VRAM (1.2B)** | Fabliq-1.2B-Agent Q4_K_M (698M) | [link](https://huggingface.co/LLM-OS-Models/Fabliq-1.2B-Agent) |

---

## 📚 전체 학습 현황

### 8B (10 variants)

| 모델 | 단계 | 데이터 | Epochs | LR | Final Loss |
| --- | --- | --- | --- | --- | --- |
| **Fabliq-8B-Agent** | Phase-1 | Fable-5 4,047 | 3 | 5e-7 | 1.277 |
| Fabliq-8B-Agent-Reasoning | Phase-2 | + 281 reasoning | 4 | 3e-7 | ~1.6 |
| Fabliq-8B-Agent-FromBase | Phase-1B (ablation) | Fable-5 4,047 | 3 | 1e-6 | - |
| Fabliq-8B-Agent-FromBase-Reasoning | Phase-2B | + 281 | 4 | 3e-7 | - |
| **Fabliq-8B-Agent-Mega** | Mega | 4,328 (all) | 3 | 1e-6 | 1.236 |
| Fabliq-8B-Agent-Mega-Reasoning | Phase-2M | Mega + 281 | 2 | 2e-7 | - |
| Fabliq-8B-Agent-Mega-1ep | epoch scan | 4,328 | 1 | 1e-6 | 1.35 |
| Fabliq-8B-Agent-Mega-5ep | epoch scan | 4,328 | 5 | 1e-6 | 1.379 |
| Fabliq-8B-Agent-Mega-10ep | epoch scan | 4,328 | 10 | 5e-7 | - |
| Fabliq-8B-Agent-Mega-lr5e7 | LR scan | 4,328 | 3 | 5e-7 | 1.415 |
| Fabliq-8B-Agent-Mega-lr2e6 | LR scan (loss 최저) | 4,328 | 3 | 2e-6 | **1.169** |

### 1.2B (6 variants)

| 모델 | Base | 데이터 | Final Loss |
| --- | --- | --- | --- |
| Fabliq-1.2B-Agent | LFM2.5-1.2B-Instruct | Fable-5 | 1.55 |
| Fabliq-1.2B-Agent-Mega | LFM2.5-1.2B-Instruct | Mega 4,328 | 1.612 |
| Fabliq-1.2B-Thinking-Agent | LFM2.5-1.2B-Thinking | Fable-5 | ~1.55 |
| Fabliq-1.2B-Thinking-Agent-Mega | LFM2.5-1.2B-Thinking | Mega | 1.613 |
| Fabliq-1.2B-Agent-Base | LFM2.5-1.2B-Base | Mega | 1.554 |
| Fabliq-1.2B-Agent-JP | LFM2.5-1.2B-JP | Mega | 1.549 |

### 2.6B (1 variant)

| 모델 | Base | Final Loss |
| --- | --- | --- |
| Fabliq-2.6B-Agent | LFM2-2.6B | 1.465 |

### 24B (실패)
- LFM2-24B-A2B: H200 8대에서도 CUDA OOM (137GB/GPU 사용). paged_adamw_8bit + max_seq 2048로 시도했으나 설정 정리만 하고 마무리.

---

## 📦 데이터 요약

| 데이터셋 | 원본 | 전처리 후 | 비고 |
| --- | --- | --- | --- |
| [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | 4,665 | 4,047 | 핵심 터미널 에이전트 데이터 |
| WithinUsAI/claude_mythos_distilled_25k | 25,000 | 135 (dedup 99% 중복) | reasoning 보조 |
| HelioAI/Fable-5-Distill-Reasoning-462x | 462 | 146 (Russian 70% 제거) | deep reasoning |
| Glint-Research/Complete-FABLE.5-traces-2M | 2,006,487 | 0 (전부 Fable-5 중복) | 미사용 |
| armand0e/claude-fable-5-claude-code | 18,370 | raw events (미처리) | 미사용 |
| lordx64/agentic-distill-fable-5-sft | 4,659 | Glint 중복 | 미사용 |

**최종 unique 학습 데이터: 4,328 rows**

---

## 🔬 GGUF 라인업 (모든 variants × 4 quants)

총 **72 GGUF 파일** HuggingFace 게시:

| Quant | 8B (10 variants) | 1.2B (6 variants) | 2.6B (1 variant) |
| --- | --- | --- | --- |
| Q4_K_M | 4.9G | 698M | 1.6G |
| Q5_K_M | 5.7G | 805M | 1.8G |
| Q6_K | 6.5G | 919M | 2.1G |
| Q8_0 | 8.4G | 1.1G | 2.7G |

---

## 🌐 HuggingFace 레포 인벤토리 (총 36개)

### 8B 모델 (10 + 10 GGUF = 20 레포)
- LLM-OS-Models/Fabliq-8B-Agent
- LLM-OS-Models/Fabliq-8B-Agent-Reasoning
- LLM-OS-Models/Fabliq-8B-Agent-FromBase
- LLM-OS-Models/Fabliq-8B-Agent-FromBase-Reasoning
- LLM-OS-Models/Fabliq-8B-Agent-Mega
- LLM-OS-Models/Fabliq-8B-Agent-Mega-Reasoning
- LLM-OS-Models/Fabliq-8B-Agent-Mega-1ep / -5ep / -10ep
- LLM-OS-Models/Fabliq-8B-Agent-Mega-lr5e7 / -lr2e6

### 1.2B 모델 (6 + 6 GGUF = 12 레포)
- LLM-OS-Models/Fabliq-1.2B-Agent (Instruct + Fable-5)
- LLM-OS-Models/Fabliq-1.2B-Agent-Mega (Instruct + Mega)
- LLM-OS-Models/Fabliq-1.2B-Thinking-Agent (Thinking + Fable-5)
- LLM-OS-Models/Fabliq-1.2B-Thinking-Agent-Mega (Thinking + Mega)
- LLM-OS-Models/Fabliq-1.2B-Agent-Base (Base + Mega)
- LLM-OS-Models/Fabliq-1.2B-Agent-JP (JP + Mega)

### 2.6B 모델 (1 + 1 GGUF = 2 레포)
- LLM-OS-Models/Fabliq-2.6B-Agent

---

## ⚠️ 알려진 이슈

1. **vLLM ABI 손상**: 시스템의 모든 vllm env에서 undefined symbol 에러. 벤치마크 eval (MMLU, HumanEval)이 transformers backend로만 가능 → 느림. 터미널 task eval만 가능.
2. **2M 데이터 중복**: Complete-FABLE.5-traces-2M은 Fable-5를 재포장한 것. 새 데이터 없음.
3. **WithinUs 템플릿 반복 심함**: 25k → 135 unique (99%가 템플릿 반복).
4. **Helio 러시아어 비율 높음**: 462 → 146 (70% 러시아어 우세 행 제거).
5. **24B OOM**: H200 141GB × 8대로도 부족. paged_adamw_8bit + max_seq 2048 시도했으나 설정 정리로 마무리.
6. **터미널 task 테스트 작음**: 5개 프롬프트로 평가. 통계적 유의성 낮음. 더 큰 eval 세트 필요.

---

## 🔗 관련 문서

- [README.md (영문)](./README.md)
- [README.ko.md (한국어)](./README.ko.md)
- [FINAL_REPORT_20260624.ko.md](./FINAL_REPORT_20260624.ko.md) — 전체 결과 상세
- [DATA_SOURCES_20260623.ko.md](./DATA_SOURCES_20260623.ko.md) — 데이터셋 분석
- [PROGRESS_20260623.ko.md](./PROGRESS_20260623.ko.md) — 진행 타임라인
- [NEXT_STEPS_20260623.ko.md](./NEXT_STEPS_20260623.ko.md) — 후속 작업
- [GitHub: gyunggyung/fable5-to-lfm](https://github.com/gyunggyung/fable5-to-lfm)

---

## 📜 라이선스

Apache 2.0 (LiquidAI LFM base 상속). 학습 데이터인 Fable-5 traces는 Claude 모델에서 증류, Anthropic 사용 정책 따름.
