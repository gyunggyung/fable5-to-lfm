# 다음 세션 후보 (2026-06-23)

Phase-1, Phase-2 완료 후 진행할 수 있는 작업들을 우선순위별로 정리. 각 항목은 참고 모델(Qwable, Qwythos, yuxinlu1 v1/v2)에서 차용할 만한 패턴을 명시.

---

## 우선순위 1: Phase-3 — 2M Traces Agentic Scale-Out

**목표:** Fable-5 agentic trajectory 수를 4K → 50K+로 확대. Qwythos가 500M 토큰 학습한 것과 비슷한 스케일 지향.

**작업:**
1. [`build_fable5_2m_to_lfm_sft.py`](../build_fable5_2m_to_lfm_sft.py) 작성
   - `Complete-FABLE.5-traces-2M/data/train.parquet` 읽기
   - `unknown` 타입 row만 필터 (Glint 구조화 데이터: cot/output/output_type 보유)
   - sessionId별 그룹화하여 동일 세션 내 trace 중복 제거
   - 기존 `build_fable5_to_lfm_sft.py` 로직 재사용 (context 파싱, tool_call 변환, cot → `<think>`)
   - 언어/길이 필터 (Cyrillic 30%, 토큰 수 8192)
2. GPU 8대 학습
   - epochs 1 (데이터 큼), LR 5e-7, max_seq 8192
   - 예상 소요 시간: 50K rows × 1 epoch / 64 batch × 8 GPU = 약 2-3시간

**리스크:**
- 2M raw events 전체를 처리하면 너무 많음 → 샘플링 전략 필요 (카테고리/길이/언어 균형)
- tool_dist 편향 가능성 (Bash 위주 등) → 카테고리별 쿼터

---

## 우선순위 2: Eval — Base 대비 성능 측정

**목표:** Phase-1/2/3 모델이 base LFM2.5-8B-A1B 대비 실제로 agentic 능력이 개선됐는지 확인.

**작업:**
1. `tb2_lite/scripts/replay_eval_transformers.py` 활용
   - base LFM2.5-8B-A1B
   - ToolBench-Full-SFT-1Epoch
   - Fabliq-8B-Agent (Phase-1)
   - Fabliq-8B-Agent-Reasoning (Phase-2)
2. 측정 항목:
   - Tool-call accuracy (올바른 툴 선택)
   - Terminal task completion (파일 read/edit/run 성공률)
   - Hallucination rate (존재하지 않는 파일/함수 인용 빈도)

**참고:** yuxinlu1 v2 — tau2-bench `telecom` (diagnose→fix→verify 루프)로 15% → 55% (3.5×). 우리도 terminal task 비슷한 루프 벤치마크 필요.

---

## 우선순위 3: Mythiq 라인 (Mythos 추가 학습)

**목표:** Qwythos-9B-Claude-Mythos-5-1M 대응 — Mythos 데이터를 활용한 general/deep reasoning 강화 라인.

**작업:**
1. `claude_mythos_distilled_25k`에서 WithinUs 전체 449 unique rows 활용 (현재는 135만 샘플링)
2. LFM2.5-8B-A1B base에서 Mythos만 SFT → `LLM-OS-Models/Mythiq-8B`
3. 그 다음 Fable-5 agentic SFT → `LLM-OS-Models/Mythiq-8B-Agent` (또는 Fabliq-8B와 동일 체인)
4. Phase-1/2 구조 그대로 재사용 가능

**차이점 (Qwythos vs 우리):**
- Qwythos는 Mythos 500M+ tokens full 학습
- 우리는 Mythos 25k rows (dedup 후 449 rows) — 데이터가 너무 적어 풀샷 학습 효과 제한적
- 대신 Complete-FABLE.5-traces-2M이 Mythos 기반일 가능성 → Phase-3로 대체 가능

---

## 우선순위 4: 긴 컨텍스트 확장

**목표:** max_seq 8192 → 16384 / 32768 / 65536 확장 (Qwythos 128K, YaRN 1M 패턴).

**이슈:**
- LFM2.5-8B-A1B는 `max_position_embeddings=128000` 이미 지원
- 현재 max_seq 8192는 VRAM 한계가 아니라 데이터 median 길이(2,886) + p99(9,352) 기반 선택
- 긴 트레이스가 많은 2M 학습 시 max_seq 16384로 올리면 자연스럽게 long-context 능력 강화

**VRAM 예상 (H200 141GB × 8):**
- max_seq 8192, batch 2 → 약 79% 사용 (현재)
- max_seq 16384, batch 1 → 약 80% 예상
- max_seq 32768, batch 1, grad_accum 8 → 약 85% 예상

---

## 우선순위 5: Test-Gated Filtering (yuxinlu1 v1 패턴)

**목표:** Fable-5 trace 중 실행 결과가 pass한 것만 학습에 사용 (hallucination 줄이고 정확도 향상).

**참고:** yuxinlu1 v1 — Composer 2.5 CoT 중 "code actually passed its tests"만 학습. Fable-5 hard cases는 synthetic CoT로 재생성.

**작업:**
1. 2M raw events에서 assistant tool_use → tool_result 페어 추출
2. tool_result가 success/error 여부 판별
3. 전체 트레이스가 success 시퀀스인 것만 필터
4. Phase-3 데이터셋 품질 크게 향상

**리스크:**
- 2M events 전부 실행 결과를 판별하는 건 코스트 큼
- 다행히 row_json에 tool_use_id / tool_result 매핑이 있으면 자동 판별 가능 — 스키마 조사 필요

---

## 우선순위 6: Agentic Loop Continuity (yuxinlu1 v2 패턴)

**목표:** read → reason → act → verify 시퀀스가 끊기지 않게 dynamic context-window 전처리.

**참고:** yuxinlu1 v2 — "built a dynamic context-window pass to keep the agent's read-before-act steps intact". Fable-5 trace에서 read/grep 없이 바로 edit 하는 케이스는 drop.

**작업:**
1. Fable-5 trace에서 tool_call 시퀀스 추출
2. `read/grep/glob` → `edit/write/bash` → `read` (verify) 패턴 보존
3. 점프 패턴 (첫 action이 edit) drop
4. tau2-bench 같은 multi-step 루프 성능 향상 기대

---

## 우선순위 7: Assistant-Only Loss 명시적 마스킹

**목표:** chat template에 의존하지 않고 completion-only loss를 더 엄격하게 적용 (Qwythos 패턴).

**참고:** Qwythos — "assistant-only loss (the model is scored only on the assistant/completion tokens; prompts are masked)" + "chunked NLL".

**이슈:**
- 현재 `train_lfm25_rlvr_json_sft.py`는 chat template 적용 후 전체 시퀀스 학습
- assistant turn만 loss 계산하려면 `labels`에서 user/system turn 위치를 -100으로 마스킹해야 함
- 코드 수정 필요

---

## 브랜드 라인업 (HuggingFace)

| 모델 | 역할 | 상태 |
| --- | --- | --- |
| `LLM-OS-Models/Fabliq-8B-Agent` | Fable-5 agentic foundation (Phase-1) | 업로드 중 |
| `LLM-OS-Models/Fabliq-8B-Agent-Reasoning` | + WithinUs+Helio reasoning (Phase-2) | 학습 중 |
| `LLM-OS-Models/Fabliq-8B-Agent-Large` | + 2M traces scale-out (Phase-3) | 계획 |
| `LLM-OS-Models/Mythiq-8B` | Mythos-only 기반 (병행 라인) | 계획 |
| `LLM-OS-Models/Mythiq-8B-Agent` | Mythiq + Fable-5 agentic | 계획 |

참고 모델 라인업과 대응:
- Qwable-9B-Claude-Fable-5 ≈ Fabliq-8B-Agent
- Qwythos-9B-Claude-Mythos-5-1M ≈ Mythiq-8B-Agent
- gemma-4-12B-coder-fable5-composer2.5-v1 ≈ (별도 coder 라인検토)
- gemma-4-12B-agentic-fable5-composer2.5-v2 ≈ Fabliq-8B-Agent-Large
