# 데이터 소스 상세 (2026-06-23)

`fable_distillation/datasets/` 에 있는 6개 데이터셋의 원본 출처, 구조, 전처리 방식, 사용 현황을 정리.

---

## 1. Fable-5-traces (Glint-Research) — Phase-1 메인

- **원본:** [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) (4,665 rows)
- **로컬:** `datasets/Fable-5-traces/fable5_cot_merged.jsonl`
- **구조:** `{uid, session, context, cot, output, output_type}` 구조화 JSONL
  - `context`: 멀티턴 대화 텍스트 (`USER:`, `ASSISTANT (message):`, slash command meta 포함)
  - `cot`: assistant reasoning chain-of-thought
  - `output`: `{tool, input}` (tool_use) 또는 텍스트 응답
  - `output_type`: `tool_use` / `text`
- **전처리:** [`build_fable5_to_lfm_sft.py`](../build_fable5_to_lfm_sft.py)
  1. `context` 파싱 → `USER`/`ASSISTANT (message)` 메시지 리스트
  2. slash command meta (local-command-caveat, command-*, etc.) 제거
  3. `{tool, input}` → LFM native 형식 `<|tool_call_start|>[ToolName(arg='value')]<|tool_call_end|>`
  4. `cot` → `<think>...</think>` 로 감싸기
  5. messages 3개 미만이면 drop (618 rows 스킵)
- **출력:** `datasets/fable5_lfm_sft_20260623.jsonl` (4,047 rows)
- **메타:** `datasets/fable5_lfm_sft_20260623.meta.json`
- **사용:** Phase-1 full SFT 3 epoch

---

## 2. claude_mythos_distilled_25k (WithinUs) — Phase-2 general reasoning

- **원본:** [claude_mythos_distilled_25k](https://huggingface.co/datasets/...) (25,000 rows, 6 카테고리)
- **로컬:** `datasets/claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl`
- **구조:** `{id, category, messages}` — 카테고리별 균등 분할 대상
  - 카테고리: advanced_coding, agentic_planning, general_qa, mathematical_reasoning, scientific_analysis, cybersecurity
  - `messages`: `[{role, content}]` Qwen chat format
  - assistant 응답이 "Drawing from the autonomous, frontier-level reasoning..." 템플릿 첫 문장으로 시작 (패턴 반복)
- **전처리:** [`build_withinus_lfm_sft.py`](../build_withinus_lfm_sft.py)
  1. user content SHA-256 hash 기반 dedup → 25k → 449 unique (24,551 중복 제거)
  2. 카테고리별 max 350 rows 샘플링 (random seed 42)
  3. assistant "Drawing from..." 첫 문장 정규식 제거
  4. system prompt LFM 터미널 에이전트용으로 교체
- **출력:** `datasets/withinus_lfm_sft_20260623.jsonl` (135 rows)
- **메타:** `datasets/withinus_lfm_sft_20260623.meta.json`
- **사용:** Phase-2 reasoning expansion
- **이슈:** dedup 후 135 rows밖에 안 남음 — 원본 데이터가 템플릿 반복이 매우 심함 (TRAining_USAGE.md 경고와 일치)

---

## 3. Fable-5-Distill-Reasoning-462x (Helio) — Phase-2 deep reasoning

- **원본:** `Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl` (462 rows)
- **로컬:** `datasets/Fable-5-Distill-Reasoning-462x/`
- **구조:** `{query, thinking, answer}` 단순 JSONL
- **전처리:** [`build_helio_lfm_sft.py`](../build_helio_lfm_sft.py)
  1. Cyrillic 비율 30% 초과 행 스킵 (Russian 우세 행 제거 — query + thinking 평균)
  2. thinking을 `<think>...</think>` 로 감싸서 assistant reasoning 학습
  3. line 192 손상 스킵
  4. max 8192 tokens 초과 시 truncate
- **출력:** `datasets/helio_lfm_sft_20260623.jsonl` (146 rows)
- **메타:** `datasets/helio_lfm_sft_20260623.meta.json`
- **사용:** Phase-2 reasoning expansion

---

## 4. Complete-FABLE.5-traces-2M — Phase-3 예정

- **원본:** 2,006,487 rows raw events
- **로컬:** `datasets/Complete-FABLE.5-traces-2M/data/train.parquet` (Parquet)
- **구조:** Parquet. 각 행은 `row_json` 필드에 raw Claude Code session event (JSON string) 보관
  - 이벤트 타입: `assistant` (14k/60k), `user` (9k/60k), `attachment` (1.7k), `last-prompt`, `file-history-snapshot`, `system`, `queue-operation`, `permission-mode`, `mode`, `ai-title`, `bridge-session`, `custom-title`, `unknown`
  - `unknown` 타입이 Glint가 미리 뽑은 구조화 row (`completion/context/cot/model/origin/output/output_type/session/source_file/uid`) — 실제 학습 가능한 트레이스는 이쪽
- **샘플링 조사 (60k row 기준):** unique sessionId 478개 → 전체 2M에선 session 수만 개 예상
- **전처리 (예정):** [`build_fable5_2m_to_lfm_sft.py`](../build_fable5_2m_to_lfm_sft.py) (작성 필요)
  - sessionId별 그룹화
  - `unknown` 타입 row만 추출 (Glint 구조화 데이터)
  - Glint 형식 → LFM messages 변환 (build_fable5_to_lfm_sft.py 재사용)
  - 길이/언어/품질 필터
- **사용 (예정):** Phase-3 agentic scale-out

---

## 5. agentic-distill-fable-5-sft (lordx64 패키징)

- **원본:** lordx64 가 Glint 원본을 Qwen template으로 리패키징한 버전 (4,659 rows)
- **로컬:** `datasets/agentic-distill-fable-5-sft/`
- **상태:** 미사용 — Glint 원본(Fable-5-traces)이 더 깨끗한 구조라 그쪽 선택

---

## 6. claude-fable-5-claude-code

- **원본:** Claude Code 원시 dump
- **로컬:** `datasets/claude-fable-5-claude-code/`
- **상태:** 미사용 — Complete-FABLE.5-traces-2M과 중복 + 전처리 비용 더 큼

---

## 혼합 데이터셋

- `datasets/fable5_mixed_lfm_sft_20260623.jsonl` (4,328 rows) = Fable-5 4,047 + WithinUs 135 + Helio 146
- Phase-1 이후 단일 모델로 모든 도메인 커버 시 사용 고려했으나, Phase-2 reasoning 단독 학습 쪽이 더 깔끔해서 현재는 Phase-2 reasoning-only (281 rows) 채택

---

## 시스템 프롬프트 (모든 데이터셋 공통)

Phase-1 agentic:
```
You are an agentic coding assistant. Read the conversation history and tool results,
think step by step inside <think>...</think>, then either call a tool using
<|tool_call_start|>[ToolName(arg=value)]<|tool_call_end|> or respond with text.
Use available tools (Bash, Edit, Read, Write, Glob, Grep, WebSearch, WebFetch, etc.)
to accomplish the user's task. Be concise but thorough.
```

Phase-2 reasoning (WithinUs):
```
You are a knowledgeable assistant. Provide rigorous, well-structured answers
across coding, cybersecurity, mathematics, scientific analysis, agentic planning,
and general expert topics. Be precise and thorough.
```

Phase-2 reasoning (Helio):
```
You are a deep-reasoning assistant. Think step by step inside <think>...</think>,
then provide a clear, structured answer.
```
