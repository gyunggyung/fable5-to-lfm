# Fable-5 / Mythos Distillation Datasets

HuggingFace에서 다운로드한 **Fable-5 / Mythos 계열 디스틸레이션 데이터셋 6종**의 구조 분석 문서.

- **작업 루트**: `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/`
- **분석 일자**: 2026-06-23
- **전체 다운로드 크기**: 약 **2.5 GB** (데이터 2.46G + 메타)

---

## 1. 데이터셋 비교표

| # | 데이터셋 | 행 수 | 크기 | 포맷 | 라이선스 | 모델 | 용도 |
|---|---|---|---|---|---|---|---|
| 1 | `Glint-Research/Complete-FABLE.5-traces-2M` | **2,006,487** | 2.0G | parquet + jsonl.gz | MIT | Fable-5 외 다수 | 전체 코퍼스 dedup (메타 인덱스) |
| 2 | `Glint-Research/Fable-5-traces` | **4,665** | 190M | jsonl + pi-traces | AGPL-3.0 | claude-fable-5 | SFT/디스틸 (CoT 포함, 핵심) |
| 3 | `armand0e/claude-fable-5-claude-code` | **18,370** (63 files) | 73M | jsonl (세션별) | 미표기 | claude-fable-5 | 원본 Claude Code 세션 로그 (thinking redacted) |
| 4 | `WithinUsAI/claude_mythos_distilled_25k` | **25,000** | 53M | jsonl (messages) | 미표기 | Mythos (mirror) | 카테고리별 고품질 Q&A |
| 5 | `lordx64/agentic-distill-fable-5-sft` | **4,659** | 15M | parquet (text 1열) | AGPL-3.0 | claude-fable-5 | Qwen 템플릿 SFT 즉시 사용 |
| 6 | `HelioAI/Fable-5-Distill-Reasoning-462x` | **462** (1 손상) | 140M | jsonl | 미표기 | Mythos V2 (주장) | 초장문 reasoning (≈26.35M 토큰) |

### 핵심 관계도
```
Fable-5 원본 세션 (Claude Code 로그, Anthropic API)
   │
   ├─▶ armand0e/claude-fable-5-claude-code  (원본 로그 그대로, thinking redacted)
   │
   ├─▶ Glint-Research/Fable-5-traces        (CoT 복원 + pi-traces 변환)
   │      │
   │      └─▶ lordx64/agentic-distill-fable-5-sft  (Qwen 템플릿 SFT 포맷)
   │
   ├─▶ Glint-Research/Complete-FABLE.5-traces-2M  (전체 코퍼스 dedup/인덱스)
   │
   ├─▶ WithinUsAI/claude_mythos_distilled_25k     (Mythos mirror, 카테고리별 Q&A)
   │
   └─▶ HelioAI/Fable-5-Distill-Reasoning-462x     (Mythos V2 주장, 초장문 reasoning)
```

---

## 2. 폴더 구조 (전체 경로)

```
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/
├── README.md                          ← 이 파일
├── analysis/
│   ├── analyze_datasets.py            ← 스키마/샘플 분석 스크립트
│   ├── analyze_stats.py               ← 분포/길이 통계 스크립트
│   ├── analysis_output.txt            ← 구조 분석 결과
│   ├── stats_output.txt               ← 통계 분석 결과
│   └── file_tree.txt                  ← 전체 파일 경로 목록
├── logs/
│   ├── download_2m.log
│   ├── download_fable5.log
│   ├── download_armand0e.log
│   ├── download_withinus.log
│   ├── download_lordx64.log
│   └── download_helio.log
└── datasets/
    ├── Complete-FABLE.5-traces-2M/     (2.0G)
    │   ├── README.md
    │   ├── data/
    │   │   └── train.parquet           ← 2,006,487 rows (937M)
    │   └── raw/
    │       └── fable5_mythos_dedup.jsonl.gz  ← 압축 1.1G
    ├── Fable-5-traces/                 (190M)
    │   ├── README.md
    │   ├── .gitattributes
    │   ├── assets/
    │   │   └── glintresearchfableheader.png
    │   ├── claude/
    │   │   └── history.jsonl           (547K, 원본 아카이브)
    │   ├── fable5_cot_merged.jsonl     ← 4,665 rows (67M, 핵심 SFT 원천)
    │   └── pi-traces/                  ← 4,007개 Pi 호환 트레이스
    │       ├── 00000-f956721a-...-e3fb728481.jsonl
    │       ├── 00001-f956721a-...-3e2dd35e82.jsonl
    │       └── ... (총 4,007개 파일)
    ├── claude-fable-5-claude-code/     (73M)
    │   ├── README.md                   (32K, tools_schema 포함)
    │   ├── .gitattributes
    │   ├── f956721a-0af7-4bdc-8678-3a493d8fcd39.jsonl  (14M, 최대)
    │   ├── 6ae0aeed-340e-4b4d-9d74-f905a74ca2fc.jsonl  (6.3M)
    │   ├── e3d0c93f-5cf8-4067-9a49-657ba5c67c80.jsonl  (4.3M)
    │   ├── 602033e7-dc55-4b7f-909f-1c67135d8f4b.jsonl  (3.3M)
    │   ├── c6d4788b-6bb6-4774-8294-5dc897346ca9.jsonl  (3.1M)
    │   └── ... (총 63개 세션 jsonl)
    ├── claude_mythos_distilled_25k/    (53M)
    │   ├── README.md
    │   ├── .gitattributes
    │   └── claude_mythos_distilled_25k.jsonl  ← 25,000 rows (53M)
    ├── agentic-distill-fable-5-sft/    (15M)
    │   ├── README.md
    │   ├── .gitattributes
    │   └── data/
    │       └── train-00000-of-00001.parquet  ← 4,659 rows (14M)
    └── Fable-5-Distill-Reasoning-462x/ (140M)
        ├── README.md                   (25K)
        ├── .gitattributes
        └── Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl  ← 462 rows (140M)
```

---

## 3. 데이터셋별 상세 분석

### 3.1 `Glint-Research/Complete-FABLE.5-traces-2M` — 전체 코퍼스 dedup 인덱스

**용도:** Fable-5 / Mythos 관련 모든 공개 코퍼스를 하나로 모아 SHA-256 해시로 dedup한 **메타 인덱스**. 원본 row는 `row_json`에 통째로 보존.

**경로:**
- `datasets/Complete-FABLE.5-traces-2M/data/train.parquet` (937M)
- `datasets/Complete-FABLE.5-traces-2M/raw/fable5_mythos_dedup.jsonl.gz` (1.1G, gzipped)
- `datasets/Complete-FABLE.5-traces-2M/README.md`

**스키마 (parquet, 7열):**
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `row_hash` | string | 정규화된 원본 row JSON의 SHA-256 |
| `first_source_dataset` | string | 처음 관측된 HF 데이터셋 |
| `first_source_config` | string | config 또는 loader surface |
| `first_source_split` | string | split / 파일 / local 추출 라벨 |
| `first_source_row_index` | int64 | 첫 관측 소스 내 row index |
| `seen_count` | int64 | dedup 패스 중 관측된 횟수 (2~486) |
| `row_json` | string | 원본 row 통째로 (파싱해서 source 필드 추출) |

**통계:**
- 총 **2,006,487 rows** (session-limit answer 604행 제거된 clean snapshot)
- 샘플링(60k) 기준 top source:
  - `ansulev/claude_mythos_distilled_25k` 25,000
  - `1EYE4ALL/Fable-5-traces` 22,904
  - `armand0e/claude-fable-5-claude-code` 12,096
- `row_json` 안의 원본 필드: `attachment`, `content`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `lastPrompt`, `leafUuid`, `message{content,role}`, `operation`, `parentUuid`, `permissionMode`, `promptId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`

**샘플 row:**
```json
{
  "row_hash": "402f68b44c68d25be5d077288712d83c788e2c2588bdc79ac29343ac01a95779",
  "first_source_dataset": "1EYE4ALL/Fable-5-traces",
  "first_source_config": "default",
  "first_source_split": "test",
  "first_source_row_index": 0,
  "seen_count": 3,
  "row_json": "{\"attachment\":null,\"content\":\"Create a file hello.py...\",\"operation\":\"enqueue\",\"sessionId\":\"65c61158-...\",\"timestamp\":\"2026-06-10T20:42:43.38...\",\"type\":\"queue-operation\",...}"
}
```

**라이선스:** MIT

---

### 3.2 `Glint-Research/Fable-5-traces` — Fable-5 CoT 디스틸레이션 핵심

**용도:** Fable-5 coding-agent 트레이스를 CoT 포함 flat JSONL + Pi 호환 agent trace로 변환. **SFT/디스틸의 실질적 출처**. cleartext CoT가 포함된 유일한 소스(lordx64 README에 따르면 armand0e/victor 계열은 thinking이 redacted됨).

**경로:**
- `datasets/Fable-5-traces/fable5_cot_merged.jsonl` ← **핵심, 4,665 rows (67M)**
- `datasets/Fable-5-traces/pi-traces/*.jsonl` ← 4,007개 Pi 호환 트레이스 (HF Data Studio 렌더링용)
- `datasets/Fable-5-traces/claude/history.jsonl` ← 원본 아카이브 (547K)
- `datasets/Fable-5-traces/README.md`

**스키마 (`fable5_cot_merged.jsonl`, 10열):**
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `uid` | string | `session_id#index` 안정 식별자 |
| `source_file` | string | 원본 트레이스 경로 |
| `session` | string | 세션 ID |
| `model` | string | 모두 `claude-fable-5` |
| `context` | string | 프롬프트 + 이전 트랜스크립트 (med 7,022 chars) |
| `cot` | string | reasoning trace (med 2,365 chars) |
| `output_type` | string | `tool_use` / `text` |
| `output` | object | tool rows면 `{name, arguments}`, text rows면 텍스트 |
| `completion` | string | reasoning + output 직렬화 (med 2,726 chars) |
| `origin` | string | `local` / `hf` |

**통계:**
- 4,665 rows / 60 source sessions / 3,799 tool actions / 866 text actions
- `output_type`: **tool_use 81.44% (3,799) / text 18.56% (866)**
- tool 분포(README 기준): Bash 1,544 / Edit 960 / Read 443 / Write 311 / PowerShell 136 / WebSearch 72 / preview_eval 63 / WebFetch 44 / TaskUpdate 37 / ToolSearch 35 / TaskCreate 26 / preview_screenshot 24 / ScheduleWakeup 23
- source mix: `-home-lane-MythosMini` 2,024 / imported HF 953 / `-home-lane-GR` 447 / `-home-lane` 425 / `-home-lane-AIArchives` 316 / 기타

**샘플 row (요약):**
```json
{
  "uid": "f956721a-0af7-4bdc-8678-3a493d8fcd39#5",
  "source_file": "/home/lane/.claude/projects/-home-lane-AIArchives/f956721a-...jsonl",
  "session": "f956721a-0af7-4bdc-8678-3a493d8fcd39",
  "model": "claude-fable-5",
  "context": "USER: Make a new one, it should be a fast paced multiplayer FPS...",
  "cot": "<think>...reasoning trace...</think>",
  "output_type": "tool_use",
  "output": {"name": "Write", "arguments": {...}},
  "completion": "<think>...</think>\n<tool_use>...</tool_use>",
  "origin": "local"
}
```

**Pi-trace 매핑 (README):**
각 merged row → `session`(UUID) / `model_change`(`claude-fable-5`) / `thinking_level_change`(high) / user message(`context`) / assistant message(`thinking` item + `text` 또는 `toolCall` item).

**라이선스:** AGPL-3.0 (상업적/폐쇄형 학습 파이프라인은 호환성 검증 필요)

---

### 3.3 `armand0e/claude-fable-5-claude-code` — 원본 Claude Code 세션 로그

**용도:** Fable-5가 구동된 Claude Code 세션의 **원본 JSONL 로그**. Anthropic API thinking 블록은 redacted (signature만 남음). lordx64 README에 따르면 **SFT용 CoT로는 부적합** (thinking 비어있음), 트레이스 무결성 검증/아카이브용.

**경로:**
- `datasets/claude-fable-5-claude-code/README.md` (32K, tools_schema JSON 포함)
- `datasets/claude-fable-5-claude-code/*.jsonl` ← **63개 세션 파일**
  - 최대: `f956721a-0af7-4bdc-8678-3a493d8fcd39.jsonl` (14M)
  - 차상위: `6ae0aeed-...` (6.3M), `e3d0c93f-...` (4.3M), `602033e7-...` (3.3M), `c6d4788b-...` (3.1M)

**스키마 (jsonl line, `type`별로 가변):**
라인마다 `type` 필드가 있고 type별 서브 필드가 다름. 공통: `type`, `sessionId`.
- `assistant`: assistant 메시지 (thinking redacted, content/content blocks)
- `user`: user 입력
- `ai-title`, `custom-title`: 세션 제목
- `last-prompt`, `mode`, `permission-mode`, `attachment`, `file-history-snapshot`, `queue-operation`, `system`, `bridge-session`

**통계 (63개 파일, 18,370 rows):**
| type | rows |
|---|---|
| assistant | 7,490 |
| user | 4,719 |
| ai-title | 1,010 |
| last-prompt | 965 |
| mode | 960 |
| permission-mode | 884 |
| attachment | 806 |
| file-history-snapshot | 596 |
| queue-operation | 376 |
| system | 349 |
| bridge-session | 128 |
| custom-title | 87 |

**샘플 (type별):**
```json
{"type":"last-prompt","leafUuid":"3df39b4b-1a0d-4094-b9c2-054dc1e167ec","sessionId":"f956721a-..."}
{"type":"mode","mode":"normal","sessionId":"f956721a-..."}
```

**README의 tools_schema:** 약 60개 tool 정의 (Agent, AskUserQuestion, Bash, BashOutput, CronCreate/Delete/List, Edit, EnterPlanMode, EnterWorktree, ExitPlanMode, ExitWorktree, Glob, Grep, KillBash, LS, MultiEdit, NotebookEdit/Read, PushNotification, Read, RemoteTrigger, ScheduleWakeup, Skill, StructuredOutput, Task/TaskCreate/Get/List/Output/Stop/Update, TodoWrite, ToolSearch, WebFetch, WebSearch, Workflow, Write, mcp__claude_ai_* (Gmail/Calendar/Drive), mcp__context7__*). 이 tool 세트가 **Claude Desktop 2026 preview** 기준.

**라이선스:** 카드에 명시 없음 (Anthropic claude-fable-5 출력물, 사용 정책 별도 확인 필요)

---

### 3.4 `WithinUsAI/claude_mythos_distilled_25k` — 카테고리별 Q&A

**용도:** Claude Mythos(미러/distilled)에서 생성한 **단답형 고품질 Q&A 25k**. coding-agent 트레이스가 아니라 일반 지식/추론 QA. assistant 응답이 비교적 짧고 균일(med 1,707 chars)해서 instruction-tuning에 적합.

**경로:**
- `datasets/claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl` ← 25,000 rows (53M)
- `datasets/claude_mythos_distilled_25k/README.md`

**스키마 (5열):**
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `messages` | array | `[{role:"user"/"assistant", content}, ...]` |
| `category` | string | 6개 카테고리 중 하나 |
| `id` | string | `mythos-distilled-XXXXX` |
| `source` | string | `synthetic_claude_mythos_distilled_mirror` |
| `timestamp` | string | ISO (전부 `2026-05-14T16:32:58.16...`) |

**통계 (category 분포):**
| category | rows |
|---|---|
| cybersecurity | 7,000 |
| advanced_coding | 5,500 |
| agentic_planning | 3,500 |
| general_expert_qa | 3,500 |
| mathematical_reasoning | 3,000 |
| scientific_analysis | 2,500 |

- assistant content 길이: min 1,278 / median 1,707 / max 2,119 / mean 1,706 chars
- 모든 assistant 응답이 "Drawing from the autonomous, frontier-level reasoning characteristic of Claude Mythos..."로 시작 (템플릿화)

**샘플:**
```json
{
  "messages": [
    {"role":"user","content":"Solve or provide a rigorous proof sketch for: compute the closed-form solution for the 3-body problem approximation..."},
    {"role":"assistant","content":"Drawing from the autonomous, frontier-level reasoning characteristic of Claude Mythos (distilled for accessibility and precision)..."}
  ],
  "category":"mathematical_reasoning",
  "id":"mythos-distilled-00000",
  "source":"synthetic_claude_mythos_distilled_mirror",
  "timestamp":"2026-05-14T16:32:58.164857"
}
```

**주의:** prompt 패턴이 한정적(약 20~30개 템플릿 반복)이라 다양성이 낮음. 카테고리 라벨 기반 밸런스 조정에 유리.

**라이선스:** 카드에 명시 없음

---

### 3.5 `lordx64/agentic-distill-fable-5-sft` — Qwen 템플릿 SFT 즉시 사용

**용도:** `Glint-Research/Fable-5-traces` 4,659행을 **Qwen chat template 단일 `text` 컬럼**으로 재포장. `SFTTrainer(dataset_text_field="text") + train_on_responses_only`에 바로 투입 가능. `lordx64/Qwable-v1` (Qwen3.6-35B-A3B, Opus 4.7 distill warm-start) 학습에 사용됨.

**경로:**
- `datasets/agentic-distill-fable-5-sft/data/train-00000-of-00001.parquet` ← 4,659 rows (14M)
- `datasets/agentic-distill-fable-5-sft/README.md`

**스키마 (3열):**
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `text` | string | Qwen `<|im_start|>...<|im_end|>` 템플릿 적용 전체 시퀀스 |
| `source` | string | `Glint-Research/Fable-5-traces` |
| `session` | string | `glint` |

**통계:**
- 4,659 rows
- `<tool_use>` 포함: 3,858행 (**82%**)
- `<think>` 포함: 4,659행 (**100%**)
- text 길이: min 673 / median **9,783** / max 80,759 / mean 10,468 chars (≈2.6k Qwen tokens/row)
- 총 48,772,554 chars (≈12.19M Qwen tokens)

**포맷 (Qwen 템플릿 + 커스텀 XML 래퍼):**
```
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{user_or_tool_result}<|im_end|>
<|im_start|>assistant
<think>
{fable5_thinking}
</think>

{fable5_response_with_tool_calls}<|im_end|>
```
- `tool_use`: `<tool_use name="X" id="Y">…</tool_use>` (Qwen 네이티브 `<tool_call>` 아님, regex 파싱)
- `tool_result`: `<tool_result id="X">…</tool_result>`

**전처리 내역 (README):**
1. Claude Code `<synthetic>` rate-limit injection 제거
2. thinking 없는 assistant 턴 / CoT < 50 chars / output 검증 실패 row 제거
3. slash-command 메타 블록(`local-command-caveat`, `command-name` 등) 및 ANSI escape 제거
4. **Groq API key 2개(204회 출현) redact** (원 세션 JSONL의 `.env` Read에서 캡처됨)
5. user-turn content 기준 SHA-256 dedup (6행 제거)

**주의:** Anthropic `claude-fable-5`는 2026-06-10~22 게이트된 preview 모델이며 2026-06-22 미국 수출통제 directive로 전역 정지. downstream 사용자는 Anthropic 사용 정책 별도 확인 필요.

**라이선스:** AGPL-3.0 (upstream 상속)

---

### 3.6 `HelioAI/Fable-5-Distill-Reasoning-462x` — 초장문 reasoning

**용도:** 이름에는 "Fable-5"가 들어가지만 README는 **"full unrestricted Mythos V2"** 디스틀이라 명시. Fable-5 안전 정렬으로 suppress되는 reasoning을 raw base weight에서 추출했다고 주장. **462행이지만 ≈26.35M 토큰**(평균 226k chars/row)의 극단적 장문 reasoning. process supervision / long-context eval 용도.

**경로:**
- `datasets/Fable-5-Distill-Reasoning-462x/Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl` ← 462 rows (140M)
- `datasets/Fable-5-Distill-Reasoning-462x/README.md` (25K)

**스키마 (가변, 2~3열):**
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `query` | string | 사용자 질문 (러시아어/영어 혼합) |
| `thinking` | string | reasoning trace (핵심, 초장문) |
| `answer` | string | (선택) 일부 row |
| `response` | string | (선택) 일부 row |

**통계 (462 rows, 1행 손상):**
- ok 461 / bad 1 (line 192: 두 JSON 객체가 붙어있어 `Extra data` 에러)
- thinking 길이: min **12,827** / median **250,385** / max **552,196** / mean 226,718 chars
- **>300k chars: 94행** (README 값과 일치)
- 총 104,731,151 chars (≈26.35M 토큰 추정)
- query script: **Cyrillic 320 / Latin·other 141** (러시아어가 다수)
- 토픽 분포(README): Cybersecurity 153 / Biomedicine 144 / Software & Distributed 98 / AI·LLM 45 / Formal Math 10 / 기타 12

**샘플:**
```json
{
  "query": "Разработай архитектуру мультиагентной системы для автоматизированного анализа финансовых рынков...",
  "thinking": "Хорошо, мне нужно разработать архитектуру мультиагентной системы... **ШАГ 1: Анализ требований...**"
}
```

**주의사항:**
- 파일명(`Claude-Opus-4.7-4.8-DeepReason`)과 README(Mythos V2)가 불일치 → 실제 출처 모델 확인 필요
- "unrestricted / uncensored" 마케팅. 사이버보안/생물의학 주제가 다수 → 민감 주제, 사용 전 로컬 규정 확인
- 러시아어 비중 70% → 한국어/영어 SFT에는 직접 활용도 낮음, 번역/필터링 전처리 필요

**라이선스:** 카드에 명시 없음

---

## 4. 다운로드 명령어 (재현용)

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets

hf download Glint-Research/Complete-FABLE.5-traces-2M --repo-type dataset --local-dir Complete-FABLE.5-traces-2M
hf download Glint-Research/Fable-5-traces --repo-type dataset --local-dir Fable-5-traces
hf download armand0e/claude-fable-5-claude-code --repo-type dataset --local-dir claude-fable-5-claude-code
hf download WithinUsAI/claude_mythos_distilled_25k --repo-type dataset --local-dir claude_mythos_distilled_25k
hf download lordx64/agentic-distill-fable-5-sft --repo-type dataset --local-dir agentic-distill-fable-5-sft
hf download HelioAI/Fable-5-Distill-Reasoning-462x --repo-type dataset --local-dir Fable-5-Distill-Reasoning-462x
```

## 5. 로딩 예제

```python
# 1. Complete-FABLE.5-traces-2M
from datasets import load_dataset
ds = load_dataset("parquet", data_files="datasets/Complete-FABLE.5-traces-2M/data/train.parquet", split="train")

# 2. Fable-5-traces (merged)
import json
with open("datasets/Fable-5-traces/fable5_cot_merged.jsonl") as f:
    rows = [json.loads(line) for line in f]  # 4,665 rows

# 3. armand0e (세션별)
import glob, json
for fp in sorted(glob.glob("datasets/claude-fable-5-claude-code/*.jsonl")):
    with open(fp) as f:
        session = [json.loads(line) for line in f]

# 4. WithinUs 25k
with open("datasets/claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl") as f:
    rows = [json.loads(line) for line in f]  # 25,000 rows

# 5. lordx64 SFT (바로 SFTTrainer)
from datasets import load_dataset
ds = load_dataset("parquet", data_files="datasets/agentic-distill-fable-5-sft/data/train-00000-of-00001.parquet", split="train")
# ds[0]["text"] → Qwen chat template 적용된 단일 문자열

# 6. Helio 462x (line 192 손상 주의)
with open("datasets/Fable-5-Distill-Reasoning-462x/Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl") as f:
    for line in f:
        try: obj = json.loads(line)
        except: continue  # 손상 라인 스킵
```

---

## 6. 활용 시나리오 (추천)

| 목표 | 추천 데이터셋 | 비고 |
|---|---|---|
| **agentic tool-use SFT** (Bash/Edit/Read/Write) | `lordx64/agentic-distill-fable-5-sft` → 바로 SFTTrainer | 4,659행, Qwen 템플릿, 82% tool_use |
| **CoT 포함 커스텀 포맷** | `Glint-Research/Fable-5-traces` (fable5_cot_merged.jsonl) | cleartext CoT 원천 |
| **전체 코퍼스 dedup/중복제거** | `Complete-FABLE.5-traces-2M` | row_hash + seen_count로 중복 통제 |
| **instruction QA 튜닝** (비-agent) | `WithinUsAI/claude_mythos_distilled_25k` | 6카테고리 밸런스 |
| **장문 reasoning / process supervision** | `HelioAI/Fable-5-Distill-Reasoning-462x` | 러시아어 비중 높음, 사이버/생물 주제 |
| **아카이브 / 무결성 검증** | `armand0e/claude-fable-5-claude-code` | thinking redacted, SFT 부적합 |

## 7. 주의사항 요약

1. **라이선스 혼재**: MIT(2M) / AGPL-3.0(Fable-5-traces, lordx64) / 미표기 3종. AGPL은 폐쇄형 상업 파이프라인 호환성 주의.
2. **claude-fable-5 정지**: 2026-06-22 미국 수출통제로 전역 정지된 Anthropic preview 모델 출력물. downstream 사용 정책 별도 확인.
3. **Helio 출처 불일치**: 파일명은 Opus-4.7-4.8, README는 Mythos V2 주장. line 192 손상.
4. **armand0e thinking redacted**: Anthropic API IP 보호로 thinking 블록 비어있음 (signature만 남음). SFT용 CoT로는 부적합.
5. **개인정보/시크릿**: lordx64는 Groq API key 2개(204회) redact 처리했으나, 원본 계열(armand0e/2M)에는 잔류할 수 있음. 사용 전 scrub 권장.
6. **WithinUs 다양성 낮음**: 프롬프트 템플릿 반복 다수 → 과적합 위험, 샘플링/중복제거 권장.
7. **Helio 언어 분포**: 러시아어 70% → 한국어/영어 작업엔 전처리 필요.
