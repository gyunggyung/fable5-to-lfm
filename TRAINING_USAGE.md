# Fable-5 / Mythos Distillation — 학습 활용 가이드

각 데이터셋을 **실제 학습 파이프라인에 어떻게 쓸 수 있는지** 정리한 문서. 구조/스키마는 `README.md` 참고.

- **작업 루트**: `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/`
- **작성일**: 2026-06-23

---

## 0. 결론 한 줄 요약

> **agentic tool-use SFT가 목표라면 `lordx64/agentic-distill-fable-5-sft` 단독으로 충분.** 추가 신호가 필요하면 `Fable-5-traces`에서 CoT/세션을 직접 추출해 붙이고, 일반 지시어튜닝 신호는 `WithinUs 25k`를 보조로 섞는다. 나머지 3개(Helio / armand0e / 2M)는 학습 직접 투입보다는 **코퍼스 분석·중복 통제·아카이브** 용도.

---

## 1. 학습 목적별 추천 매트릭스

| 학습 목적 | 1순위 | 2순위 | 비고 |
|---|---|---|---|
| **Agentic tool-use SFT** (Bash/Edit/Read/Write/Task) | `lordx64/agentic-distill-fable-5-sft` | `Fable-5-traces` (재포장) | Qwen3 계열에 최적화 |
| **CoT + tool-call 커스텀 포맷** | `Fable-5-traces` (fable5_cot_merged.jsonl) | — | cleartext CoT 원천 |
| **일반 instruction QA 튜닝** (수학/코딩/사이버) | `WithinUs 25k` | — | 카테고리 밸런스 조정 |
| **장문 reasoning SFT** (영어) | `Fable-5-traces` (긴 cot 행만 추출) | `Helio 462x` (영어 30%만 필터) | Helio는 러시아어 70%라 비효율 |
| **DPO / preference pair 생성** | `Fable-5-traces` (text vs tool_use 쌍) | — | output_type별 preference 설계 |
| **Continual pretrain / 대규모 코퍼스** | `Complete-FABLE.5-traces-2M` (row_json) | — | 파싱 비용 큼, 중복 다수 |
| **모델 행동 분석 / 평가용 트레이스** | `armand0e/claude-fable-5-claude-code` | `Fable-5-traces/pi-traces/` | 학습이 아닌 eval/inspect 용 |

---

## 2. 데이터셋별 학습 가치 평가

### 2.1 `lordx64/agentic-distill-fable-5-sft` ★★★★★
**학습에 바로 투입 가능한 유일한 데이터셋.**

- **포맷**: Qwen `<|im_start|>` 템플릿 + 커스텀 XML tool_use 래퍼 → `SFTTrainer(dataset_text_field="text")`
- **학습 검증됨**: `lordx64/Qwable-v1` (Qwen3.6-35B-A3B, Opus 4.7 distill warm-start) 2 epoch / 582 steps / 최종 loss 0.7956 (H200 싱글)
- **신호 품질**:
  - 100% `<think>` 포함 (CoT 강제)
  - 82% `<tool_use>` 종단 (agentic policy 학습에 최적)
  - 시크릿 스크럽(Groq key 204회 redact), dedup, 노이즈 제거 완료
- **평균 길이**: 2.6k Qwen tokens/row → 시퀀스 패딩 손실 적음
- **추천 레시피**:
  ```
  SFTTrainer(
      model=qwen3_base,
      dataset=text_field="text",
      train_on_responses_only=True,  # assistant 턴만 로스
      epochs=2~3,
      lr=1e-5 ~ 2e-5,
      max_seq_len=4096,  # max 80k chars 행은 잘라냐거나 별도 처리
  )
  ```
- **한계**: 4,659행은 agentic 파인튜닝엔 충분하지만 일반화엔 부족 → 다른 출처 보조 필요

---

### 2.2 `Glint-Research/Fable-5-traces` ★★★★★
**cleartext CoT를 확보할 수 있는 사실상 유일 원천.**

- **포맷**: flat jsonl (`fable5_cot_merged.jsonl`), 10필드. `context` + `cot` + `output_type` + `output` + `completion` 구조라 자유도 높음
- **신호 품질**:
  - 60개 원본 세션 → 4,665행 (median 38행/세션, P90 207행, 최대 439행)
  - tool 분포 다양: Bash 1,544 / Edit 960 / Read 443 / Write 311 / WebSearch 72 / preview 87 / Task 63 / 기타
  - `output_type` 라벨로 tool-use vs text 분리 가능
- **활용 방식**:
  1. **직접 SFT 포맷팅**: `{context}` → assistant `{cot}{output}` 매핑 (lordx64가 한 방식과 동일)
  2. **세션 단위 복원**: 같은 `session` ID끼리 묶어 multi-turn 대화로 재구성
  3. **DPO pair**: 같은 context에 대해 text 응답 vs tool_use 응답 쌍 설계
  4. **Pi-trace 시각화**: `pi-traces/*.jsonl` 4,007개로 HF Data Studio에서 행동 검사
- **전처리 체크리스트**:
  - `context`가 7,022 chars로 truncate된 행 다수 → 풀 컨텍스트 필요시 `claude/history.jsonl` 또는 armand0e 원본 참조
  - `origin=hf` 953행은 imported slice라 품질 편차 있을 수 있음
  - `<think>` 태그 정규화 필요 (completion 필드에 섞여있음)
- **라이선스**: AGPL-3.0 → 폐쇄형 상업 학습 시 호환성 법무 검토 필수

---

### 2.3 `WithinUsAI/claude_mythos_distilled_25k` ★★★
**instruction QA 튜닝용. agentic이 아닌 일반 지식/추론 보강.**

- **포맷**: OpenAI 호환 `messages` 형식 → TRL/Unsloth `chat_template` 바로 적용
- **신호 품질**:
  - 카테고리 밸런스: cybersecurity 7k / advanced_coding 5.5k / agentic_planning 3.5k / general_qa 3.5k / math 3k / science 2.5k
  - assistant 응답 길이 균일 (med 1,707 chars) → instruction following에 적합
- **약점**:
  - **프롬프트 템플릿 반복 심함** (약 20~30개 템플릿이 25k행에 순환) → 모델이 템플릿에 과적합될 위험
  - assistant 응답이 "Drawing from the autonomous, frontier-level reasoning..."으로 획일적 시작 → 스타일 오염
  - Mythos mirror라 신뢰도 / 출처 모델 불확실
- **추천 활용**:
  - 소수(2~5k)만 샘플링해서 메인 SFT 데이터에 **비율 10~20%로 섞기**
  - 카테고리별로 균등 샘플링 후 중복 프롬프트 제거 (uid 기준)
  - 응답 첫 문장("Drawing from...") truncate 고려
- **비추천**: 단독 SFT → 스타일 고정화, 일반화 손상

---

### 2.4 `Complete-FABLE.5-traces-2M` ★★
**학습 직접 투입보다는 코퍼스 인덱스/중복 통제 용도.**

- **포맷**: parquet 2M행. 핵심은 `row_json`에 원본 row가 통째로 들어있고 `row_hash`/`seen_count`로 dedup 메타 제공
- **학습 활용 시나리오**:
  1. **Continual PT**: row_json을 파싱해서 텍스트 스트림으로 추출 → 대규모 코퍼스 PT. 단, 1.1GB 압축이라 전처리 비용 큼
  2. **중복 통제**: `seen_count` 기준으로 다른 데이터셋(lordx64/WithinUs)과의 교집합 검증. `row_hash`로 자체 데이터셋의 중복 제거
  3. **희소 신호 추출**: `first_source_split`별로 source 비율 보고 다양성 확보
- **한계**:
  - 대부분의 row가 위 3개 데이터셋(lordx64/Fable-5-traces/WithinUs)과 중복 → 별도 가공 없이 SFT에 넣으면 이미 학습한 신호 재주입
  - parquet을 매번 스캔하면 I/O 비용 큼 → 추출 후 별도 파일로 캐싱 권장
- **추천**: 학습 파이프라인에서 **직접 안 씀**. 데이터 카탈로그/중복 검증용으로만 보존

---

### 2.5 `HelioAI/Fable-5-Distill-Reasoning-462x` ★★
**장문 reasoning 신호원이지만 활용 효율이 나쁨.**

- **포맷**: jsonl, `{query, thinking}` (+ 선택 answer/response)
- **신호 품질**:
  - thinking 길이: median **250,385 chars** (≈62k tokens) → 초장문 reasoning 특화
  - 총 ≈26.35M 토큰 → 462행이지만 토큰 밀도 매우 높음
- **치명적 약점**:
  - **러시아어 70% (320/462행)** → 한국어/영어 타깃 모델엔 직접 사용 불가
  - 영어만 필터하면 약 **138행** (≈8M 토큰) → SFT 데이터로는 너무 적음
  - 사이버보안/생물의학 민감 주제 다수 → 공개 모델 학습 시 리스크
  - 출처 모델 불일치 (파일명은 Opus-4.7-4.8, README는 Mythos V2 주장)
  - line 192 손상 (JSON 2개 붙어있음)
- **추천 활용**:
  - 영어 행만 추출 → process supervision / PRM (Process Reward Model) 학습용 평가 데이터로 활용 가치
  - 장문 coherence 평가용 벤치마크 (학습이 아닌 eval)
- **비추천**: 메인 SFT 데이터로 사용 → 언어 분포 왜곡, 주제 편향

---

### 2.6 `armand0e/claude-fable-5-claude-code` ★
**학습 직접 사용 불가. 아카이브/검증 용도.**

- **치명적 한계**: Anthropic API의 IP 보호로 **thinking 블록이 모두 redacted** (signature만 남음). lordx64 README가 명시적으로 "SFT 부적합, thinking 빈 화면"이라 판정
- **유일 가치**:
  - 원본 Claude Code 세션 로그 → Fable-5-traces의 변환 무결성 검증용
  - 63개 세션의 tool call 패턴 / permission mode / file-history-snapshot 메타데이터 분석
  - tools_schema JSON (약 60개 tool 정의) → 자체 에이전트 tool 세트 설계 참고용
- **추천**: 학습 파이프라인에서 제외, 디버깅/레퍼런스 전용 보존

---

## 3. 추천 학습 데이터 조합 시나리오

### 시나리오 A: agentic SFT 최소 구성 (가장 빠름)
```
lordx64/agentic-distill-fable-5-sft  4,659행 (100%)
→ Qwen3 / Llama3 / Gemma3 계열에 SFTTrainer + train_on_responses_only
→ 예상: tool-call 정확도 급상승, 코딩 에이전트 벤치마크 +10~20%
→ 소요: H200 1장 기준 2~4시간
```

### 시나리오 B: agentic + 일반 지식 혼합 (균형)
```
lordx64/agentic-distill-fable-5-sft      4,659행 (70%)
WithinUs 25k (카테고리 균등 샘플링)       2,000행 (30%)  ← 중복 프롬프트 제거 후
→ 총 6,659행
→ 일반 추론/지식과 tool-call 능력 균형 확보
→ 소요: H200 1장 기준 3~5시간
```

### 시나리오 C: Fable-5 원천에서 커스텀 포맷 구축 (고품질)
```
Fable-5-traces (fable5_cot_merged.jsonl)  4,665행
  ├─ 세션 복원 (multi-turn)
  ├─ context truncate 행 보강 (claude/history.jsonl에서)
  └─ 타겟 모델 chat template으로 재포장
→ WithinUs 1,000행 보조 섞기
→ DPO pair 추가 구축 (text vs tool_use preference)
→ 총 5,000~6,000행 (가공 후)
→ 소요: 전처리 1~2일 + H200 학습 3~6시간
```

### 시나리오 D: 장문 reasoning 보강 (고급)
```
시나리오 A 또는 B 결과물에 추가:
Helio 462x → 영어 행만 필터 (~138행) → thinking만 추출
  → 기존 SFT 데이터에 5~10% 비율로 섞기
→ 장문 추론 일관성 향상, 단 언어/주제 편향 주의
```

### 시나리오 E: 대규모 continual pretrain (연구용)
```
Complete-FABLE.5-traces-2M (2M행)
  ├─ row_json 파싱 → 텍스트 추출
  ├─ seen_count 기반 중복 가중치 적용
  └─ 언어/주제 필터
→ 5~10억 토큰 규모 코퍼스 구축
→ 소요: 전처리 1주일 + GPU 다수 장비 수일
```

---

## 4. 전처리 체크리스트 (모든 데이터셋 공통)

학습 투입 전 반드시 수행할 작업:

| 항목 | 대상 | 작업 |
|---|---|---|
| **시크릿 스크럽** | Fable-5-traces, 2M, armand0e | API key / 토큰 / `.env` 내용 regex 스캔. lordx64는 Groq key 처리 완료했으나 원본 계열은 미처리 |
| **개인정보 제거** | 전체 | 이메일/전화/경로(`/home/lane/...`) 정규화. 특히 source_file 필드 |
| **언어 필터** | Helio 462x | 러시아어 행(Cyrillic 비율 >50%) 제거 → 영어 138행만 잔류 |
| **중복 제거** | WithinUs, 2M | SHA-256 또는 MinHash로 프롬프트 중복 제거. WithinUs는 템플릿 반복 심함 |
| **길이 필터** | Helio, 2M | max_seq_len 초과 행 분리 (Helio는 max 552k chars → chunking 또는 제외) |
| **CoT 정규화** | Fable-5-traces | `<think>...</think>` 태그 통일, completion 필드에서 reasoning/output 분리 |
| **툴 포맷 정규화** | lordx64 | Qwen 네이티브 `<tool_call>` 토큰으로 변환 필요 시 regex 치환 (현재는 커스텀 XML) |
| **토크나이저 정합** | 전체 | 타깃 모델 토크나이저로 special token 매핑 확인 (`<\|im_start\|>` 등) |

---

## 5. 리스크 및 주의사항

### 5.1 라이선스
- **MIT**: Complete-FABLE.5-traces-2M → 상업 사용 가능
- **AGPL-3.0**: Fable-5-traces, lordx64 → 네트워크 서비스 포함 폐쇄형 사용 시 소스 공개 의무. 법무 검토 필수
- **미표기**: armand0e, WithinUs, Helio → Anthropic 출력물 정책 + 각 카드 명시 없음. 보수적으로 AGPL 취급 권장

### 5.2 모델 정책
- `claude-fable-5`는 **2026-06-22 미국 수출통제 directive로 전역 정지**된 Anthropic preview 모델. 출력물 사용 정책 별도 확인 필요
- Anthropic ToS에 따라 출력물 재배포/상업 이용 제한 가능 → 공개 모델 학습 데이터로 사용 시 법무 리뷰 권장

### 5.3 데이터 품질 리스크
- **WithinUs 템플릿 과적합**: 프롬프트/응답 패턴 획일 → 단독 사용 금지
- **Fable-5-traces context truncate**: 7,022 chars로 잘린 행 다수 → 풀 컨텍스트 학습 불가
- **armand0e thinking 빈 화면**: SFT에 쓰면 빈 reasoning 학습 → 모델 성능 하락
- **2M 중복**: 다른 데이터셋과 90%+ 중복 예상 → 중복 학습 시 과적합
- **Helio 언어/주제 편향**: 러시아어 70% + 사이버/생물 주제 → 한국어/영어 일반 모델엔 부적합

### 5.4 평가 오염 주의
- 2M 코퍼스에 기존 벤치마크(Terminal-bench, SWE-bench 등)의 프롬프트가 섞여 있을 가능성 → eval contamination 사전 검증 필요

---

## 6. 빠른 시작 (5분)

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation

# 가장 빠른 경로: lordx64 단독 SFT
python3 -c "
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ds = load_dataset(
    'parquet',
    data_files='datasets/agentic-distill-fable-5-sft/data/train-00000-of-00001.parquet',
    split='train'
)
# ds[0]['text'] 에 Qwen chat template 적용된 시퀀스가 이미 들어있음
print(f'rows: {len(ds)}')
print(f'sample length: {len(ds[0][\"text\"])} chars')
"
```

이후 본 프로젝트(`Terminal/`)의 기존 SFT 스크립트(qwen_sft/liquid_sft/gemma4_sft 등)의 `dataset_text_field="text"` 경로에 이 parquet을 넘기면 바로 학습 시작 가능.
