[**English**](README.md) | [**한국어**](README.ko.md)

# fable5-to-lfm

> **fable5-to-lfm** — [Fable-5](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) agentic 터미널 트레이스를 [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) (8B MoE, ~1B active) 에 증류하여 터미널 에이전트 모델 **Fabliq** 라인을 구축하는 프로젝트.

이름 그대로 **Fable-5 → LFM** 방향의 증류. 참고 라인: [Qwable-9B](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5), [Qwythos-9B](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M), [gemma-4-12B-agentic-fable5](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).

---

## 🌊 Fabliq 모델 라인업

| 모델 | 단계 | 베이스 | 데이터 | 상태 | HF 링크 |
| --- | --- | --- | --- | --- | --- |
| **Fabliq-8B-Agent** | Phase-1 | ToolBench-Full-SFT-1Epoch | Fable-5 4,047 rows × 3 epoch | ✅ 완료 | [LLM-OS-Models/Fabliq-8B-Agent](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent) |
| **Fabliq-8B-Agent-Reasoning** | Phase-2 | Fabliq-8B-Agent | + WithinUs 135 + Helio 146 = 281 rows × 4 epoch | ✅ 완료 | [LLM-OS-Models/Fabliq-8B-Agent-Reasoning](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-Reasoning) |
| **Fabliq-8B-Agent-FromBase** | Phase-1B | raw LiquidAI/LFM2.5-8B-A1B | Fable-5 4,047 rows × 3 epoch | ✅ 완료 | [LLM-OS-Models/Fabliq-8B-Agent-FromBase](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-FromBase) |
| **Fabliq-8B-Agent-FromBase-Reasoning** | Phase-2B | Fabliq-8B-Agent-FromBase | + WithinUs+Helio 281 rows × 4 epoch | 🔄 학습 중 | - |
| **Fabliq-8B-Agent-Large** | Phase-3 | (Phase-2 또는 base) | Complete-FABLE.5-traces-2M 3,866 rows × 2 epoch | 📋 대기 | - |
| **Fabliq-8B-Agent-GGUF** | 양자화 | Fabliq-8B-Agent | Q4_K_M/Q5_K_M/Q6_K/Q8_0 | ✅ 완료 | [LLM-OS-Models/Fabliq-8B-Agent-GGUF](https://huggingface.co/LLM-OS-Models/Fabliq-8B-Agent-GGUF) |

---

## 📂 파일 인벤토리

### 최상위 markdown 문서

| 파일 | 설명 |
| --- | --- |
| `README.md` | 영문 버전. |
| `README.ko.md` | **이 파일** (한국어). |
| `PROGRESS_20260623.ko.md` | 진행 상황 타임라인 (Phase-1/2/3 + GPU 활용 기록). |
| `DATA_SOURCES_20260623.ko.md` | 데이터셋 6종 상세 분석 (원본 구조, 전처리 방식, 사용 현황). |
| `NEXT_STEPS_20260623.ko.md` | 후속 작업 후보 (우선순위 1-7). |
| `TRAINING_USAGE.md` | 초기 데이터셋 분석 보고서 (6종 비교표 + 시나리오 A-E). |
| `README.old.md` | 이전 README (데이터셋 분석만 있던 초기 버전). |

### 데이터 전처리 스크립트 (`*.py` in root)

| 파일 | 입력 | 출력 | 설명 |
| --- | --- | --- | --- |
| `build_fable5_to_lfm_sft.py` | `datasets/Fable-5-traces/fable5_cot_merged.jsonl` (4,665 rows) | `datasets/fable5_lfm_sft_20260623.jsonl` (4,047 rows) | **Phase-1 메인 빌더.** Glint 원본의 `context` → 멀티턴 messages, `{tool, input}` → `<\|tool_call_start\|>...<\|tool_call_end\|>`, `cot` → `<think>...</think>`. 618 short rows drop. |
| `build_withinus_lfm_sft.py` | `datasets/claude_mythos_distilled_25k/...` (25k rows) | `datasets/withinus_lfm_sft_20260623.jsonl` (135 rows) | **Phase-2 WithinUs 빌더.** 카테고리별 균등 샘플링, SHA-256 dedup (25k → 135 unique), "Drawing from the autonomous..." 템플릿 첫 문장 제거. |
| `build_helio_lfm_sft.py` | `datasets/Fable-5-Distill-Reasoning-462x/...` (462 rows) | `datasets/helio_lfm_sft_20260623.jsonl` (146 rows) | **Phase-2 Helio 빌더.** Cyrillic 비율 30% 미만 필터 (Russian 우세 행 제거), `<think>` wrapping, line 192 손상 스킵. |
| `build_fable5_2m_to_lfm_sft.py` | `datasets/Complete-FABLE.5-traces-2M/data/train.parquet` (2M events) | `datasets/fable5_2m_lfm_sft_20260623.jsonl` (3,866 rows) | **Phase-3 빌더.** Parquet `row_json`에서 Glint 구조화 row (`output_type` 키 포함) 만 추출, `seen_count<=100`. build_fable5_to_lfm_sft.convert_row 재사용. |

### `training/` — LFM2.5 학습 스크립트 (harness-1에서 복사)

| 파일 | 설명 |
| --- | --- |
| `training/train_lfm25_rlvr_json_sft.py` | **메인 학습 스크립트.** full parameter SFT + LoRA 지원. FSDP full_shard, Lfm2MoeDecoderLayer auto_wrap, activation_checkpointing. `apply_chat_template(messages)` 로 토큰화. final_model/ (full) 또는 final_lora/ (lora) 저장. |
| `training/build_lfm25_agentic_sft.py` | agentic SFT 데이터 빌더 (터미널 툴 사용 JSONL). |
| `training/build_lfm25_rlvr_json_sft.py` | RLVR SFT 데이터 빌더. |
| `training/mix_lfm25_sft_jsonl.py` | 여러 JSONL을 mix/shuffle 하는 유틸리티. |

### `eval_scripts/` — LFM2.5 평가 스크립트 (harness-1에서 복사)

| 파일 | 설명 |
| --- | --- |
| `eval_scripts/eval_lfm25_agentic_vllm.py` | **agentic tool-use 평가.** vLLM HTTP로 multi-turn 터미널 trajectory 실행. Harness-1 retrieval-curation 시스템 프롬프트 사용. |
| `eval_scripts/eval_lfm25_rlvr_retrieval_vllm.py` | retrieval-curation 평가 (단발 응답). |

### `scripts/` — 실행 셸 스크립트

| 파일 | 설명 |
| --- | --- |
| `scripts/run_fable5_full_sft_20260623.sh` | **Phase-1 러너.** ToolBench-Full-SFT-1Epoch → Fable-5 SFT. 8 H200, max_seq 8192, epochs 3, LR 5e-7, batch 2×accum 4. |
| `scripts/run_phase2_reasoning_sft_20260623.sh` | **Phase-2 러너.** Fabliq-8B-Agent → + WithinUs+Helio reasoning. 8 H200, max_seq 8192, epochs 4, LR 3e-7. |
| `scripts/run_fable5_from_base_sft_20260623.sh` | **Phase-1B 러너 (ablation).** raw LiquidAI/LFM2.5-8B-A1B → Fable-5 SFT. ToolBench foundation 효과 분리. LR 1e-6 (base에서 직접이라 더 높게). |
| `scripts/run_phase2b_reasoning_sft_20260623.sh` | **Phase-2B 러너.** Fabliq-8B-Agent-FromBase + reasoning expansion (Phase-2를 FromBase 변형에 그대로 적용). |
| `scripts/run_fable5_2m_phase3_sft_20260623.sh` | **Phase-3 러너 (대기).** Phase-1B/Phase-2 + 2M traces scale-out. |
| `scripts/convert_lfm25_gguf.sh` | 기존 Liquid-CLI에서 복사한 GGUF 변환기. LFM2.5 tokenizer hash (9e45...) 패치 포함. |
| `scripts/convert_fabliq_gguf.sh` | **Fabliq 전용 GGUF 변환기.** `convert_lfm25_gguf.sh` 기반, 모델명/출력 경로 파라미터화. Q4_K_M/Q5_K_M/Q6_K/Q8_0 생성 + HF 업로드 옵션. |
| `scripts/convert_fabliq_to_gguf.sh` | 초기 단순 변환 스크립트 (tokenizer 패치 없어서 실패 - reference용 보관). |

### `datasets/` — 원본 + 전처리 결과

| 디렉토리 | 원본 출처 | 행 수 | 비고 |
| --- | --- | --- | --- |
| `datasets/Fable-5-traces/` | [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | 4,665 | Phase-1 입력. `fable5_cot_merged.jsonl` 외 pi-traces 등. |
| `datasets/claude_mythos_distilled_25k/` | WithinUsAI | 25,000 | Phase-2 WithinUs 원본. |
| `datasets/Fable-5-Distill-Reasoning-462x/` | HelioAI | 462 | Phase-2 Helio 원본. |
| `datasets/Complete-FABLE.5-traces-2M/` | Glint-Research | 2,006,487 | Phase-3 입력. Parquet + jsonl.gz. |
| `datasets/agentic-distill-fable-5-sft/` | lordx64 | 4,659 | 미사용 (Glint 원본보다 깨끗하지 않음). |
| `datasets/claude-fable-5-claude-code/` | armand0e | 18,370 | 미사용 (Phase-3 대체 후보). |

**전처리 결과 JSONL:**

| 파일 | rows | 용도 |
| --- | --- | --- |
| `datasets/fable5_lfm_sft_20260623.jsonl` | 4,047 | Phase-1 입력 |
| `datasets/withinus_lfm_sft_20260623.jsonl` | 135 | Phase-2 입력 (WithinUs 부분) |
| `datasets/helio_lfm_sft_20260623.jsonl` | 146 | Phase-2 입력 (Helio 부분) |
| `datasets/phase2_reasoning_lfm_sft_20260623.jsonl` | 281 | Phase-2 입력 (WithinUs+Helio 결합) |
| `datasets/fable5_mixed_lfm_sft_20260623.jsonl` | 4,328 | Phase-1+2 통합 (대안) |
| `datasets/fable5_2m_lfm_sft_20260623.jsonl` | 3,866 | Phase-3 입력 |

---

## 🔧 환경 설정

```bash
# 학습 환경 (.liquid-sft-env) — 프로젝트 루트에 있음
cd /home/work/.projects/LLM-OS-Models/Terminal
source .liquid-sft-env/bin/activate

# 의존성 (이미 설치됨)
# - torch 2.10+cu128 (FSDP full_shard)
# - transformers 5.5+ (Lfm2MoeForCausalLM)
# - trl 0.16+, datasets, accelerate
# - lm-eval 0.4.11, vllm (평가용)
```

**하드웨어:** 8× NVIDIA H200 141GB (full_shard 분산)

---

## 🚀 학습 실행 (재현)

### Phase-1: Fable-5 Agentic Foundation (✅ 완료)

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_full_sft_20260623.sh
```

### Phase-2: Reasoning Expansion (✅ 완료)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_phase2_reasoning_sft_20260623.sh
```

### Phase-1B: Base → Fable-5 (Ablation, ✅ 완료)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_from_base_sft_20260623.sh
```

### Phase-2B: FromBase + Reasoning (🔄 학습 중)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_phase2b_reasoning_sft_20260623.sh
```

### Phase-3: 2M Traces Scale-Out (📋 대기)

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_fable5_2m_phase3_sft_20260623.sh
```

---

## 📊 데이터 전처리 파이프라인

### Phase-1 데이터 빌드

```bash
python fable_distillation/build_fable5_to_lfm_sft.py \
  --input fable_distillation/datasets/Fable-5-traces/fable5_cot_merged.jsonl \
  --output fable_distillation/datasets/fable5_lfm_sft_20260623.jsonl
# 결과: 4,047 rows (618 short-context drop)
```

### Phase-2 데이터 빌드

```bash
python fable_distillation/build_withinus_lfm_sft.py
python fable_distillation/build_helio_lfm_sft.py

# 결합
head -n 135 fable_distillation/datasets/withinus_lfm_sft_20260623.jsonl > phase2.jsonl
cat fable_distillation/datasets/helio_lfm_sft_20260623.jsonl >> phase2.jsonl
mv phase2.jsonl fable_distillation/datasets/phase2_reasoning_lfm_sft_20260623.jsonl
```

### Phase-3 데이터 빌드

```bash
python fable_distillation/build_fable5_2m_to_lfm_sft.py \
  --max-rows 50000 --seen-count-max 100
# 결과: 2M events에서 3,866 rows 추출
```

---

## 🔍 평가 (Eval)

### tb2_lite 터미널 task 평가

```bash
# vLLM으로 모델 서빙
bash Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh &

# 터미널 task replay 평가
python tb2_lite/scripts/replay_eval_lfm_vllm.py \
  --model-path $MODEL_PATH \
  --vllm-base-url http://127.0.0.1:8137/v1 \
  --dataset-path tb2_lite/data/replay_dev_20.jsonl \
  --output-jsonl /tmp/eval.jsonl
```

### 표준 벤치마크 (MMLU, GPQA, HumanEval) — lm-eval-harness

```bash
vllm serve $MODEL_PATH --max-model-len 8192 --dtype bfloat16 --port 8000 &

lm_eval --model vllm \
  --model_args pretrained=$MODEL_PATH,base_url=http://localhost:8000/v1,dtype=bfloat16 \
  --tasks mmlu,gpqa_diamond,humaneval \
  --batch_size auto
```

---

## 📦 GGUF 변환 (CPU 병렬)

```bash
bash fable_distillation/scripts/convert_fabliq_gguf.sh
# 출력: /home/work/.data/gguf/Fabliq-8B-Agent/Fabliq-8B-Agent.{Q4_K_M,Q5_K_M,Q6_K,Q8_0}.gguf
# HF 자동 업로드: UPLOAD_REPO_ID=LLM-OS-Models/Fabliq-8B-Agent-GGUF
```

---

## 📈 학습 결과 요약

### Phase-1 (Fable-5 Agentic, 4,047 rows × 3 epoch)
- final train_loss: **1.277**
- train_runtime: 831초 (≈14분)
- global_step: 192
- LR: 5e-7 constant, batch 2×accum 4×8 GPU = global 64

### Phase-2 (Reasoning Expansion, 281 rows × 4 epoch)
- final train_loss: **~1.6** (데이터 작고 reasoning이라 loss 높음 - 정상)
- train_runtime: ~6분
- global_step: 20
- LR: 3e-7 (Phase-1보다 낮춰 forgetting 방지)

### Phase-1B (Base → Fable-5, ablation)
- Phase-1과 동일 데이터, 베이스만 raw LiquidAI/LFM2.5-8B-A1B (ToolBench 안 거침)
- LR 1e-6 (base에서 직접이라 더 높게)

---

## 🔗 관련 문서

- [진행 상황 타임라인](./PROGRESS_20260623.ko.md)
- [데이터 소스 상세](./DATA_SOURCES_20260623.ko.md)
- [후속 작업 계획](./NEXT_STEPS_20260623.ko.md)
- [TRAINING_USAGE.md (초기 데이터셋 분석)](./TRAINING_USAGE.md)
- [영문 README](./README.md)

---

## 🏷️ 라이선스

Apache 2.0 (LFM2.5-8B-A1B base 상속). 학습 데이터인 Fable-5 traces는 Claude 모델에서 증류된 데이터로, 원본 Anthropic 사용 정책을 따름.
