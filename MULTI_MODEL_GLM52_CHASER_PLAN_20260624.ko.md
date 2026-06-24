# Multi-Model GLM-5.2 Chaser Plan (2026-06-24)

## 현재 판단

여러 모델과 여러 방식으로 제대로 비교하려면 며칠이 걸린다. 오늘 할 일은 GPU를 놀리지 않는 장기 SFT를 먼저 태우고, 그 사이에 전처리/학습 스크립트/문서/평가 루프를 정리하는 것이다.

현재 GPU에서 돌고 있는 1순위 실험:

- `scripts/run_glm52_chaser_mix_sft_20260624.sh`
- base: `Fabliq-8B-Agent-Reasoning`
- data: `datasets/glm52_chaser_terminal_toolmix_20260624.jsonl`
- rows: 11,416
- method: 8 GPU full SFT, max sequence 16,384, 8 epochs, LR 2e-7
- follow-up eval: TB2-lite vLLM replay 자동 실행

## 왜 며칠 걸리는가

| 범위 | 예상 |
| --- | ---: |
| LFM 계열 추가 SFT + vLLM TB2-lite 평가 | 반나절-1일 |
| Gemma 4 12B / Qwen3.5 9B LoRA smoke + 전처리 검증 | 1일 |
| Gemma/Qwen long LoRA 또는 full SFT 후보 2-4개 | 2-4일 |
| DiffusionGemma 26B-A4B LoRA smoke + NeMo/DLLM 안정화 | 1-2일 |
| DiffusionGemma long run + 별도 평가 루프 | 2-4일 |
| UniRL/GRPO류 RL 실험 | 최소 2-5일, Terminal-Bench rollouts까지 넣으면 더 길어짐 |

즉, “작동하는 후보 하나”는 오늘/내일 가능하지만, GLM-5.2급 benchmark를 공개적으로 주장할 만한 비교표는 3-7일을 잡는 게 현실적이다.

## 외부 기준

- GLM-5.2 model card benchmark: https://huggingface.co/zai-org/GLM-5.2#benchmark
- Terminal-Bench: https://www.tbench.ai/
- HF Agent Traces: https://huggingface.co/docs/hub/en/agent-traces
- Hermes agent traces dataset: https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces
- Gemma 4 12B IT: https://huggingface.co/google/gemma-4-12B-it
- Qwen3.5 9B: https://huggingface.co/Qwen/Qwen3.5-9B
- DiffusionGemma 26B-A4B IT: https://huggingface.co/google/diffusiongemma-26B-A4B-it

GLM-5.2 card에서 바로 노릴 만한 공개 benchmark 축은 Terminal Bench 2.1, MCP-Atlas, Tool-Decathlon, coding/SWE 계열이다. 로컬에서는 우선 TB2-lite vLLM replay로 빠르게 회귀를 보고, 상위 후보만 public Terminal-Bench/agent harness로 올리는 구조가 맞다.

## 새로 준비한 전처리

### Hermes agent traces 변환

스크립트:

- `scripts/build_hermes_agent_traces_mix_20260624.py`

입력:

- `lambda/hermes-agent-reasoning-traces`
- configs: `kimi`, `glm-5.1`
- upstream 규모: Kimi 7,646, GLM-5.1 7,055

출력:

- `datasets/hermes_agent_traces_chat_20260624.jsonl`
- `datasets/hermes_agent_traces_chat_20260624.meta.json`
- 변환 결과: 6,835 rows (Kimi 6,092, GLM-5.1 743)
- JSONL 본문 크기: 약 555MB라 Git에는 넣지 않고 ignored 산출물로 둔다. meta와 변환 스크립트만 커밋한다.

기본 정책:

- ShareGPT `human/gpt/tool/system`을 `user/assistant/system` 중심으로 정규화
- tool 실행 결과는 기본적으로 `user` message의 `Tool result:` 블록으로 변환
- upstream tool schema는 system message에 주입
- 마지막 turn이 assistant가 아닌 row는 SFT target으로 쓰기 애매해서 제외

## 새로 준비한 학습 큐

### Gemma/Qwen smoke LoRA

스크립트:

- `scripts/run_multifamily_sft_smoke_20260624.sh`

예시:

```bash
RUN_NOW=1 MODEL_PRESET=gemma4_12b_it \
  bash fable_distillation/scripts/run_multifamily_sft_smoke_20260624.sh

RUN_NOW=1 MODEL_PRESET=qwen35_9b \
  bash fable_distillation/scripts/run_multifamily_sft_smoke_20260624.sh
```

내부 trainer:

- `training/train_multifamily_chat_sft.py`

목적:

- Gemma 4 12B IT: `AutoModelForMultimodalLM`
- Qwen3.5 9B: `AutoModelForImageTextToText`
- 우선 LoRA 2,000 rows / 100 steps smoke
- tokenizer/chat template/data collation 문제가 없으면 long run으로 확장

### DiffusionGemma smoke LoRA

스크립트:

- `scripts/run_diffusiongemma_fable_lora_20260624.sh`

config:

- `configs/diffusiongemma_26b_a4b_fable_agent_lora_smoke_20260624.yaml`

목적:

- NeMo AutoModel DLLM SFT recipe로 26B-A4B LoRA wiring 확인
- 기존 `/home/work/.data/harness1/envs/diffusiongemma-nemo` 환경 사용
- smoke 성공 후 max_steps/seq_length/dataset scale up

### Post-chaser 자동 큐

스크립트:

- `scripts/run_post_chaser_multimodel_queue_20260624.sh`

역할:

- 현재 `glm52_chaser_mix` launcher PID가 끝날 때까지 대기
- chaser script의 자동 vLLM 평가까지 끝난 뒤 Gemma 4 12B IT smoke 실행
- 이어서 Qwen3.5 9B smoke 실행
- 마지막으로 DiffusionGemma 26B-A4B smoke 실행

실행:

```bash
mkdir -p fable_distillation/logs/20260624_post_chaser_multimodel_queue
setsid env RUN_NOW=1 bash fable_distillation/scripts/run_post_chaser_multimodel_queue_20260624.sh \
  > fable_distillation/logs/20260624_post_chaser_multimodel_queue/nohup.log 2>&1 &
```

## 추천 실행 순서

1. 현재 `glm52_chaser_mix` full SFT 완료까지 유지한다.
2. 자동 TB2-lite vLLM 평가 결과가 나오면 README와 benchmark 문서에 즉시 반영한다.
3. 점수가 51.59를 넘으면 해당 checkpoint를 우선 HF/model card/GGUF 후보로 잡는다.
4. 점수가 부족하면 같은 data로 LR 1e-7/5e-7, epoch 2-4 early checkpoint 평가를 추가한다.
5. GPU가 비면 Gemma 4 12B IT smoke를 먼저 돌린다.
6. Qwen3.5 9B smoke를 돌린다. Qwen card가 vLLM tool-call parser를 명시하므로, 추후 eval은 vLLM `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` 쪽도 확인한다.
7. DiffusionGemma는 NeMo DLLM 경로로 smoke 후 long run을 별도로 잡는다.
8. 위 SFT 후보 중 TB2-lite 상위 2개만 Terminal-Bench 2.1/ToolBench류 긴 평가로 올린다.
9. RL/UniRL은 SFT 후보가 확정된 뒤 reward를 TB2-lite/JSON validity/command F1로 구성해서 시작한다.

## 성공 기준

단기:

- TB2-lite vLLM score `51.59` 초과
- cmd F1 `0.53` 이상
- valid JSON `80%` 이상
- first command `52%` 이상

중기:

- public Terminal-Bench 2.1/Terminus 계열에서 비교 가능한 pass@1 확보
- BFCL/TIR/Tool-Decathlon류 tool-call 벤치로 모델 카드에 올릴 수 있는 숫자 확보

장기:

- Gemma/Qwen/DiffusionGemma 중 하나를 Fabliq/Fable/Hermes agent specialty로 packaging
- HF model card에 데이터/평가/사용법/vLLM command를 넣고 GGUF 또는 adapter release까지 정리
