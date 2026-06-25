# Claude Code 인계용 다음 작업 총정리 (2026-06-25)

이 문서는 Claude Code나 다른 에이전트가 `fable_distillation` 작업을 바로 이어받을 수 있게 만든 실행 중심 인계 문서다. GPU 학습을 바로 시작하지 말고, 아래 순서대로 상태 확인, 문서/코드 정리, 짧은 smoke, 긴 run 순서로 진행한다.

## 절대 기준

- 작업 루트는 `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation`이다.
- 사용자가 따로 지시하지 않으면 새 실험 파일, 문서, helper 코드는 모두 `fable_distillation/` 아래에 둔다.
- Docker는 사용할 수 없다.
- GPU를 쓰기 전에는 반드시 dry-run stop check와 `nvidia-smi`를 먼저 본다.
- 대용량 산출물은 커밋하지 않는다. `datasets/`, `logs/`, 일부 local benchmark artifact는 `.gitignore`에 들어가 있다.
- 기존 untracked 파일은 함부로 삭제하지 않는다. 현재 남아 있는 `scripts/run_fabliq_lm_eval_wave2_20260624.sh`는 이전 작업 산출물로 보이며, 삭제하지 말고 내용 확인 후 필요한 경우 별도 커밋한다.

## 첫 5분 확인 명령

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation

git status --short
git log --oneline -8

# fable_distillation 관련 GPU/eval job만 dry-run 확인
bash scripts/stop_fable_gpu_jobs_20260624.sh

# 실제 GPU compute app 확인
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

현재 기준 GPU compute app은 없어야 한다. GPU 메모리는 GPU 0-7 모두 1MiB 수준이어야 정상이다.

## 현재 커밋 기준

최근 중요 커밋:

- `1fd8574` `Add fable GPU stop handoff cleanup`
- `87c4484` `Document experiment handoff after GPU stop`
- `03830aa` `Skip PEFT optimizer checkpoint for DiffusionGemma`
- `aadc8dc` `Plan DiffusionGemma strength-task LoRA`
- `7971963` `Add Qwen LoRA merge and eval watcher`
- `4e76bb2` `Ignore local tokenized cache`
- `9522eb6` `Fix Qwen text-only LoRA DDP training`
- `18bdff1` `Stabilize dLLM eval and Qwen ChatML training`

현재 `git status --short`에서 보일 수 있는 항목:

```text
?? scripts/run_fabliq_lm_eval_wave2_20260624.sh
```

이 파일은 아직 추적되지 않는다. 다음 담당자는 먼저 내용을 읽고 다음 중 하나를 결정한다.

- 실제로 필요한 Fabliq lm-eval wave2 runner라면 README/문서에 설명을 추가하고 커밋한다.
- 임시 파일이면 `.gitignore`에 넣을지, 보존만 할지 사용자에게 확인한다.
- 삭제는 하지 않는다.

## 핵심 문서 지도

- 현재 실험 전체 인계: `EXPERIMENT_HANDOFF_20260624.ko.md`
- 현재 실험 상태: `CURRENT_EXPERIMENT_STATUS_20260624.ko.md`
- DiffusionGemma dLLM 평가 계획: `DIFFUSIONGEMMA_DLLM_EVAL_PLAN_20260624.ko.md`
- DiffusionGemma strength-task 방법론: `DIFFUSIONGEMMA_STRENGTH_TASKS_20260624.ko.md`
- GLM-5.2 chaser 결과: `GLM52_CHASER_EXPERIMENT_20260624.ko.md`
- 멀티모델 계획: `MULTI_MODEL_GLM52_CHASER_PLAN_20260624.ko.md`
- TB2-lite vLLM benchmark 설명: `TB2_VLLM_BENCHMARK_20260624.ko.md`
- 최종/요약 문서: `FINAL_REPORT_20260624.ko.md`, `RESULTS_SUMMARY_20260624.ko.md`

## 중요한 경로

### repo 내부

```text
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
```

주요 파일:

```text
configs/diffusiongemma_26b_a4b_strength_lora_20260624.yaml
scripts/build_diffusiongemma_strength_mix_20260624.py
scripts/diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py
scripts/run_diffusiongemma_strength_lora_20260624.sh
scripts/stop_fable_gpu_jobs_20260624.sh
scripts/merge_multifamily_lora_for_vllm.py
scripts/watch_qwen35_lora_merge_eval_20260624.sh
scripts/run_qwen35_9b_tb2_vllm_sharded_20260624.sh
scripts/run_glm52_chaser_mix_sft_20260624.sh
```

생성 데이터:

```text
datasets/diffusiongemma_strength_mix_20260624.jsonl
datasets/diffusiongemma_strength_mix_20260624.meta.json
```

이 파일들은 `.gitignore` 대상이다. 필요하면 다시 생성 가능하다.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/work/.data/harness1/envs/diffusiongemma-nemo/bin/python \
  scripts/build_diffusiongemma_strength_mix_20260624.py \
  --output datasets/diffusiongemma_strength_mix_20260624.jsonl \
  --meta datasets/diffusiongemma_strength_mix_20260624.meta.json
```

### 외부 환경/모델 캐시

DiffusionGemma NeMo 학습 환경:

```text
/home/work/.data/harness1/envs/diffusiongemma-nemo
```

DiffusionGemma Automodel checkout:

```text
/home/work/.data/harness1/diffusiongemma_retriever/Automodel
```

DiffusionGemma Transformers 평가용 uv env:

```text
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/.venvs/diffusiongemma-transformers-cu128
```

HF cache:

```text
/home/work/.data/harness1/hf_home
/home/work/.data/harness1/hf_home/hub
```

## 현재 결과 요약

### TB2-lite 로컬 기준

TB2-lite는 public Terminal-Bench 2.1과 같은 공식 harness가 아니다. 303개 terminal next-action replay prompt를 빠르게 보는 내부 회귀 벤치다.

- 현재 local best: `Fabliq-8B-Agent-Reasoning`, score `51.59`
- GLM-5.2 chaser final: score `51.13`
- GLM-5.2 chaser checkpoint-1400: score `50.56`
- Qwen3.5-9B base: score `36.75`
- DiffusionGemma base Transformers dLLM corrected full: score `25.12`

### 공식 공개 benchmark 목표

사용자가 원한 방향은 단순 내부 TB2-lite가 아니라 공식 공개 benchmark에서 의미 있는 성능을 만드는 것이다.

DiffusionGemma 공식 공개 축:

- Tau2 average over 3
  - DiffusionGemma base 공개값: `56.2`
  - 1차 목표: `60+`
  - stretch 목표: Gemma 4 26B 공개값 `68.2+`
  - HF benchmark 섹션: `https://huggingface.co/google/diffusiongemma-26B-A4B-it#benchmark-results`
- LiveCodeBench v6
  - DiffusionGemma base 공개값: `69.1`
  - 1차 목표: `72+`
  - stretch 목표: `77.1+`

GLM-5.2 추격 축:

- Tool-Decathlon `48.2+`
- MCP-Atlas Public Set `76.8+`
- Terminal-Bench 2.1 `81.0+/82.7+`
- HF benchmark 섹션: `https://huggingface.co/zai-org/GLM-5.2#benchmark`

## 완료된 작업

### DiffusionGemma base 실행

Docker 없이 Transformers backend로 DiffusionGemma dLLM 평가를 돌리는 데 성공했다.

- corrected full run:
  - score `25.12`
  - Cmd F1 `0.2980`
  - First Cmd `14.2%`
  - Valid JSON `55.1%`
  - probe speed `97.88 tok/s`
- 해석:
  - Docker-free 실행과 decode fix는 성공.
  - TB2-lite next-action 태스크는 DiffusionGemma base와 맞지 않았다.
  - 그래서 다음 학습을 structured repair/tool-call/code 쪽으로 돌렸다.

### DiffusionGemma strength-task 데이터

데이터 빌더:

```text
scripts/build_diffusiongemma_strength_mix_20260624.py
```

출력:

```text
datasets/diffusiongemma_strength_mix_20260624.jsonl
datasets/diffusiongemma_strength_mix_20260624.meta.json
```

주요 수치:

- total rows: `11,352`
- Fable source rows: `4,970`
- task 구성:
  - `original_terminal_or_tool_trace`: `6,050`
  - `structured_repair`: `4,565`
  - `original_code_or_tool_trace`: `737`
- source 구성:
  - `fable5_terminal`: `4,970`
  - `glm52_terminal_toolmix`: `5,076`
  - `hermes_agent_function_code`: `1,242`
  - `phase2_reasoning_code`: `64`

### DiffusionGemma LoRA 학습

학습 config:

```text
configs/diffusiongemma_26b_a4b_strength_lora_20260624.yaml
```

runner:

```text
scripts/run_diffusiongemma_strength_lora_20260624.sh
```

NeMo PEFT optimizer checkpoint 우회 wrapper:

```text
scripts/diffusiongemma_finetune_skip_peft_optim_ckpt_20260624.py
```

retry1:

- run id: `20260624_diffusiongemma_strength_lora_fable_structured_repair_retry1`
- step `199`까지 정상 학습.
- step `200` checkpoint 저장 중 optimizer nested dict를 `safetensors`로 저장하려다 실패.
- 실패 메시지 핵심:

```text
ValueError: Key `optim` is invalid, expected torch.Tensor but received <class 'dict'>
```

retry1 adapter는 남아 있다.

```text
/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-20260624/epoch_0_step_199/model/adapter_model.safetensors
```

no-optimizer retry:

- log: `logs/debug_nooptim_foreground_223859/train.log`
- checkpoint dir:

```text
/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-NoOptim-20260624/
```

- KST `2026-06-24 22:53:11` step `0` 시작.
- 사용자가 중단 요청.
- KST `2026-06-24 22:59:17` graceful stop signal 수신.
- step `112`까지 진행.
- rank0가 checkpoint 저장.
- wrapper가 optimizer 저장을 건너뛰고 adapter 저장 성공.

step 112 adapter:

```text
/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-NoOptim-20260624/epoch_0_step_112/model/adapter_model.safetensors
```

주의:

- step 112 checkpoint는 optimizer state가 없다.
- adapter 평가/분석에는 쓸 수 있다.
- 그대로 optimizer resume은 어렵다.
- graceful stop 뒤 torch elastic launcher가 `SignalException` traceback을 남긴다. 이건 중단 신호 때문이고, adapter 저장 자체는 성공했다.

### Qwen3.5-9B LoRA

학습 run:

```text
20260624_qwen35_9b_glm52_terminalmix_lora_sft300_chatml_ddptrue
```

출력 LoRA:

```text
/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-20260624/final_lora
```

merged checkpoint:

```text
/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-Merged-20260624
```

상태:

- SFT300 완료.
- train loss `0.6287`.
- LoRA merge 완료.
- vLLM load 실패.
- 오류 성격: `Qwen3_5ForConditionalGeneration` 또는 VLM/text wrapper 계열 config/weight key mismatch.
- 다음 작업은 text-only export 경로를 다시 만드는 것이다.

### GLM-5.2 chaser

full SFT output:

```text
/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624
```

결과:

- final score `51.13`
- checkpoint-1400 score `50.56`
- local best `51.59` 미달.

## 해야 할 작업 우선순위

### P0. GPU를 쓰지 않는 정리

1. `scripts/run_fabliq_lm_eval_wave2_20260624.sh`를 읽고 처리 방침을 정한다.

```bash
sed -n '1,240p' scripts/run_fabliq_lm_eval_wave2_20260624.sh
```

결정:

- 계속 쓸 스크립트면 README와 관련 문서에 추가하고 커밋.
- 임시 파일이면 사용자에게 보존/삭제/ignore 중 선택을 확인.

2. Qwen merge 실패 원인을 코드 기준으로 좁힌다.

읽을 파일:

```text
scripts/merge_multifamily_lora_for_vllm.py
scripts/watch_qwen35_lora_merge_eval_20260624.sh
logs/20260624_qwen35_9b_glm52_terminalmix_lora_sft300_chatml_ddptrue_post_eval/watcher.nohup.log
```

확인할 것:

- merged config의 `architectures`
- model weight key prefix
- vLLM이 기대하는 text-only `AutoModelForCausalLM` 구조
- `language_model.model...` missing weight가 어디서 생겼는지

3. DiffusionGemma step 112 adapter 평가 경로를 설계한다.

먼저 읽을 파일:

```text
/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-NoOptim-20260624/epoch_0_step_112/model/adapter_config.json
/home/work/.data/harness1/models/diffusiongemma-26b-a4b-it__Fable-StructuredRepair-LoRA-NoOptim-20260624/epoch_0_step_112/model/automodel_peft_config.json
scripts/diffusiongemma_transformers_eval.py
scripts/run_diffusiongemma_dllm_eval_20260624.sh
```

목표:

- base DiffusionGemma + LoRA adapter를 Transformers 또는 NeMo 경로로 로드할 수 있는지 확인.
- 바로 full TB2-lite를 돌리기 전에 10-20개 structured repair prompt만 먼저 평가.

### P1. DiffusionGemma 재개 방향 결정

세 선택지가 있다.

#### 선택 A: step 112 adapter 평가만

목적:

- 짧게 학습한 adapter가 structured JSON/tool-call repair를 조금이라도 개선했는지 확인.
- GPU를 오래 쓰기 전에 방향성만 검증.

해야 할 코드:

```text
scripts/eval_diffusiongemma_strength_adapter_20260625.py
```

권장 기능:

- 입력 adapter path를 CLI 인자로 받기.
- 10-50개 structured repair prompt를 생성하거나 `datasets/diffusiongemma_strength_mix_20260624.jsonl`에서 샘플링.
- 출력 JSON validity, tool-call key recovery, blank output 여부 기록.
- 결과를 `benchmarks/20260625_diffusiongemma_step112_adapter_probe/` 아래 저장.

#### 선택 B: base부터 1200 step 재시작

장점:

- 가장 단순하다.
- optimizer state 문제가 없다.

단점:

- step 112까지의 학습은 버린다.
- 초기 모델 로딩이 13분 정도 걸린다.

실행 전 반드시 dry-run:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
bash scripts/run_diffusiongemma_strength_lora_20260624.sh
```

실행 명령:

```bash
RUN_ID=20260625_diffusiongemma_strength_lora_fable_structured_repair_nooptim_full \
RUN_NOW=1 \
SKIP_BUILD_DATASET=1 \
bash scripts/run_diffusiongemma_strength_lora_20260624.sh
```

모니터:

```bash
tail -f logs/20260625_diffusiongemma_strength_lora_fable_structured_repair_nooptim_full/train.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

중단:

```bash
RUN_NOW=1 bash scripts/stop_fable_gpu_jobs_20260624.sh
```

#### 선택 C: step 112 adapter를 초기값으로 이어가기

장점:

- 이미 내려간 loss를 활용할 수 있다.

단점:

- optimizer state가 없어서 진짜 resume은 아니다.
- adapter를 base에 로드한 뒤 새 optimizer로 이어가는 custom path가 필요하다.

해야 할 코드:

```text
scripts/run_diffusiongemma_strength_lora_from_adapter_20260625.sh
scripts/diffusiongemma_finetune_from_adapter_skip_optim_ckpt_20260625.py
```

먼저 확인할 것:

- NeMo AutoModel recipe가 기존 PEFT adapter load를 지원하는지.
- 지원하지 않으면 wrapper에서 model setup 후 adapter weight를 로드해야 하는지.
- adapter key format이 `PatchedLinearLoRA`와 바로 맞는지.

## P2. VRAM 활용 개선 계획

현재 H200 8장 기준 VRAM 사용량은 낮다.

- 로그 기준 대략 `21-50GiB/GPU`.
- step time은 대략 3초 전후.
- tps는 대략 5k-10k aggregate.

다음 긴 run 전에는 VRAM sweep을 짧게 한다.

새 config 후보:

```text
configs/diffusiongemma_26b_a4b_strength_lora_vram_sweep_20260625.yaml
```

실험 원칙:

- max_steps는 `20` 또는 `50`.
- checkpoint는 실험 중간 저장을 피하려면 `ckpt_every_steps`를 `100000`으로 둔다.
- OOM 없이 step time과 VRAM을 비교한다.
- smoke 결과는 `DIFFUSIONGEMMA_STRENGTH_TASKS_20260624.ko.md` 또는 새 `DIFFUSIONGEMMA_VRAM_SWEEP_20260625.ko.md`에 적는다.

비교 후보:

1. 현재 기준 유지:
   - `global_batch_size: 8`
   - `local_batch_size: 1`
   - `seq_length: 4096`
2. local batch 증가:
   - `global_batch_size: 16`
   - `local_batch_size: 2`
   - `seq_length: 4096`
3. sequence length 증가:
   - `global_batch_size: 8`
   - `local_batch_size: 1`
   - `seq_length: 6144` 또는 `8192`
4. 둘 다 증가:
   - `global_batch_size: 16`
   - `local_batch_size: 2`
   - `seq_length: 6144`

주의:

- DiffusionGemma는 `canvas_length: 256`, `block_size: 256`, `pad_seq_len_divisible: 256` 설정이 중요하다.
- OOM이 나면 먼저 `seq_length`보다 `local_batch_size`를 낮춘다.
- 사용자가 “지금은 그대로”라고 한 run은 이미 중단됐다. 다음 run에서는 별도 smoke로 조정해도 된다.

## P3. Qwen3.5 vLLM export mismatch 수정

현재 문제:

- LoRA 학습과 merge는 됐다.
- merged checkpoint를 vLLM이 로드하지 못한다.
- 오류 핵심은 text-only causal LM이 아니라 VLM/text wrapper 계열 weight key를 기대하는 형태다.

작업 순서:

1. merged checkpoint config 확인.

```bash
python - <<'PY'
from pathlib import Path
import json
p = Path('/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-Merged-20260624/config.json')
print(p)
print(json.dumps(json.loads(p.read_text()), indent=2)[:4000])
PY
```

2. weight key prefix 확인.

```bash
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
root = Path('/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-Merged-20260624')
for f in sorted(root.glob('*.safetensors'))[:1]:
    print(f)
    with safe_open(f, framework='pt') as sf:
        for k in list(sf.keys())[:80]:
            print(k)
PY
```

3. `scripts/merge_multifamily_lora_for_vllm.py`에서 Qwen3.5가 `AutoModelForCausalLM` text-only로 저장되도록 고친다.

검증 목표:

- vLLM이 checkpoint를 로드한다.
- TB2-lite sharded eval이 끝까지 돈다.
- base Qwen3.5 score `36.75`보다 올라가는지 본다.

평가 runner:

```text
scripts/run_qwen35_9b_tb2_vllm_sharded_20260624.sh
```

## P4. GLM-5.2 chaser 후속

현재 결과:

- final `51.13`
- checkpoint-1400 `50.56`
- local best `51.59` 미달.

후속 후보:

1. 더 이른 checkpoint가 남아 있으면 평가한다.
2. 데이터 비율을 바꾼다.
   - Fable terminal 비중 증가.
   - GLM-5.2 imitation 비중 감소.
   - structured tool-call repair 비중 소량 추가.
3. LR을 낮춘 짧은 continuation을 한다.
4. 공식 benchmark용으로는 TB2-lite가 아니라 Tool-Decathlon/MCP-Atlas/Terminal-Bench 2.1 harness 연결을 우선 조사한다.

읽을 파일:

```text
scripts/build_glm52_chaser_mix_20260624.py
scripts/run_glm52_chaser_mix_sft_20260624.sh
GLM52_CHASER_EXPERIMENT_20260624.ko.md
```

## P5. 공식 benchmark 연결

사용자가 요구한 것은 “공개적이고 공식적인 benchmark”다. 내부 TB2-lite는 빠른 회귀용으로만 쓴다.

우선 조사 대상:

- Tau2
- LiveCodeBench v6
- Tool-Decathlon
- MCP-Atlas Public Set
- Terminal-Bench 2.1

새 문서 후보:

```text
OFFICIAL_BENCHMARK_SETUP_20260625.ko.md
```

문서에 반드시 적을 것:

- benchmark 공식 링크
- local install 방법
- 입력 포맷
- 모델 서빙 방식
- vLLM 사용 가능 여부
- DiffusionGemma dLLM과 일반 autoregressive 모델의 실행 차이
- 예상 GPU 시간
- 점수 산출 방식

## 진행할 때마다 업데이트할 문서

작업 결과가 나오면 아래 파일 중 맞는 곳에 바로 반영한다.

- `NEXT_ACTIONS_CLAUDE_CODE_20260625.ko.md`
  - 현재 문서. 다음 담당자가 바뀌면 제일 먼저 갱신.
- `EXPERIMENT_HANDOFF_20260624.ko.md`
  - 실험 중단/재개 상태, checkpoint 상태 변경 시 갱신.
- `CURRENT_EXPERIMENT_STATUS_20260624.ko.md`
  - 큰 상태 변화가 있을 때 갱신.
- `DIFFUSIONGEMMA_STRENGTH_TASKS_20260624.ko.md`
  - DiffusionGemma 학습 방법론, VRAM sweep, adapter 평가 결과 반영.
- `README.md`, `README.ko.md`
  - 새 핵심 문서 링크와 주요 상태 1줄 반영.

커밋 기준:

- 코드 수정과 관련 문서 수정은 같은 커밋에 묶는다.
- 대용량 데이터/로그는 커밋하지 않는다.
- 실험을 시작하기 전 config/runner는 먼저 커밋한다.
- 실험 결과 문서도 별도 커밋한다.

## 다음 담당자에게 추천하는 실제 순서

GPU를 바로 쓰지 않는다면:

1. `scripts/run_fabliq_lm_eval_wave2_20260624.sh` 처리 방침 결정.
2. Qwen3.5 merged checkpoint config/key mismatch 원인 조사.
3. DiffusionGemma step 112 adapter 로드 가능성 조사.
4. `OFFICIAL_BENCHMARK_SETUP_20260625.ko.md` 작성.
5. README 링크 반영 후 커밋.

GPU를 다시 쓰기로 했다면:

1. `bash scripts/stop_fable_gpu_jobs_20260624.sh` dry-run.
2. `nvidia-smi`로 GPU idle 확인.
3. DiffusionGemma step 112 adapter 10-50 sample structured repair probe 먼저 실행.
4. 그 결과가 좋으면 VRAM sweep 20-50 step.
5. 최적 config로 1200 step 이상 run.
6. step 200 checkpoint 통과 여부를 반드시 확인.
7. run이 시작되면 30분마다 문서에 진행 상태를 남긴다.

## 마지막으로 확인된 안전 상태

- DiffusionGemma 학습 프로세스는 중단됨.
- `nvidia-smi` compute app 없음.
- GPU 0-7 메모리 1MiB 수준.
- stop script dry-run 결과: `No fable_distillation GPU/eval jobs matched.`
- 마지막 커밋 전 기준 untracked는 `scripts/run_fabliq_lm_eval_wave2_20260624.sh` 하나뿐이었다.
