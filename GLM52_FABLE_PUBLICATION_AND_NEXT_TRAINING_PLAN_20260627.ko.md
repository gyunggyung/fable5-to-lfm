# GLM-5.2 Fable 출판/다음 학습 계획 (2026-06-27)

## 결론

아직 완료된 GLM-5.2 Fable LoRA adapter checkpoint는 없다.

대신 다음 상태까지 준비되어 있다.

- Fable/Mythos 스타일 official-agentic SFT mix 준비 완료.
- GLM-5.2-FP8 vLLM 8xH200 serving/probe 성공.
- Axolotl 8-bit/4-bit MoE LoRA 실패 지점 문서화 완료.
- Megatron-SWIFT 환경, 데이터 변환, GLM 템플릿 설정, TP/PP/EP model-parallel runner 준비 완료.
- Hugging Face adapter repo 이름, 모델 카드, checkpoint watcher/upload script 준비 완료.

공개 adapter target:

```text
https://huggingface.co/LLM-OS-Models/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA
```

로컬 모델 카드 원본:

```text
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/model_cards/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-README.md
```

## 이름/포지셔닝

공개 이름은 다음으로 고정한다.

```text
GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA
```

의도:

- `GLM-5.2-FP8`: local serving/loading 기준 checkpoint를 명확히 표시.
- `Agentic-Fable5`: Fable/Mythos식 terminal/tool-call behavior를 목표로 표시.
- `Composer2.5`: 사용자가 준 인기 Fable 계열 naming과 맞춘 style/curriculum signal.
- `LoRA`: base model을 포함하지 않는 adapter-only repo임을 표시.

혼동 방지:

- `TP8` 같은 물리 실행 방식은 공개 이름에서 뺐다. 실제 최신 runner는 `TP=4, PP=2, EP=4`도 시도했고, 다음 retry는 offload/CPU-init 여부에 따라 바뀔 수 있다.
- repo에는 GLM-5.2 base shard를 올리지 않는다. adapter/config/card/log metadata만 올린다.

## 현재 데이터

원본 mix:

```text
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets/official_agentic_sft_mix_20260627.jsonl
```

Swift agent-format 변환본:

```text
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets/official_agentic_sft_mix_20260627.swift_agent.jsonl
```

현재 수량:

| 항목 | 값 |
| --- | ---: |
| source rows | 19,536 |
| Swift agent rows kept | 14,374 |
| skipped rows | 5,162 |

목표 behavior:

- terminal command planning
- shell/edit/read/write 흐름의 다음 행동 선택
- structured tool-call formatting
- malformed JSON/tool-call repair
- MCP/Tool-Decathlon 스타일 tool routing
- Fable/Mythos 계열의 짧고 실용적인 에이전트 응답

## 현재 모델/디스크

GLM-5.2 BF16 snapshot:

```text
/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2/snapshots/f2263102df303b2faa54a6861a29d1770ce846c0
```

GLM-5.2-FP8 snapshot:

```text
/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
```

학습 output root:

```text
/home/work/.data/harness1/models
```

현재 GLM Fable output 기본값:

```text
/home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-20260627
```

2026-06-27 최신 확인 기준 `/home/work/.data` 여유:

```text
674G available, 98% used
```

## 지금까지 실패 지점

### 1. FP8 direct LoRA

`training/train_glm52_fp8_device_map_lora.py` 경로에서 GLM-5.2-FP8 로딩과 LoRA attach까지는 성공했다.

실패:

```text
w8a8_block_dynamic_fp8_matmul backward/autograd formula 없음
```

판단:

- 현재 Transformers fine-grained FP8 inference kernel은 이 checkpoint를 그대로 학습시키는 backward를 제공하지 않는다.
- GLM-5.2-FP8은 serving/eval/teacher로는 유효하지만, 이 경로로 adapter 학습은 막혔다.

### 2. BF16 + BitsAndBytes QLoRA

GLM-5.2 BF16 snapshot 로드 시도 결과:

```text
total_bf16_gib=1403.19
raw_moe_expert_bf16_gib=1368.00
```

manual `device_map=glm_layers`로 guard는 넘었지만 `294/1344`, 약 22% weight loading에서 GPU OOM.

판단:

- GLM MoE expert weight 대부분이 일반 Linear가 아니라 raw Parameter라 local BnB QLoRA 방식으로 충분히 줄지 않는다.

### 3. Axolotl 8-bit/4-bit MoE LoRA

8-bit:

```text
logs/20260627_glm52_axolotl_8bit_moe_lora_chunk_patch2/train.log
```

결과:

```text
278/1344 weights, 21%, GPU OOM at about 140GiB/GPU
```

4-bit:

```text
logs/20260627_glm52_axolotl_4bit_moe_qlora_fallback1/train.log
logs/20260627_glm52_axolotl_4bit_moe_qlora_flatten_patch2/train.log
```

결과:

```text
40/1344, bitsandbytes /src/csrc/ops.cu invalid configuration argument
```

### 4. Megatron-SWIFT BF16/FP8 snapshot LoRA

환경:

```text
/home/work/.cache/fable_distillation/venvs/glm52-swift-megatron
```

핵심 스크립트:

```text
scripts/setup_glm52_swift_megatron_env_20260627.sh
scripts/prepare_swift_glm52_agent_jsonl_20260627.py
scripts/run_glm52_swift_megatron_tp8_lora_20260627.sh
```

성공한 부분:

- `transformer-engine[core_cu12,pytorch]==2.16.1` import 문제 해결.
- local HF snapshot 사용으로 ModelScope 중복 다운로드 방지.
- `template=glm5_2`, `agent_template=glm5_1`로 데이터 전처리 통과.
- `TP=4, PP=2, EP=4` model-parallel launch까지 진행.

실패:

```text
torch.OutOfMemoryError during model construction
GPU당 약 139.75-139.80GiB 사용, train step 전 실패
```

FP8 snapshot을 써도 MCore가 BF16 trainable parameter를 생성해서 같은 OOM이 난다.

`TORCH_DTYPE=float8_e4m3fn` 강제 시도는 Swift argument validation에서 거부됐다.

## 다음 학습 우선순위

### 우선순위 1: GLM-5.2 Megatron-SWIFT offload smoke

목표:

- 무조건 성능 run이 아니라, `train step >= 1`과 adapter checkpoint 생성을 먼저 확인.
- 성공하면 checkpoint watcher가 HF adapter repo에 계속 업로드한다.

실행:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation

tmux new-session -d -s fable_glm52_swift_offload_smoke_20260627 \
  "RUN_NOW=1 \
   RUN_ID=20260627_glm52_fp8_swift_offload_smoke \
   MAX_STEPS=20 SAVE_STEPS=5 MAX_LENGTH=512 \
   LORA_RANK=4 LORA_ALPHA=8 \
   USE_CPU_INITIALIZATION=true \
   OFFLOAD_MODEL=true \
   OFFLOAD_BRIDGE=true \
   OPTIMIZER_CPU_OFFLOAD=true \
   MOE_GROUPED_GEMM=false \
   bash scripts/run_glm52_swift_megatron_tp8_lora_20260627.sh"
```

로그:

```text
logs/20260627_glm52_fp8_swift_offload_smoke/train.log
```

주의:

- 이 smoke는 느릴 수 있다.
- 그래도 성공하면 최초 GLM-5.2 Fable adapter checkpoint가 생긴다.
- 실패하면 “8xH200 단일 노드 + 현재 Swift/MCore stack에서 GLM-5.2 adapter 학습 불가” 판단을 더 강하게 문서화한다.

중단:

```bash
tmux kill-session -t fable_glm52_swift_offload_smoke_20260627
```

재개:

- checkpoint가 생겼다면 같은 `OUTPUT_DIR`로 다시 실행한다.
- checkpoint가 없으면 같은 명령 재실행은 처음부터 시작한다.

2026-06-27 09:55 UTC / 18:55 KST 실행 상태:

```text
tmux: fable_glm52_swift_offload_smoke_20260627
run_id: 20260627_glm52_fp8_swift_offload_smoke_retry7
log: logs/20260627_glm52_fp8_swift_offload_smoke_retry7/train.log
output: /home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-20260627/v0-20260627-184743
```

현재 확인:

```text
status: running
GPU memory: about 1.5GiB per GPU
CPU RSS: about 58-61GiB per rank, 8 ranks
dataset: train_dataset printed
train step: not reached yet
adapter checkpoint: none yet
```

해석:

- 이전처럼 argument/template 단계에서 바로 실패하지 않았다.
- `USE_CPU_INITIALIZATION=true`, `OFFLOAD_MODEL=true`, `OFFLOAD_BRIDGE=true`, `OPTIMIZER_CPU_OFFLOAD=true` 때문에 GPU VRAM은 아직 낮다.
- 이 run은 성능 run이 아니라 GLM-5.2에서 최초 `train step >= 1`과 adapter save 가능 여부를 확인하는 smoke다.

### 우선순위 2: GLM-5.2는 teacher, trainable base에 Fable adapter 학습

GLM 자체 adapter가 계속 막히면 GPU를 비우지 말고 다음 trainable base로 Fable-style adapter를 만든다.

권장 순서:

1. `Qwen/Qwen3.5-9B`
2. `google/gemma-4-12B-it`
3. `MiniMaxAI/MiniMax-M3`

이 경우 모델 이름은 다음 규칙을 쓴다.

```text
<Base>-Agentic-Fable5-Composer2.5-LoRA
<Base>-Agentic-Fable5-Composer2.5-GGUF
```

GLM-5.2-FP8은 vLLM teacher/evaluator로 사용한다.

## 업로드 운영

카드만 먼저 올리기:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/work/.cache/fable_distillation/venvs/glm52-swift-megatron/bin/python \
  scripts/upload_glm52_fable_lora_adapter_20260627.py \
  --allow-empty \
  --repo-id LLM-OS-Models/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA \
  --commit-message "Initialize GLM-5.2 Fable LoRA model card"
```

2026-06-27 09:55 UTC 확인:

```text
result: failed
error: 401 Unauthorized at https://huggingface.co/api/repos/create
action needed: set HF_TOKEN or HUGGINGFACE_HUB_TOKEN with write access to LLM-OS-Models
```

checkpoint 자동 업로드 watcher:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
tmux new-session -d -s fable_glm52_lora_upload_watch_20260627 \
  "UPLOAD_CARD_ON_START=1 bash scripts/watch_upload_glm52_fable_lora_20260627.sh"
```

watcher는 다음 패턴 아래에서 adapter checkpoint를 찾는다.

```text
/home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2*Fable5*LoRA*/**/adapter_model.safetensors
/home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2*Fable5*LoRA*/**/adapter_model.bin
/home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2*Fable5*LoRA*/**/adapter_config.json
```

상태 파일:

```text
logs/20260627_glm52_fable_upload_seen.txt
```

## 공식 benchmark 목표

공식/public 계열 목표는 다음으로 유지한다.

| Benchmark | GLM-5.2 official 참고값 | 목표 |
| --- | ---: | ---: |
| MCP-Atlas public subset | 76.8 | 78.0 이상 |
| Tool-Decathlon | 48.2 | 52.8 이상 |
| Terminal-Bench 2.1 / Terminus-2 | 81.0 | 82.0 이상 |
| Terminal-Bench 2.1 best reported harness | 82.7 | 83.0 이상 |

로컬 TB2-lite는 빠른 regression check일 뿐이다. 공개 모델 카드에 성능을 쓸 때는 official/public benchmark 결과와 평가 조건을 분리해서 적는다.

## Claude Code 인계 체크리스트

1. `git status --short`로 미커밋 변경 확인.
2. `df -h /home/work/.data`로 남은 공간 확인.
3. `nvidia-smi`로 GPU가 비어 있는지 확인.
4. upload watcher를 먼저 켜거나 README-only 업로드.
5. GLM-5.2 offload smoke를 20 step 이하로 실행.
6. adapter checkpoint가 생기면 HF repo에 올라갔는지 확인.
7. smoke가 또 OOM이면 GLM 자체 학습은 중지하고, GLM-5.2-FP8 vLLM teacher + trainable base adapter 실험으로 전환.
8. 결과가 생길 때마다 이 문서와 README를 갱신하고 커밋.

## 관련 문서

- `GLM52_FP8_FABLE_TUNING_STATUS_20260627.ko.md`
- `GLM52_FABLE_QLORA_RUNBOOK_20260627.ko.md`
- `DATA_CLEANUP_GIT_AND_GLM_TRAINING_OPTIONS_20260627.ko.md`
- `model_cards/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-README.md`
