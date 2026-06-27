# GLM-5.2 Fable QLoRA Runbook (2026-06-27)

작성 시각: 2026-06-27 04:59 UTC / 2026-06-27 13:59 KST

## 현재 결론

GLM-5.2-FP8은 8xH200 vLLM serving/eval 용도로 성공했다. 하지만 FP8 checkpoint를 그대로 LoRA 학습에 쓰는 경로는 실패했다. 실패 원인은 `w8a8_block_dynamic_fp8_matmul` fine-grained FP8 kernel에 backward/autograd formula가 없기 때문이다. 따라서 학습은 `zai-org/GLM-5.2` BF16 원본을 다운로드한 뒤, 로드 시 4bit BitsAndBytes QLoRA로 양자화해서 진행한다.

2026-06-27 13:59 KST 기준 BF16 원본 다운로드는 완료됐다. 첫 1-step QLoRA smoke는 venv 내부 `bitsandbytes` 누락을 고친 뒤 다시 돌렸고, 다음 단계에서 `device_map=auto`가 일부 module을 CPU/disk로 dispatch하려고 해서 BitsAndBytes guard로 중단됐다. runner 기본값은 GPU-only placement를 유도하도록 `GPU_MAX_MEMORY_GIB=140`, `CPU_MAX_MEMORY_GIB=0`으로 바꿨다.

## 공식 목표 Benchmark

GLM-5.2 공식 카드 기준으로 우리가 노릴 공식 benchmark는 다음이다.

| 우선순위 | Benchmark | GLM-5.2 공식 점수 | 목표 |
| --- | --- | ---: | ---: |
| 1 | MCP-Atlas (Public Set) | 76.8 | 78.0 이상 |
| 2 | Tool-Decathlon | 48.2 | 52.8 이상, stretch 60.0 |
| 3 | Terminal-Bench 2.1 (Terminus-2) | 81.0 | 82.0 이상 |
| 4 | Terminal-Bench 2.1 (Best Reported Harness) | 82.7 | 83.0 이상 |

이 목표를 택한 이유는 우리가 가진 Fable/Mythos 데이터가 tool routing, MCP first-tool selection, terminal JSON action 안정성, multi-step repair에 직접 닿아 있기 때문이다. GPQA/AIME 같은 순수 reasoning은 이번 데이터로 단기간에 이길 확률이 낮다.

공식 링크:

- GLM-5.2-FP8 모델 카드/benchmark: https://huggingface.co/zai-org/GLM-5.2-FP8
- GLM-5.2 benchmark 섹션: https://huggingface.co/zai-org/GLM-5.2#benchmark
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- Transformers fine-grained FP8 docs: https://huggingface.co/docs/transformers/en/quantization/finegrained_fp8
- FP8 LoRA 학습 guard 관련 HF 이슈: https://github.com/huggingface/transformers/issues/46736

## 데이터

현재 학습 데이터:

```text
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets/official_agentic_sft_mix_20260627.jsonl
/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets/official_agentic_sft_mix_20260627.meta.json
```

요약:

| 항목 | 값 |
| --- | ---: |
| total rows | 19,536 |
| rows with tools field | 8,694 |
| rows with final tool call | 7,794 |
| tau2 telecom style | 2,700 |
| MCP-Atlas-like | 7,200 |
| Tool-Decathlon-like | 6,000 |
| Fable-style agentic coding | 3,500 |
| reasoning style | 136 |

중요: 공식 benchmark task 자체를 train에 넣으면 안 된다. 위 데이터는 official-style synthetic/converted mix이고, official public set은 평가용으로만 둔다.

## 모델/디스크 상태

이미 받은 모델:

```text
/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
```

FP8 cache size는 약 `707G`다.

BF16 원본 `zai-org/GLM-5.2` 다운로드 완료 snapshot:

```text
/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2/snapshots/f2263102df303b2faa54a6861a29d1770ce846c0
```

snapshot checker 결과:

```text
ready snapshot=/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2/snapshots/f2263102df303b2faa54a6861a29d1770ce846c0 shards=282
```

2026-06-27 13:59 KST 기준 `/home/work/.data` 여유는 약 `276G`다. 그래서 학습 output은 full model 저장 없이 LoRA adapter checkpoint만 남긴다.

## 현재 상태와 다음 실행

다운로드는 완료됐고, 현재 GLM 학습 장기 세션은 없다. 상태 확인:

```bash
tmux ls
df -h /home/work/.data
```

BF16 snapshot 확인:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
.venvs/glm52-vllm-cu129-release-driver570/bin/python \
  scripts/check_glm52_bf16_snapshot_ready_20260627.py \
  --model-id zai-org/GLM-5.2 \
  --cache-dir /home/work/.data/huggingface/hub
```

자동 학습 watcher는 다운로드 완료 후 한번 실행됐지만, smoke 실패와 함께 종료됐다. 지금은 아래 1-step smoke를 직접 재실행하는 것이 맞다.

```text
scripts/check_glm52_bf16_snapshot_ready_20260627.py
scripts/watch_glm52_bf16_ready_then_qlora_20260627.sh
```

권장 실행:

```bash
tmux new-session -d -s fable_glm52_bf16_ready_then_qlora \
  "cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation && bash scripts/watch_glm52_bf16_ready_then_qlora_20260627.sh"
```

이 watcher는 `model.safetensors.index.json`에 나온 모든 shard가 local cache에 있는지 확인한다. 완료되면 1-step smoke, 25-step pilot, long resumable QLoRA 순서로 이어간다. `START_LONG_AFTER_PILOT=0`을 주면 pilot까지만 하고 멈춘다.

다운로드 재시작이 필요하면:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
RUN_NOW=1 bash scripts/download_glm52_bf16_20260627.sh
```

## 학습 코드

공통 trainer:

```text
training/train_glm52_fp8_device_map_lora.py
```

이 trainer는 이제 두 경로를 지원한다.

- `zai-org/GLM-5.2-FP8`: vLLM serving/eval 확인용. 학습은 blocked.
- `zai-org/GLM-5.2` + `--load-in-4bit`: BF16 원본을 4bit QLoRA로 로드해 LoRA adapter 학습.

BF16 QLoRA runner:

```text
scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

기본 설정:

| 설정 | 값 |
| --- | --- |
| env | `.venvs/glm52-vllm-cu129-release-driver570` |
| model | `zai-org/GLM-5.2` |
| quantization | BitsAndBytes 4bit NF4 |
| compute dtype | bfloat16 |
| LoRA target | `q_proj,k_proj,v_proj,o_proj` |
| LoRA rank/alpha | `64 / 128` |
| max seq | `2048` |
| GPU max memory | `140GiB` per visible GPU |
| CPU max memory | `0GiB` by default, to avoid BnB CPU/disk dispatch |
| save steps | `25` |
| output | `/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-20260627` |

## 학습 실행 순서

다음 1-step smoke:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
RUN_NOW=1 \
RUN_ID=20260627_glm52_bf16_qlora_smoke \
OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-smoke-20260627 \
MAX_STEPS=1 MAX_TRAIN_ROWS=2 MAX_SEQ_LENGTH=512 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

smoke 성공 조건:

```text
trainable_params=...
step=1 loss=...
saved checkpoint: ...
training complete final_lora=...
```

이미 해결한 smoke failure:

```text
ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes
```

조치: `.venvs/glm52-vllm-cu129-release-driver570` 안에 `bitsandbytes==0.49.2`를 `--no-deps`로 설치했다. `torch 2.11.0+cu129`, `vllm 0.23.0`, `transformers 5.12.1`는 유지됐다.

재현 규칙: `scripts/setup_glm52_vllm_uv_20260627.sh`가 `bitsandbytes`를 `--no-deps --force-reinstall`로 설치한다. `--ignore-installed` 또는 일반 dependency install로 실행하면 `torch`가 CUDA 13 wheel로 바뀔 수 있으므로 피한다.

현재 해결 중인 smoke failure:

```text
ValueError: Some modules are dispatched on the CPU or the disk.
```

조치: `scripts/run_glm52_bf16_qlora_device_map_20260627.sh` 기본값을 `GPU_MAX_MEMORY_GIB=140`, `CPU_MAX_MEMORY_GIB=0`으로 바꿨다. 다음 smoke에서 GPU-only load가 되는지 확인한다.

smoke 성공 후 pilot:

```bash
RUN_NOW=1 \
RUN_ID=20260627_glm52_bf16_qlora_pilot25 \
OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-pilot25-20260627 \
MAX_STEPS=25 MAX_SEQ_LENGTH=1024 SAVE_STEPS=5 SAVE_TOTAL_LIMIT=2 \
bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

pilot에서 step time과 VRAM을 본 뒤 긴 학습:

```bash
tmux new-session -d -s fable_glm52_bf16_qlora_long \
  "cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation && RUN_NOW=1 bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh"
```

## 중단/재개

중단:

```bash
tmux kill-session -t fable_glm52_bf16_qlora_long 2>/dev/null || true
tmux kill-session -t fable_glm52_bf16_ready_then_qlora 2>/dev/null || true
pgrep -af 'train_glm52_fp8_device_map_lora.py'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
```

필요한 PID만:

```bash
kill -TERM <pid>
sleep 10
kill -KILL <pid>
```

재개:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
RUN_NOW=1 RESUME_FROM_CHECKPOINT=auto bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

trainer는 `checkpoint-*` 안의 `manual_trainer_state.json`을 읽고 `global_step`부터 이어간다.

## 평가 순서

1. vLLM으로 FP8 base baseline을 같은 prompt/harness에서 확보한다.
2. QLoRA adapter를 merge하거나 adapter serving이 가능한 경로를 만든다.
3. 우선 MCP-Atlas public subset 스타일의 first-tool/action accuracy를 로컬 proxy로 빠르게 본다.
4. 공식 접근이 되는 순서대로 MCP-Atlas public set, Tool-Decathlon official evaluation service, Terminal-Bench 2.1 Terminus-2를 실행한다.
5. 목표 미달이면 실패 case를 다시 Fable 데이터로 변환해 2차 SFT 또는 preference/RL-style correction 데이터로 만든다.

## 현재 막힌 경로

FP8 direct LoRA:

```text
RuntimeError: Trying to backward through _finegrained_fp8_cuda_...w8a8_block_dynamic_fp8_matmul.default but no autograd formula was registered.
```

이 경로는 현재 보류한다. FP8은 계속 vLLM serving/eval용으로 쓴다.

HF Trainer + DeepSpeed ZeRO-3:

```text
FP8 model on CPU warning
rank별 CPU RSS 약 150-179GB 이상 증가
GPU VRAM 약 1GiB 수준
```

이 경로도 full run 금지다. rank별 CPU 복제 로딩으로 OOM 위험이 높다.
