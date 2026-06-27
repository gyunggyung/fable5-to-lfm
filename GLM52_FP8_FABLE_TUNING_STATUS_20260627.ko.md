# GLM-5.2-FP8 Fable 스타일 튜닝/평가 상태 (2026-06-27)

작성 시각: 2026-06-27 04:59 UTC / 2026-06-27 13:59 KST

## 한줄 결론

GLM-5.2-FP8 다운로드와 Fable/Mythos 스타일 데이터 준비는 끝났다. vLLM `0.23.0+cu129` 서버도 8xH200에서 정상 기동했고 OpenAI-compatible probe 3개가 통과했다. 그러나 FP8 checkpoint 직접 LoRA 학습은 fine-grained FP8 matmul backward 미지원으로 실패했다. 실제 학습 경로는 `zai-org/GLM-5.2` BF16 원본을 4bit BitsAndBytes QLoRA로 로드하는 방식으로 전환했다.

## 최신 업데이트 (2026-06-27 13:59 KST)

- GLM-5.2-FP8 vLLM server 성공:
  - env: `.venvs/glm52-vllm-cu129-release-driver570`
  - vLLM: `0.23.0+cu129`
  - torch: `2.11.0+cu129`
  - transformers: `5.12.1`
  - launch: `MAX_MODEL_LEN=131072 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=16384 RUN_NOW=1 bash scripts/run_glm52_fp8_vllm_server_20260627.sh`
  - model load: worker당 `91.82 GiB`, 약 `1527s`
  - KV cache: `465,088 tokens`, 131,072-token request 기준 concurrency `3.55x`
  - CUDA graph capture: `97s`, 약 `1.33 GiB`
  - probe: 3개 prompt 모두 성공, 응답 시간 약 `6.79-7.54s`
- GLM HF Trainer + ZeRO-3 path는 계속 보류:
  - `FP8 model on CPU` 경고와 rank별 CPU RSS 증가가 반복된다.
  - full run 시작 금지. 문서화/비교용으로만 남긴다.
- 새 GLM device-map LoRA/QLoRA path 추가:
  - trainer: `training/train_glm52_fp8_device_map_lora.py`
  - runner: `scripts/run_glm52_fp8_device_map_lora_20260627.sh`
  - BF16 QLoRA runner: `scripts/run_glm52_bf16_qlora_device_map_20260627.sh`
  - BF16 download runner: `scripts/download_glm52_bf16_20260627.sh`
  - BF16 ready watcher: `scripts/watch_glm52_bf16_ready_then_qlora_20260627.sh`
  - 기본 env: `.venvs/glm52-vllm-cu129-release-driver570`
  - 핵심 설정: `HF_DEACTIVATE_ASYNC_LOAD=1`, main-thread CUDA warmup, `device_map=auto`, `gpu_max_memory_gib=140`, `cpu_max_memory_gib=0`
  - 추가 의존성: `peft 0.19.1`, `accelerate 1.14.0`, `datasets 5.0.0`, `kernels 0.12.3`
  - 주의: transformers 5.12.1은 `kernels>=0.12,<0.13`을 요구한다. `kernels 0.16.0`은 import 단계에서 깨진다.
  - `bitsandbytes`는 `scripts/setup_glm52_vllm_uv_20260627.sh`에서 `--no-deps --force-reinstall bitsandbytes==0.49.2`로 설치한다. 일반 `pip install --ignore-installed bitsandbytes`는 `torch`까지 CUDA 13 wheel로 바꾸려 하므로 쓰지 않는다.
- FP8 direct LoRA smoke 결과:
  - CUDA init, tokenizer, 8GPU device-map 로딩, LoRA attach까지는 성공.
  - 1-step backward에서 실패: `w8a8_block_dynamic_fp8_matmul` autograd formula 없음.
  - 판단: FP8은 vLLM serving/eval 전용으로 유지하고 학습에는 쓰지 않는다.
- BF16 GLM-5.2 다운로드 완료:
  - snapshot: `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2/snapshots/f2263102df303b2faa54a6861a29d1770ce846c0`
  - checker: `ready ... shards=282`
  - 2026-06-27 13:59 KST 기준 `/home/work/.data` 여유는 약 `276G`
- BF16 QLoRA smoke 상태:
  - venv 내부 `bitsandbytes==0.49.2` 설치 완료. `torch 2.11.0+cu129`, `vllm 0.23.0`, `transformers 5.12.1`는 유지됨.
  - 첫 재시도 실패 원인: `device_map=auto`가 일부 module을 CPU/disk로 보내려고 하면서 BitsAndBytes guard 발생.
  - `GPU_MAX_MEMORY_GIB=140`, `CPU_MAX_MEMORY_GIB=0`만으로는 auto estimator가 계속 CPU/disk dispatch를 선택했다.
  - 조치: `scripts/run_glm52_bf16_qlora_device_map_20260627.sh` 기본값을 `DEVICE_MAP=glm_layers`로 바꾸고, trainer에 GLM 78-layer manual device map을 추가했다.
  - manual map 재시도 결과: guard는 넘겼지만 `Loading weights` 22%, `294/1344`, 약 8분 52초 지점에서 GPU1 OOM.
  - metadata inspector 결과: BF16 총량 `1403.19GiB`, raw MoE expert weight `1368.00GiB`, tensors `58,368`.
  - 판단: 로컬 Transformers+BitsAndBytes GLM-5.2 BF16 QLoRA는 현재 환경에서 보류한다. GLM은 FP8 vLLM eval/teacher로 쓰고, Fable-style 튜닝은 trainable base로 돌린다.
  - safety guard: `scripts/run_glm52_bf16_qlora_device_map_20260627.sh`와 watcher는 기본 실행을 막는다. 재현 목적이면 `ALLOW_EXPERIMENTAL_GLM52_BF16_QLORA=1`을 명시한다.
- Axolotl 8-bit MoE LoRA 경로 추가:
  - env: `/home/work/.cache/fable_distillation/venvs/glm52-axolotl-8bit-moe`
  - config: `configs/axolotl_glm52_8bit_moe_lora_20260627.yml`
  - key settings: `load_in_8bit: true`, `quantize_moe_experts: true`, `fsdp_version: 2`, `optimizer: adamw_torch_8bit`, `sequence_len: 2048`, `max_steps: 200`
  - data: `datasets/official_agentic_sft_mix_20260627.axolotl_chatml.jsonl`, 19,536 rows converted to completion-style ChatML text
  - patch: `scripts/patch_axolotl_moe_8bit_flatten_20260627.py` chunks GLM fused 3D expert tensors before `bitsandbytes` int8 quantization.
  - active tmux: `fable_glm52_axolotl_8bit_chunk_patch2_20260627`
  - active log: `logs/20260627_glm52_axolotl_8bit_moe_lora_chunk_patch2/train.log`
  - output: `/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-Axolotl-8bit-MoE-LoRA-20260627`
  - status: previous unchunked CUDA illegal memory access was bypassed; active run is still in weight loading and has reached roughly 14%+ with about 96-98GiB VRAM per GPU.

## 현재 상태

- 작업 루트: `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation`
- HF cache: `/home/work/.data/huggingface/hub`
- GLM snapshot: `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1`
- GLM cache size: 약 `707G`
- BF16 GLM-5.2 snapshot: safetensors `282` shards, snapshot `f2263102df303b2faa54a6861a29d1770ce846c0`
- GPU 상태: Axolotl GLM-5.2 8-bit MoE LoRA active run이 8xH200에서 weight loading/on-the-fly int8 quantization 중이다.
- Git 상태: 로컬 `main`은 `origin/main`보다 여러 커밋 앞섬. 최신 GLM Axolotl chunk patch 커밋은 `8b9ab73`.
- Push 상태: HTTPS GitHub credential 없음으로 실패. 에러는 `fatal: could not read Username for 'https://github.com': No such device or address`.

## 이미 준비한 데이터

데이터 빌더:

```text
scripts/build_official_agentic_sft_mix_20260627.py
```

출력:

```text
datasets/official_agentic_sft_mix_20260627.jsonl
datasets/official_agentic_sft_mix_20260627.meta.json
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

중요: 이 데이터는 공식 benchmark task 파일을 읽지 않는다. 공식 public set은 held-out 평가용으로만 써야 한다.

## 이미 준비한 코드/스크립트

- `configs/deepspeed_zero3_glm52_lora_20260627.json`
- `scripts/download_glm52_fp8_loop_20260627.sh`
- `scripts/setup_glm52_lora_env_20260627.sh`
- `scripts/run_glm52_official_agentic_lora_20260627.sh`
- `scripts/watch_glm52_fp8_ready_then_train_20260627.sh`
- `scripts/upload_adapter_to_hf_20260627.sh`
- `scripts/setup_glm52_vllm_uv_20260627.sh`
- `scripts/run_glm52_fp8_vllm_server_20260627.sh`
- `scripts/probe_glm52_vllm_server_20260627.py`
- `scripts/run_glm52_fp8_device_map_lora_20260627.sh`
- `scripts/download_glm52_bf16_20260627.sh`
- `scripts/run_glm52_bf16_qlora_device_map_20260627.sh`
- `scripts/check_glm52_bf16_snapshot_ready_20260627.py`
- `scripts/watch_glm52_bf16_ready_then_qlora_20260627.sh`
- `scripts/inspect_glm52_bf16_qlora_memory_20260627.py`
- `training/train_multifamily_chat_sft.py`
- `training/train_glm52_fp8_device_map_lora.py`

`training/train_multifamily_chat_sft.py`에는 다음 GLM 호환성 수정이 들어갔다.

- GLM native chat template가 요구하는 tool call `arguments` dict 변환
- `GlmMoeDsaConfig.n_routed_experts`를 `num_experts`/`num_local_experts` alias로 보강
- `HfDeepSpeedConfig`를 model load 전에 초기화
- `TrainingArguments`를 model load 전에 생성
- LoRA 입력 gradient 활성화

## 실행한 smoke와 판단

### 1차 실패

GLM tokenizer/chat template가 `tc.arguments.items()`를 기대했는데, 데이터의 OpenAI-style `tool_calls[].function.arguments`가 JSON string이라 실패했다.

조치:

- GLM template가 mapping arguments를 요구할 때만 JSON string을 dict로 변환하도록 수정.
- 16-row tokenizer smoke 확인 완료.

### 2차 실패

FP8 quantizer가 `config.num_experts`를 기대했는데 GLM config에는 `n_routed_experts=256`만 있어 실패했다.

조치:

- GLM config compat patch 추가.

### 3차 smoke 중단

DeepSpeed 설정과 `HfDeepSpeedConfig`를 넣고 다시 시작했지만, 다음 패턴이 나왔다.

- weight loading은 진행: 최종 확인 시 약 `15%`, `333/2160`
- GPU VRAM: GPU당 약 `1038 MiB`, utilization `0%`
- CPU RSS: rank별 약 `150-179GB`
- 시스템 RAM은 회수 가능했지만, 진행률 대비 RSS 증가가 커서 full load 시 OOM 위험이 높음

판단:

이 경로는 8xH200 tensor/ZeRO shard 학습이 아니라 rank별 CPU 복제 로딩에 가깝다. smoke 목적은 학습 가능 경로 확인이므로 OOM 전에 중단했다.

### 4차 FP8 device-map LoRA 실패

단일 프로세스 `device_map=auto` 방식으로 FP8 base를 8GPU에 직접 분산 로딩하는 경로를 추가했다. `HF_DEACTIVATE_ASYNC_LOAD=1`, main-thread CUDA warmup, `kernels==0.12.3`, `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`을 적용했다.

결과:

- tokenizer/cache 생성 성공
- GLM-5.2-FP8 weight `2160/2160` 로딩 성공
- LoRA attach 성공
- trainable params: `112,459,776`
- forward 후 backward에서 실패

실패 원문:

```text
RuntimeError: Trying to backward through _finegrained_fp8_cuda_...w8a8_block_dynamic_fp8_matmul.default but no autograd formula was registered.
```

판단:

현재 Transformers fine-grained FP8 inference kernel은 이 GLM FP8 checkpoint를 학습시키는 backward를 제공하지 않는다. 이 경로는 더 밀지 않는다. FP8은 vLLM serving/eval 전용으로 유지한다.

## 공식 근거

- GLM-5.2-FP8 모델 카드: https://huggingface.co/zai-org/GLM-5.2-FP8
- GLM-5.2 benchmark 섹션: https://huggingface.co/zai-org/GLM-5.2#benchmark
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- GLM-5.2 blog: https://huggingface.co/blog/zai-org/glm-52-blog
- Fireworks GLM-5.2-FP8 hosted fine-tuning page: https://fireworks.ai/models/fireworks/glm-5p2-fp8

공식/recipe에서 확인한 기준:

- GLM-5.2는 약 743B MoE, active 약 39B급 모델이다.
- GLM-5.2-FP8은 8xH200/8xH20 단일 노드 serving을 practical default로 둔다.
- recipe 기준은 vLLM `0.23.0`, transformers `>=5.9.0`.
- vLLM launch 핵심 옵션은 `--tensor-parallel-size 8`, `--kv-cache-dtype fp8`, `--tool-call-parser glm47`, `--reasoning-parser glm45`, `--enable-auto-tool-choice`.
- 공식 benchmark footnote 기준:
  - Tool-Decathlon은 official evaluation service, `max_token=128K`.
  - MCP-Atlas는 500-task public subset, think mode, 10분 timeout/task.
  - Terminal-Bench 2.1은 Terminus-2 framework, parser=json, timeout=4h, max_new_tokens=48k, max_episodes=500, 256K context.

## 다음 우선순위

### A. GLM official vLLM serving smoke

현재 기존 `.vllm-lfm-cu12`는 vLLM `0.19.1`, transformers `5.5.4`라 GLM-5.2 recipe 요구치보다 낮다. GLM 전용 env를 별도로 만든다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
bash scripts/setup_glm52_vllm_uv_20260627.sh
```

2026-06-27 11:30 KST 기준 첫 GLM 전용 env 설치 확인:

```text
env: .venvs/glm52-vllm-cu128
torch: 2.11.0+cu129
transformers: 5.12.1
vllm: 0.23.0
```

이 env는 `vllm --version` import까지는 통과했지만, 실제 worker 초기화에서 `CUDA driver version is insufficient for CUDA runtime version`로 실패했다. 원인은 `uv --torch-backend=auto`가 CUDA 13 runtime wheel을 선택했기 때문이다. 로컬 드라이버는 `570.86.10`, CUDA `12.9` 노출이므로 GLM용 vLLM env는 `--torch-backend=cu128`로 다시 만들어야 한다.

두 번째로 `--torch-backend=cu128` env도 만들었지만, vLLM 0.23.0 release asset에는 `cu128` wheel이 없고 PyPI wheel은 계속 CUDA 13 `libcudart.so.13`에 링크된다. 따라서 로컬 드라이버 `570.86.10`/CUDA `12.9`에 맞춰 GitHub release의 `vllm-0.23.0+cu129` wheel을 직접 설치하는 경로로 바꿨다.

현재 스크립트 기본 env는 다음으로 바뀌었다.

```text
env: .venvs/glm52-vllm-cu129-release-driver570
vLLM wheel: https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl
torch backend: cu129
torch: 2.11.0+cu129
transformers: 5.12.1
vllm --version: 0.23.0+cu129
```

중요: 이 env는 반드시 `env -u PYTHONPATH PYTHONNOUSERSITE=1`로 실행해야 한다. 그렇지 않으면 `/home/work/.local/lib/python3.12/site-packages`가 먼저 잡혀 기존 `torch 2.12.0.dev20260407+cu128`, `transformers 5.5.4`가 섞일 수 있다. 또한 `scripts/run_glm52_fp8_vllm_server_20260627.sh`가 venv 내부 CUDA runtime library 경로를 찾아 `LD_LIBRARY_PATH`를 자동으로 구성한다.

`ldd` 확인에서 `vllm/_C.abi3.so`는 `libcudart.so.12`를 `nvidia/cuda_runtime/lib`에서 잡는다. 이 경로가 `nvidia/cu13/lib`보다 먼저 오도록 스크립트에 반영했다.

서버 dry-run:

```bash
bash scripts/run_glm52_fp8_vllm_server_20260627.sh
```

실행:

```bash
RUN_NOW=1 bash scripts/run_glm52_fp8_vllm_server_20260627.sh
```

별도 shell에서 probe:

```bash
python scripts/probe_glm52_vllm_server_20260627.py \
  --base-url http://127.0.0.1:8000 \
  --model glm-5.2-fp8
```

서버가 뜨면 먼저 `MAX_MODEL_LEN=131072` 또는 `262144`로 안정성을 본다. 8xH200에서 full 1M context를 바로 목표로 잡지 않는다.

### B. 공식 benchmark 방향

내부 TB2-lite만 보면 안 된다. 다음 순서가 맞다.

1. vLLM으로 GLM-5.2-FP8 base를 공식-adjacent prompt/probe에서 정상 serving.
2. Tool-Decathlon/MCP-Atlas/Terminal-Bench 2.1 중 접근 가능한 공식 public harness 확인.
3. 우리 Fable/Mythos 스타일 데이터가 직접 개선할 가능성이 높은 것은 tool-call routing, MCP first-tool selection, terminal JSON action 안정성이다.
4. GLM FP8 direct training은 막혔으므로 BF16 원본을 4bit QLoRA로 로드해 adapter만 학습한다.
5. 바로 비교 가능한 공식 목표는 `MCP-Atlas (Public Set) 76.8 -> 78.0+`, `Tool-Decathlon 48.2 -> 52.8+`, `Terminal-Bench 2.1 (Terminus-2) 81.0 -> 82.0+`다.

### C. GLM BF16 QLoRA 튜닝 경로

FP8 LoRA long run은 시작하지 않는다. 학습은 다운로드 완료된 BF16 원본 `zai-org/GLM-5.2`를 QLoRA로 로드해서 진행한다.

현재 상태:

```bash
tmux ls
df -h /home/work/.data
.venvs/glm52-vllm-cu129-release-driver570/bin/python scripts/check_glm52_bf16_snapshot_ready_20260627.py \
  --model-id zai-org/GLM-5.2 \
  --cache-dir /home/work/.data/huggingface/hub
```

자동 학습 watcher:

```bash
tmux new-session -d -s fable_glm52_bf16_ready_then_qlora \
  "cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation && bash scripts/watch_glm52_bf16_ready_then_qlora_20260627.sh"
```

주의: watcher는 이미 한번 snapshot을 감지했고 smoke 실패와 함께 종료됐다. 이후 manual map 재시도에서도 22% load에서 GPU1 OOM이 확인됐으므로, 기본 자동 재실행은 막아두고 재현이 필요할 때만 명시적으로 실행한다.

다운로드 재시작이 필요하면:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
RUN_NOW=1 bash scripts/download_glm52_bf16_20260627.sh
```

QLoRA dry-run:

```bash
bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

다음 1-step smoke:

```bash
RUN_NOW=1 \
RUN_ID=20260627_glm52_bf16_qlora_smoke \
OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-smoke-20260627 \
MAX_STEPS=1 MAX_TRAIN_ROWS=2 MAX_SEQ_LENGTH=512 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

pilot 25 steps:

```bash
RUN_NOW=1 \
RUN_ID=20260627_glm52_bf16_qlora_pilot25 \
OUTPUT_DIR=/home/work/.data/harness1/models/GLM-5.2__Fable-OfficialAgentic-QLoRA-pilot25-20260627 \
MAX_STEPS=25 MAX_SEQ_LENGTH=1024 SAVE_STEPS=5 SAVE_TOTAL_LIMIT=2 \
bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh
```

long resumable run은 pilot step time을 확인한 뒤 시작한다.

```bash
tmux new-session -d -s fable_glm52_bf16_qlora_long \
  "cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation && RUN_NOW=1 bash scripts/run_glm52_bf16_qlora_device_map_20260627.sh"
```

## 중단/재개 명령

다운로드 중단:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
tmux kill-session -t fable_glm52_bf16_download 2>/dev/null || true
pgrep -af 'hf download zai-org/GLM-5.2|download_glm52_bf16_20260627.sh'
```

GLM smoke/training 중단:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
tmux kill-session -t fable_glm52_retry_train 2>/dev/null || true
tmux kill-session -t fable_glm52_device_map_smoke 2>/dev/null || true
tmux kill-session -t fable_glm52_bf16_ready_then_qlora 2>/dev/null || true
tmux kill-session -t fable_glm52_bf16_qlora_long 2>/dev/null || true
pgrep -af 'train_multifamily_chat_sft.py|torch.distributed.run'
pgrep -af 'train_glm52_fp8_device_map_lora.py'
```

필요한 PID만 골라서:

```bash
kill -TERM <pid>
sleep 5
kill -KILL <pid>
```

GPU 확인:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h
```

## Hugging Face 업로드

adapter가 생기면 다음 스크립트를 쓴다.

```bash
MODEL_DIR=/home/work/.data/harness1/models/<adapter-or-model-dir> \
REPO_ID=LLM-OS-Models/<repo-name> \
RUN_NOW=1 \
bash scripts/upload_adapter_to_hf_20260627.sh
```

`../.env`의 `HF_TOKEN`은 정상 파싱된다. 마지막 확인 identity는 user `gyung`, org `LLM-OS-Models` 접근 가능 상태였다.

## Git 상태

로컬 커밋:

- `eca48b2 Prepare GLM-5.2 FP8 Fable LoRA workflow`
- `5395f28 Fix GLM FP8 training compatibility`
- `513a7fa Document GLM FP8 status and vLLM setup`
- `8c1669d Use CUDA 12.9 vLLM wheel for GLM server`

push는 GitHub credential 문제로 실패했다. 현재 remote:

```text
origin https://github.com/gyunggyung/fable5-to-lfm.git
```

해결 방법:

- HTTPS PAT credential을 git credential helper에 넣거나
- `origin`을 등록된 SSH key가 있는 remote로 바꾸거나
- GitHub connector token을 다시 유효화한다.

그 뒤:

```bash
git push origin main
```
