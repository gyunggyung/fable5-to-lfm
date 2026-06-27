# GLM-5.2-FP8 Fable 스타일 튜닝/평가 상태 (2026-06-27)

작성 시각: 2026-06-27 02:18 UTC / 2026-06-27 11:18 KST

## 한줄 결론

GLM-5.2-FP8 다운로드와 Fable/Mythos 스타일 데이터 준비는 끝났다. 다만 현재 HF Trainer + DeepSpeed ZeRO-3 LoRA smoke는 8xH200 학습 경로로 안전하지 않다. 15% weight 로딩에서 GPU VRAM은 GPU당 약 1GB 그대로였고, rank별 CPU RSS가 150-179GB까지 증가해 CPU 복제 로딩으로 판단하고 OOM 전에 중단했다.

## 현재 상태

- 작업 루트: `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation`
- HF cache: `/home/work/.data/huggingface/hub`
- GLM snapshot: `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1`
- GLM cache size: 약 `707G`
- GPU 상태: GLM smoke는 중단됨. GPU compute app 없음. GPU 메모리는 1MiB 수준까지 회수됨.
- Git 상태: 로컬 `main`은 `origin/main`보다 2커밋 앞섬.
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
- `training/train_multifamily_chat_sft.py`

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

### 3차/최신 smoke 중단

DeepSpeed 설정과 `HfDeepSpeedConfig`를 넣고 다시 시작했지만, 다음 패턴이 나왔다.

- weight loading은 진행: 최종 확인 시 약 `15%`, `333/2160`
- GPU VRAM: GPU당 약 `1038 MiB`, utilization `0%`
- CPU RSS: rank별 약 `150-179GB`
- 시스템 RAM은 회수 가능했지만, 진행률 대비 RSS 증가가 커서 full load 시 OOM 위험이 높음

판단:

이 경로는 8xH200 tensor/ZeRO shard 학습이 아니라 rank별 CPU 복제 로딩에 가깝다. smoke 목적은 학습 가능 경로 확인이므로 OOM 전에 중단했다.

## 공식 근거

- GLM-5.2-FP8 모델 카드: https://huggingface.co/zai-org/GLM-5.2-FP8
- GLM-5.2 benchmark 섹션: https://huggingface.co/zai-org/GLM-5.2#benchmark
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- GLM-5.2 blog: https://huggingface.co/blog/zai-org/glm-52-blog

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
4. GLM 743B급 LoRA는 현재 HF Trainer path가 불안정하므로, full 모델 튜닝은 Megatron/DeepSpeed-MoE/Unsloth GLM 지원 여부를 따로 검증해야 한다.
5. 바로 성능 개선 실험이 필요하면 Qwen3.5-9B/Gemma 4 12B 계열에 같은 `official_agentic_sft_mix_20260627.jsonl`을 먼저 적용하고, vLLM 평가로 방법론을 검증한다.

### C. GLM 튜닝 경로 재검증

GLM 튜닝을 계속 시도하려면 다음을 먼저 확인한다.

- vLLM/Transformers 최신 GLM 지원 env와 training env를 분리한다.
- current HF Trainer + ZeRO-3가 FP8 quantized GLM을 model-load 단계에서 실제 shard하는지 확인하는 1-step smoke를 만든다.
- rank별 RSS가 진행률 20% 이전에 120GB를 넘으면 즉시 중단한다.
- GPU VRAM이 model-load 이후 GPU당 수십 GB 이상으로 올라가지 않으면 학습으로 판단하지 않는다.
- full run은 smoke가 `model loaded`, `LoRA/model trainable setup complete`, `trainer.train complete`를 통과한 뒤에만 시작한다.

## 중단/재개 명령

GLM smoke/training 중단:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/fable_distillation
tmux kill-session -t fable_glm52_retry_train 2>/dev/null || true
pgrep -af 'train_multifamily_chat_sft.py|torch.distributed.run'
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
