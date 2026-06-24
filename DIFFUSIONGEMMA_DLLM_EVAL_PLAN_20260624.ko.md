# DiffusionGemma dLLM 검증 계획 (2026-06-24)

## 질문

목표는 “DiffusionGemma를 한 번 돌려봤다”가 아니다. 보고 싶은 것은 **dLLM이 실제로 좋은가**이다.

따라서 판단 기준은 세 개로 둔다.

| 축 | 볼 것 | 이유 |
| --- | --- | --- |
| 품질 | TB2-lite next-action score, valid JSON, command F1 | 터미널 에이전트로 쓸 수 있는지 |
| 속도 | output tok/s, sec/prompt | dLLM의 핵심 주장인 빠른 block generation 확인 |
| 형태 안정성 | JSON/tool-call, 긴 코드, 긴 설명, code review 출력 | dLLM이 짧게 끝내거나 형식을 깨는지 확인 |

## 공식 자료에서 확인한 점

- Google model card: DiffusionGemma는 Gemma 4 26B-A4B MoE 기반의 discrete diffusion text model이고, 텍스트/이미지/비디오 입력을 받아 텍스트를 생성한다.
- NeMo AutoModel 문서: 26B total / 약 4B active DiffusionGemma에 대해 full SFT와 LoRA SFT를 지원한다.
- NeMo 학습 recipe: `DiffusionGemmaSFTRecipe`는 마지막 response turn을 canvas로 두고 single-turn SFT loss를 건다.
- vLLM recipe: DiffusionGemma는 block diffusion 특성 때문에 `max_num_seqs<=4`, `diffusion_config.canvas_length=256`, `diffusion_sampler=entropy_bound` 설정이 필요하다.
- vLLM blog: dLLM은 TTFT는 높아질 수 있지만, 256-token canvas를 병렬 denoising하므로 generation throughput이 높아지는 구조다.
- 최근 arXiv 분석은 DiffusionGemma의 commit order가 완전 병렬/완전 left-to-right 둘 다 아니며, structured JSON에서는 commit order 특성이 다르게 나타난다고 보고한다. 그래서 JSON/tool-call 안정성은 별도 축으로 본다.

참고:

- https://huggingface.co/google/diffusiongemma-26B-A4B-it
- https://docs.nvidia.com/nemo/automodel/nightly/guides/dllm/diffusiongemma.html
- https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it
- https://vllm.ai/blog/2026-06-10-diffusion-gemma
- https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- https://arxiv.org/abs/2606.14620

## 실행 순서 변경

post-chaser queue는 DiffusionGemma를 먼저 실행한다.

1. DiffusionGemma base Transformers dLLM 평가
2. DiffusionGemma LoRA smoke SFT
3. Gemma 4 12B IT smoke
4. Qwen3.5 9B smoke

## 새 스크립트

### Base dLLM Transformers 평가

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_diffusiongemma_dllm_eval_20260624.sh
```

이 스크립트는 Docker 없이 두 가지를 실행한다.

- TB2-lite replay: `scripts/diffusiongemma_transformers_eval.py`
- long-output/code/tool-call probe: `scripts/diffusiongemma_transformers_eval.py`

현재 로컬 PyPI index의 최신 vLLM wheel은 `0.23.0`이고, 이 wheel에는 DiffusionGemma vLLM model code가 없다. 공식 vLLM recipe는 DiffusionGemma용 `vllm/vllm-openai:gemma` image와 vLLM 0.24+ 계열을 전제로 한다.

현재 컨테이너에서는 Docker CLI는 있지만 daemon이 떠 있지 않다. `docker pull vllm/vllm-openai:gemma`는 `Cannot connect to the Docker daemon`으로 실패했다. 따라서 이 환경에서는 Docker를 사용하지 않는다.

기본 runner는 `BACKEND=transformers`로 고정했다. 즉 post-chaser 자동 실행은 `fable_distillation/.venvs/diffusiongemma-transformers-cu128` 가상환경 안에서 HF `DiffusionGemmaForBlockDiffusion` 경로로 간다.

실패/수정 기록:

- `fable_distillation/.venvs/diffusiongemma-vllm`의 vLLM 0.23 wheel은 `torch 2.11.0+cu130`을 끌고 왔다.
- 현재 드라이버는 CUDA 12.9 계열이라 해당 torch가 `The NVIDIA driver on your system is too old`로 CUDA를 못 켰다.
- 그래서 Docker도 vLLM wheel도 쓰지 않고, 별도 `torch 2.11.0+cu128` Transformers env로 분리했다.
- cu128 env에서는 CUDA 자체는 잡혔지만 Transformers 5.12 async loader가 worker thread에서 `tensor.to(cuda)`를 하다가 `CUDA driver error: initialization error`가 났다.
- DiffusionGemma 재시도는 `HF_DEACTIVATE_ASYNC_LOAD=1`로 async weight loading을 끄고 진행한다.
- `AutoProcessor` 로드에는 `pillow`, multimodal processor 초기화에는 `torchvision`이 필요해서 setup script에 추가했다.
- 첫 full sharded run은 실행/merge는 성공했지만 score가 `0.0`이었다. 원인은 모델 품질이 아니라 evaluator decode bug였다.
- DiffusionGemma `generate()` 출력은 일반 autoregressive 모델처럼 항상 prompt prefix + new tokens 형태로 안정적으로 오지 않는다. 기존 `output[0][prompt_len:]` slicing이 빈 문자열을 만들었다.
- 이후 `processor.decode()`가 list를 반환하는 케이스를 문자열로 정규화했고, decoded output에 렌더링된 prompt가 포함되는 케이스를 텍스트 prefix와 `model` marker 기준으로 제거했다.
- decode fix smoke: 1 sample에서 `valid_json=100%`, score `25.56`으로 실제 assistant JSON만 평가되는 것을 확인했다.
- corrected full run: `20260624_diffusiongemma_dllm_base_transformers_cu128_decodefix_full`.
- corrected full result: score `25.12`, Cmd F1 `0.2980`, First Cmd `14.2%`, Valid JSON `55.1%`.
- probe result: `97.88 tok/s`.

해석: Docker-free dLLM 실행과 출력 안정화는 성공했다. 다만 base DiffusionGemma는 TB2-lite 터미널 next-action에서는 Fabliq 51.59와 Qwen3.5-9B 36.75를 넘지 못했다. 다음 dLLM 단계는 base 재평가가 아니라 LoRA/SFT 또는 prompt format tuning이다.

TB2-lite는 기본적으로 8개 shard로 나누어 실행한다.

- `TRANSFORMERS_SHARD_COUNT=8`
- `TRANSFORMERS_GPUS=0,1,2,3,4,5,6,7`
- 각 shard는 자기 GPU에 DiffusionGemma를 한 벌씩 올린다.
- 완료 후 `scripts/merge_diffusiongemma_transformers_shards.py`가 shard 결과를 하나의 JSON으로 합친다.

vLLM/OpenAI-compatible evaluator는 남겨 두되, 이 환경의 기본 자동 큐에서는 사용하지 않는다.

- `scripts/replay_eval_openai_chat.py`
- `scripts/openai_prompt_probe.py`

DiffusionGemma vLLM을 나중에 Python wheel로 직접 쓸 수 있게 되면 적용할 핵심 설정:

```json
{
  "hf_overrides": {
    "diffusion_sampler": "entropy_bound",
    "diffusion_entropy_bound": 0.1
  },
  "engine_kwargs": {
    "diffusion_config": {
      "canvas_length": 256
    }
  },
  "max_num_seqs": 4,
  "gpu_memory_utilization": 0.85
}
```

### Optional isolated Python env

```bash
bash fable_distillation/scripts/setup_diffusiongemma_transformers_uv_20260624.sh
```

DiffusionGemma용 Python 환경은 `uv`로 `fable_distillation/.venvs/diffusiongemma-transformers-cu128`에 만든다. 이 환경은 Docker 없이 Transformers backend 실행에 사용한다.

## 해석 기준

dLLM이 “좋다”고 말하려면 최소한 아래 중 하나는 만족해야 한다.

- TB2-lite 품질이 Fabliq 상위권에 근접하면서 long-output tok/s가 확실히 높다.
- TB2-lite는 낮더라도 긴 코드/긴 설명/구조화 JSON에서 throughput과 완성도가 뚜렷하게 좋다.
- SFT 후 command F1/valid JSON이 올라가면서 dLLM speed 장점이 유지된다.

반대로 아래면 “현 터미널 에이전트 목적에는 아직 별로”라고 본다.

- output tok/s는 빠르지만 JSON/tool-call이 무너진다.
- 긴 생성은 빠른데 next-action score가 base LFM/Fabliq보다 크게 낮다.
- LoRA SFT 후 형식 안정성은 오르지만 속도나 품질 중 하나가 크게 깨진다.
