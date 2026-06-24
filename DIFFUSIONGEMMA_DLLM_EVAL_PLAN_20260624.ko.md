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

1. DiffusionGemma base vLLM 평가
2. DiffusionGemma LoRA smoke SFT
3. Gemma 4 12B IT smoke
4. Qwen3.5 9B smoke

## 새 스크립트

### Base dLLM vLLM 평가

```bash
RUN_NOW=1 bash fable_distillation/scripts/run_diffusiongemma_dllm_eval_20260624.sh
```

이 스크립트는 두 가지를 실행한다.

- Docker/OpenAI backend: `scripts/replay_eval_openai_chat.py`, `scripts/openai_prompt_probe.py`
- Offline vLLM backend: `scripts/replay_eval_vllm.py`, `scripts/vllm_prompt_probe.py`

현재 로컬 PyPI index의 최신 vLLM wheel은 `0.23.0`이고, 이 wheel에는 DiffusionGemma vLLM model code가 없다. 따라서 실제 vLLM 고속 검증은 공식 recipe가 지정한 `vllm/vllm-openai:gemma` Docker image를 우선 사용한다.

현재 컨테이너에서는 Docker CLI는 있지만 daemon이 떠 있지 않다. `docker pull vllm/vllm-openai:gemma`는 `Cannot connect to the Docker daemon`으로 실패했다. 그래서 자동 runner는 `BACKEND=auto`일 때 다음 순서로 간다.

1. Docker daemon 사용 가능: `vllm/vllm-openai:gemma` + OpenAI-compatible evaluator
2. Docker daemon 사용 불가: HF `DiffusionGemmaForBlockDiffusion` Transformers fallback

Docker server script:

```bash
bash fable_distillation/scripts/run_diffusiongemma_vllm_docker_server_20260624.sh
```

OpenAI-compatible evaluator:

- `scripts/replay_eval_openai_chat.py`
- `scripts/openai_prompt_probe.py`

DiffusionGemma vLLM 핵심 설정:

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

### Optional isolated vLLM env

```bash
bash fable_distillation/scripts/setup_diffusiongemma_vllm_uv_20260624.sh
```

기본은 기존 `.vllm-lfm-cu12`를 쓰지만, DiffusionGemma 지원이 부족하면 `uv`로 `fable_distillation/.venvs/diffusiongemma-vllm`를 만든다.

## 해석 기준

dLLM이 “좋다”고 말하려면 최소한 아래 중 하나는 만족해야 한다.

- TB2-lite 품질이 Fabliq 상위권에 근접하면서 long-output tok/s가 확실히 높다.
- TB2-lite는 낮더라도 긴 코드/긴 설명/구조화 JSON에서 throughput과 완성도가 뚜렷하게 좋다.
- SFT 후 command F1/valid JSON이 올라가면서 dLLM speed 장점이 유지된다.

반대로 아래면 “현 터미널 에이전트 목적에는 아직 별로”라고 본다.

- output tok/s는 빠르지만 JSON/tool-call이 무너진다.
- 긴 생성은 빠른데 next-action score가 base LFM/Fabliq보다 크게 낮다.
- LoRA SFT 후 형식 안정성은 오르지만 속도나 품질 중 하나가 크게 깨진다.
