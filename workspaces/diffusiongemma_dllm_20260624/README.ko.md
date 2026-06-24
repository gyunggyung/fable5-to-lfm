# DiffusionGemma dLLM 작업 폴더 (2026-06-24)

## 목표

Docker 없이 `fable_distillation/.venvs/diffusiongemma-transformers-cu128` 가상환경 안에서 DiffusionGemma가 실제로 좋은지 확인한다.

## 실행 경로

- backend: Transformers
- model class: `DiffusionGemmaForBlockDiffusion`
- processor: `AutoProcessor`
- main runner: `../../scripts/run_diffusiongemma_dllm_eval_20260624.sh`
- model prefetch: `../../scripts/prefetch_diffusiongemma_hf_20260624.py`
- shard merge: `../../scripts/merge_diffusiongemma_transformers_shards.py`
- env setup: `../../scripts/setup_diffusiongemma_transformers_uv_20260624.sh`

## 기본 실행

```bash
RUN_NOW=1 BACKEND=transformers \
  bash fable_distillation/scripts/run_diffusiongemma_dllm_eval_20260624.sh
```

## 기본 병렬 설정

- `TRANSFORMERS_SHARD_COUNT=8`
- `TRANSFORMERS_GPUS=0,1,2,3,4,5,6,7`
- TB2-lite replay를 8개 shard로 나눠서 각 GPU에서 별도 프로세스로 실행
- shard 결과는 `benchmarks/20260624_diffusiongemma_dllm_base_transformers/results/` 아래에서 merge

## 현재 제약

- Docker daemon이 없어서 `vllm/vllm-openai:gemma`는 사용하지 않는다.
- PyPI의 최신 vLLM wheel은 `0.23.0`이고 DiffusionGemma vLLM model code가 없다.
- 해당 vLLM env의 `torch 2.11.0+cu130`은 현재 CUDA 12.9 드라이버에서 CUDA init에 실패했다.
- 현재 실행 env는 `torch 2.11.0+cu128`, `transformers 5.12.1`, `accelerate 1.14.0` 조합이다.
- `torchvision`, `pillow`가 필요해서 env setup에 포함했다.
- DiffusionGemma Transformers 재시도는 `HF_DEACTIVATE_ASYNC_LOAD=1`로 async loader를 끄고 진행한다.
- 따라서 지금은 vLLM 고속 serving 평가가 아니라 Transformers backend로 품질/속도 probe를 먼저 본다.

## 2026-06-24 디코드 수정

첫 8-GPU sharded TB2-lite run은 merge까지 성공했지만 score가 `0.0`이었다. 원인은 모델이 전부 틀린 것이 아니라 evaluator가 DiffusionGemma 출력을 autoregressive 모델처럼 `output[0][prompt_len:]`로 잘라 빈 문자열을 만든 것이다.

수정 사항:

- `output_ids`가 prompt prefix를 실제로 포함할 때만 token slicing
- `processor.decode()`가 list를 반환할 때 문자열로 정규화
- decoded output에 렌더링된 prompt가 포함될 때 prompt prefix와 `model` marker를 제거
- smoke3 1-sample 확인: `valid_json=100%`, score `25.56`

현재 full 재평가 run id:

- `20260624_diffusiongemma_dllm_base_transformers_cu128_decodefix_full`

결과:

- TB2-lite score: `25.12`
- Cmd F1: `0.2980`
- First Cmd: `14.2%`
- Valid JSON: `55.1%`
- probe: `97.88 tok/s`

판단: 실행 경로는 성공했지만, base DiffusionGemma는 현재 터미널 next-action benchmark에서는 바로 쓸 수준이 아니다. dLLM은 SFT/LoRA 또는 prompt format tuning 후 다시 평가한다.
