# DiffusionGemma dLLM 작업 폴더 (2026-06-24)

## 목표

Docker 없이 `fable_distillation/.venvs/diffusiongemma-vllm` 가상환경 안에서 DiffusionGemma가 실제로 좋은지 확인한다.

## 실행 경로

- backend: Transformers
- model class: `DiffusionGemmaForBlockDiffusion`
- processor: `AutoProcessor`
- main runner: `../../scripts/run_diffusiongemma_dllm_eval_20260624.sh`
- model prefetch: `../../scripts/prefetch_diffusiongemma_hf_20260624.py`
- shard merge: `../../scripts/merge_diffusiongemma_transformers_shards.py`

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
- 따라서 지금은 vLLM 고속 serving 평가가 아니라 Transformers backend로 품질/속도 probe를 먼저 본다.
