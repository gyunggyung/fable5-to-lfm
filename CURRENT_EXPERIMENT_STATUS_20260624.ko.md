# 현재 실험 상태 (2026-06-24)

## 기준 벤치마크

현재 빠른 로컬 회귀 기준은 `TB2-lite replay`다.

- 데이터: `tb2_lite/data/replay_full.jsonl`
- 샘플: 303 terminal next-action steps
- 평가 스크립트: `scripts/replay_eval_vllm.py`, `scripts/replay_metrics.py`
- 점수: `next_action_score = 100 * (0.7 * avg_command_f1 + 0.3 * first_cmd_exact)`
- 주의: public Terminal-Bench 2.1 harness 전체 실행이 아니라, 터미널 다음 행동을 빠르게 비교하는 내부 replay benchmark다.

현재 1위 로컬 기준:

| 모델 | Score | Cmd F1 | First Cmd | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| Fabliq-8B-Agent-Reasoning | 51.59 | 0.5193 | 50.8% | 76.2% |

## 공식 자료 확인

- DiffusionGemma HF model card: 26B total, 약 4B active Gemma 4 MoE 기반 discrete diffusion 모델이며, Transformers와 vLLM 사용 예제가 있다.
- Google developer guide: DiffusionGemma는 vLLM OpenAI-compatible server로 서빙 가능하며 `diffusion_sampler=entropy_bound`, `diffusion_entropy_bound=0.1`, `diffusion_config.canvas_length=256` 설정을 권장한다.
- vLLM recipe/blog: DiffusionGemma는 dLLM 특성상 bidirectional attention, iterative refinement, block generation 경로가 필요하다.
- Qwen3.5-9B HF model card: Transformers/vLLM/SGLang/KTransformers 호환 weight로 공개되어 있다.
- Qwen3.5 SFT 관련 공개 자료: `messages` 컬럼을 그대로 넘기는 SFT보다 assistant-token loss masking을 명시적으로 하는 편이 안전하며, Qwen3.5 계열은 Transformers v5 계열과 bf16 LoRA 경로가 권장된다.
- Qwen3.5 vLLM recipe: tool calling에는 `--enable-auto-tool-choice --tool-call-parser qwen3_coder` 경로가 문서화되어 있다.

참고 링크:

- https://huggingface.co/google/diffusiongemma-26B-A4B-it
- https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it
- https://vllm-project.github.io/2026/06/10/diffusion-gemma.html
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://discuss.huggingface.co/t/is-this-a-common-reasonable-recipe-for-full-finetuning-qwen3-5-4b/174873
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html

## 완료된 작업

### GLM-5.2 chaser full SFT

- 스크립트: `scripts/run_glm52_chaser_mix_sft_20260624.sh`
- 데이터: `datasets/glm52_chaser_terminal_toolmix_20260624.jsonl` (11,416 rows)
- 출력 모델: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__Terminal-GLM52-Chaser-Mix-FullSFT-20260624`
- 상태: 학습 완료

Final model vLLM TB2-lite sharded 결과:

| 모델 | Score | Cmd F1 | First Cmd | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| GLM-5.2 chaser final | 51.13 | 0.5153 | 50.2% | 75.6% |
| GLM-5.2 chaser checkpoint-1400 | 50.56 | 0.5046 | 50.8% | 76.2% |

해석: 기존 1위 `Fabliq-8B-Agent-Reasoning` 51.59에 final은 0.46점 모자랐고, checkpoint-1400은 더 낮았다. 이 run 안에서는 1위를 넘는 checkpoint를 아직 찾지 못했다.

### Qwen3.5-9B vLLM fallback 평가

- 스크립트: `scripts/run_qwen35_9b_tb2_vllm_sharded_20260624.sh`
- 실행 설정: 8 shard, `MAX_MODEL_LEN=32768`, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=65536`
- 결과 파일: `benchmarks/20260624_qwen35_9b_tb2_vllm_sharded_32k/results/qwen35-9b-base-vllm.json`

| 모델 | Score | Cmd F1 | First Cmd | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B base vLLM | 36.75 | 0.4358 | 20.8% | 78.9% |

해석: vLLM 실행 경로는 성공했지만, base Qwen3.5-9B는 TB2-lite next-action에서는 Fabliq 51.59를 넘지 못했다.

### DiffusionGemma Docker-free 실행 경로

Docker daemon이 없어서 공식 Docker/vLLM image는 쓰지 않는다. 대신 별도 uv env를 만들었다.

- env: `fable_distillation/.venvs/diffusiongemma-transformers-cu128`
- setup: `scripts/setup_diffusiongemma_transformers_uv_20260624.sh`
- torch: `2.11.0+cu128`
- transformers: `5.12.1`
- 추가 의존성: `torchvision`, `pillow`
- loader 안정화: `HF_DEACTIVATE_ASYNC_LOAD=1`

DiffusionGemma 평가 스크립트에서 고친 문제:

- 기존 autoregressive 방식 `output[0][prompt_len:]` slicing은 DiffusionGemma에서 빈 출력을 만들 수 있었다.
- `processor.decode()`가 문자열 대신 list를 반환하는 케이스를 정규화했다.
- decoded output에 렌더링된 prompt가 포함되는 케이스를 텍스트 prefix 기준으로 제거했다.
- 1-sample smoke 결과: `valid_json=100%`, score `25.56`; 실제 assistant JSON만 평가되는 것을 확인했다.
- corrected full TB2-lite 결과: score `25.12`, Cmd F1 `0.2980`, First Cmd `14.2%`, Valid JSON `55.1%`.
- long-output/code/tool-call probe: `97.88 tok/s`.

결과 파일:

- `benchmarks/20260624_diffusiongemma_dllm_base_transformers_cu128_decodefix_full/results/diffusiongemma-26b-a4b-it-base.json`
- `benchmarks/20260624_diffusiongemma_dllm_base_transformers_cu128_decodefix_full/results/diffusiongemma-26b-a4b-it-base.probe.transformers.json`

| 모델 | Backend | Score | Cmd F1 | First Cmd | Valid JSON |
| --- | --- | ---: | ---: | ---: | ---: |
| DiffusionGemma 26B-A4B IT | Transformers dLLM | 25.12 | 0.2980 | 14.2% | 55.1% |

해석: Docker-free 실행과 dLLM 출력은 성공했지만, base DiffusionGemma는 TB2-lite terminal next-action에서 Fabliq 51.59와 Qwen3.5-9B 36.75보다 낮다. 이 모델은 그대로 next-action agent로 쓰기보다 SFT/LoRA 또는 prompt format tuning이 필요하다.

## 현재 진행 중

Qwen3.5-9B LoRA SFT 재시도가 실행 중이다.

초기 Qwen LoRA run은 모델/VRAM 문제가 아니라 Qwen chat template이 `user` 메시지가 없는 Fable terminal replay 샘플을 거부해서 실패했다. 전체 11,416 rows 중 2,062 rows가 `system -> assistant...` 형태라, 데이터를 버리지 않기 위해 `training/train_multifamily_chat_sft.py`에 `--chat-serialization simple-chatml` 경로를 추가했다. Qwen tokenizer는 `<|im_start|>`/`<|im_end|>`를 단일 special token으로 갖고 있어서 이 우회가 Qwen 계열에 적합하다.

추가로 `--tokenized-cache-dir`을 trainer에 넣었다. 직전 ChatML run은 패치 전 커맨드라 rank별 중복 tokenization을 했지만, 현재 `ddptrue` run부터는 `scripts/run_multifamily_sft_smoke_20260624.sh`가 tokenized cache를 공유해서 GPU idle startup 시간을 줄인다.

두 번째 Qwen ChatML run은 tokenization과 1 step 학습까지 갔지만, Qwen3.5-9B가 VLM 구조라 text-only batch에서는 일부 vision/비사용 모듈이 loss에 참여하지 않아 DDP가 `Expected to have finished reduction... unused parameters`로 중단했다. 이건 OOM이 아니라 DDP 설정 문제다. trainer에 `--ddp-find-unused-parameters true|false`를 추가했고, Qwen preset은 기본 `true`로 재시작한다.

```bash
RUN_NOW=1 \
SKIP_BUILD_DATASET=1 \
MODEL_PRESET=qwen35_9b \
RUN_ID=20260624_qwen35_9b_glm52_terminalmix_lora_sft300_chatml_ddptrue \
TRAIN_JSONL=/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets/glm52_chaser_terminal_toolmix_20260624.jsonl \
OUTPUT_DIR=/home/work/.data/harness1/models/Qwen3.5-9B__GLM52-TerminalMix-LoRA-SFT300-ChatML-DDPTrue-20260624 \
TOKENIZED_CACHE_DIR=/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/.cache/tokenized/qwen35_9b_glm52_terminalmix_6000_8192_chatml_seed52_v1 \
MAX_TRAIN_ROWS=6000 \
MAX_STEPS=300 \
LEARNING_RATE=2e-5 \
GRAD_ACCUM=4 \
LORA_RANK=64 \
LORA_ALPHA=128 \
MAX_SEQ_LENGTH=8192 \
SAVE_STEPS=100 \
CHAT_SERIALIZATION=simple-chatml \
DDP_FIND_UNUSED_PARAMETERS=true \
bash scripts/run_multifamily_sft_smoke_20260624.sh
```

예상 후속 절차:

- LoRA 완료 후 full HF model로 merge
- `scripts/run_qwen35_9b_tb2_vllm_sharded_20260624.sh`로 TB2-lite vLLM sharded 평가
- base Qwen3.5-9B 점수 `36.75`를 얼마나 끌어올리는지 확인

## 다음 판단

1. Qwen3.5-9B ChatML LoRA가 정상 학습되는지 확인한다.
2. 완료되면 adapter를 merge해서 vLLM 평가를 돌린다.
3. Qwen LoRA가 51.59에 접근하지 못하면 terminal/tool-call 데이터 비율, LR, epoch를 조정하거나 RL/GRPO 계열로 이어간다.
3. DiffusionGemma base는 낮으므로 dLLM은 LoRA/SFT나 prompt format tuning을 한 뒤 다시 본다.
4. DiffusionGemma dLLM은 실행 성공/성능 실패 상태이므로, 다음 dLLM 실험은 NeMo AutoModel LoRA/SFT 또는 prompt format tuning으로 좁힌다.
